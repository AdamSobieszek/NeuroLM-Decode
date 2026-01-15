import torch
import torch.nn as nn
import torch.fft
import inspect # For optimizer configuration
import numpy as np
# Use relative imports if these files are in the same package/directory
from .lblm_input_processor import LBLMInputProcessor
from .lgtransformer import LBLMConformerBackbone
from .lblm_prediction_heads import SpectroTemporalPredictionHeads
from .lblm_reguralization_heads import ThinkingOfLatents as WordPredictionHead


from .lblm_loss import HuberLoss
from .lamb_optimizer import Lamb

class InnerSpeech_LBLM_MSTP(nn.Module):
    def __init__(self,
                 patch_length=25,
                 patch_stride=6,
                 num_subjects=12,
                 subject_embed_dim=1,
                 mask_ratio=0.10,
                 use_rev_in=True,
                 eeg_channels=62,
                 conformer_input_dim=64,
                 conformer_num_layers=4,
                 conformer_d_model=64,
                 conformer_n_head=8,
                 conformer_d_ff=256,
                 conformer_dropout_rate=0.1,
                 conformer_conv_kernel_size=31,
                 huber_delta=1.0,
                 lambda_amplitude=0.5,
                 lambda_phase=0.1,
                 lambda_word=0.0022
                 ):
        super().__init__()

        self.patch_length = patch_length
        self.fft_target_dim = patch_length // 2 + 1
        self.eeg_channels = eeg_channels

        self.input_processor = LBLMInputProcessor(
            patch_length=patch_length,
            patch_stride=patch_stride,
            embed_dim=conformer_input_dim,
            num_subjects=num_subjects,
            subject_embed_dim=subject_embed_dim,
            mask_ratio=mask_ratio,
            use_rev_in=use_rev_in,
            eeg_channels=eeg_channels
        )

        self.conformer_backbone = LBLMConformerBackbone(
            input_dim=conformer_input_dim,
            num_layers=conformer_num_layers,
            d_model=conformer_d_model,
            n_head=conformer_n_head,
            d_ff=conformer_d_ff,
            conv_kernel_size=conformer_conv_kernel_size,
            dropout_rate=conformer_dropout_rate
        )

        self.prediction_heads = SpectroTemporalPredictionHeads(
            d_model=conformer_d_model,
            patch_length=patch_length
        )

        self.word_prediction_head = WordPredictionHead(
            hidden_dim=conformer_d_model,
            n_classes=13
        )

        self.loss_fn_huber = HuberLoss(delta=huber_delta)
        self.loss_fn_ce = nn.CrossEntropyLoss(reduction='none')
        self.lambda_amplitude = lambda_amplitude
        self.lambda_phase = lambda_phase
        self.example_masked_indices_bool = None
        self.lambda_word = lambda_word


    def _calculate_fft_targets(self, original_patches_masked):
        # original_patches_masked: [TotalNumberOfMaskedPatches, PatchLength]
        if original_patches_masked.numel() == 0:
            # Return empty tensors with correct trailing dimensions if no patches
            return (torch.empty((0, self.fft_target_dim), device=original_patches_masked.device),
                    torch.empty((0, self.fft_target_dim), device=original_patches_masked.device))
        with torch.no_grad():
            fft_result = torch.fft.rfft(original_patches_masked.float(), dim=-1, norm="ortho") # Using ortho norm
            amplitude = torch.abs(fft_result).to(dtype=original_patches_masked.dtype)
            phase = torch.angle(fft_result.cpu()).to(dtype=original_patches_masked.dtype).to(original_patches_masked.device)
        return amplitude, phase

    def forward(self, x_raw_eeg, subject_ids=None, input_mask_override=None, targets=None):
        # x_raw_eeg: [batch_size, eeg_channels, num_timepoints]
        
        batch_size_orig, num_input_channels, _ = x_raw_eeg.shape

        # 1. Process Input
        processed_input = self.input_processor(x_raw_eeg.contiguous(), subject_ids.contiguous())
        embedded_patches, original_patches_for_target, masked_indices_bool, rev_in_mean, rev_in_std = processed_input
        # embedded_patches: [B*M, NumPatches, embed_dim]
        # original_patches_for_target: [B, M, NumPatches, PatchLength] (original scale)
        # masked_indices_bool: [B*M, NumPatches] (True where masked by input_processor)

        num_bm, num_patches_seq, _ = embedded_patches.shape
        
        # Override mask if provided
        if input_mask_override is not None:
            if input_mask_override.shape[0] == batch_size_orig and input_mask_override.dim() == 2 and \
               input_mask_override.shape[1] == num_bm * num_patches_seq / batch_size_orig : # [B, M*NumPatches]
                masked_indices_bool = input_mask_override.reshape(num_bm, num_patches_seq).bool()
            elif input_mask_override.shape == (num_bm, num_patches_seq): # [B*M, NumPatches]
                 masked_indices_bool = input_mask_override.bool()
            else:
                raise ValueError(f"input_mask_override shape {input_mask_override.shape} incompatible.")



        # Prepare input for Conformer: zero out content of masked tokens
        conformer_input = embedded_patches.clone()
        # masked_indices_bool is [B*M, NumPatches]. Need to expand for embedding dim.
        conformer_input[masked_indices_bool] = 0.0 

        # 2. Pass through Conformer Backbone
        # MHA key_padding_mask: True for positions to *ignore*.
        backbone_output = self.conformer_backbone(
            conformer_input,
            attention_mask=masked_indices_bool # True for masked tokens
        )
        # backbone_output: [B*M, NumPatches, conformer_d_model]

        # 3. Select only the output tokens corresponding to the *masked* input patches
        masked_backbone_outputs = backbone_output[masked_indices_bool]
        # masked_backbone_outputs: [TotalNumberOfMaskedPatches, conformer_d_model]

        log = {}
        if masked_backbone_outputs.shape[0] == 0: # No patches were masked
            dummy_loss = torch.tensor(0.0, device=x_raw_eeg.device, requires_grad=True)
            log.update({
                'mstp/loss_wave': torch.tensor(0.0, device=x_raw_eeg.device),
                'mstp/loss_amplitude': torch.tensor(0.0, device=x_raw_eeg.device),
                'mstp/loss_phase': torch.tensor(0.0, device=x_raw_eeg.device),
                'mstp/total_loss': dummy_loss.detach()
            })
            return dummy_loss, None, log

        # 4. Predict Wave, Amplitude, Phase for masked tokens
        pred_wave, pred_amplitude, pred_phase = self.prediction_heads(masked_backbone_outputs)
        # pred_wave: [TotalMasked, patch_length]

        # 5. Prepare Targets for the masked patches (from original_patches_for_target)
        # original_patches_for_target is [B, M, NumPatches, PatchLength] (original scale, before RevIN if used)
        flat_original_patches = original_patches_for_target.reshape(num_bm, num_patches_seq, self.patch_length)
        target_wave_patches = flat_original_patches[masked_indices_bool] # [TotalMasked, patch_length]

        # Denormalize predictions if RevIN was used by input_processor
        if self.input_processor.use_rev_in and rev_in_mean is not None and rev_in_std is not None:
            # rev_in_mean/std were [B, M, NumPatches, 1]
            flat_rev_in_mean = rev_in_mean.reshape(num_bm, num_patches_seq, 1)
            flat_rev_in_std = rev_in_std.reshape(num_bm, num_patches_seq, 1)
            
            masked_means = flat_rev_in_mean[masked_indices_bool] # [TotalMasked, 1]
            masked_stds = flat_rev_in_std[masked_indices_bool]   # [TotalMasked, 1]
            
            # Ensure pred_wave has a shape compatible with broadcasting for denorm
            # pred_wave is [TotalMasked, patch_length], masked_means/stds are [TotalMasked, 1]
            pred_wave = self.input_processor.rev_in_denorm(pred_wave, masked_means, masked_stds)

    
        target_amplitude, target_phase = self._calculate_fft_targets(target_wave_patches)

        # 6. Calculate Losses
        loss_wave = self.loss_fn_huber(pred_wave, target_wave_patches)
        loss_amplitude = self.loss_fn_huber(pred_amplitude, target_amplitude)
        loss_phase = self.loss_fn_huber(pred_phase[:,:8], target_phase[:,:8])


        do_mean_regularization = True
        if do_mean_regularization:
            erp_original_patches_for_target = original_patches_for_target.mean(dim=1) # [B, M, NumPatches, PatchLength]
            erp_pred_wave, erp_pred_amplitude, erp_pred_phase = self.prediction_heads(backbone_output.reshape(batch_size_orig, num_input_channels, num_patches_seq, backbone_output.shape[-1]).mean(dim=1))
            
            erp_pred_wave = self.input_processor.rev_in_denorm(erp_pred_wave, 
                                                                erp_original_patches_for_target.mean(dim=-1, keepdim=True), 
                                                                erp_original_patches_for_target.std(dim=-1, keepdim=True))
            
            erp_target_amplitude, erp_target_phase = self._calculate_fft_targets(erp_original_patches_for_target)
            loss_erp_wave = self.loss_fn_huber(erp_pred_wave, erp_original_patches_for_target)
            loss_erp_amplitude = self.loss_fn_huber(erp_pred_amplitude, erp_target_amplitude)
            loss_erp_phase = self.loss_fn_huber(erp_pred_phase[:,:8], erp_target_phase[:,:8])
            loss_erp_total = loss_erp_wave + self.lambda_amplitude * loss_erp_amplitude + self.lambda_phase * loss_erp_phase
        else:
            loss_erp_total = 0.0
            loss_erp_wave = 0.0
            loss_erp_amplitude = 0.0
            loss_erp_phase = 0.0

        if targets is not None:
            # Block 1: Word prediction head
            backbone_output_detached = backbone_output.reshape(batch_size_orig, num_input_channels, num_patches_seq, backbone_output.shape[-1])
            
            # backbone_output_detached = backbone_output_detached.detach()
            word_pred_orig = self.word_prediction_head(backbone_output_detached)
            word_pred = word_pred_orig.clone()
            channel_mean_word_pred = self.word_prediction_head(backbone_output_detached.mean(dim=1))
            total_mean_word_pred = self.word_prediction_head(backbone_output_detached.reshape(batch_size_orig, -1, backbone_output_detached.shape[-1]).mean(dim=1))
            
            word_pred = word_pred.reshape(-1, word_pred.shape[-1])
            channel_mean_word_pred = channel_mean_word_pred[targets.view(-1) != -1].reshape(-1, channel_mean_word_pred.shape[-1])
            total_mean_word_pred = total_mean_word_pred[targets.view(-1) != -1].reshape(-1, total_mean_word_pred.shape[-1])
            y_one_hot = torch.zeros_like(word_pred, device=x_raw_eeg.device)
            valid_batch_size_orig = sum(targets != -1)
            # y_one_hot_channel_mean = torch.zeros_like(channel_mean_word_pred, device=x_raw_eeg.device)
            # y_one_hot_total_mean = torch.zeros_like(total_mean_word_pred, device=x_raw_eeg.device)
                    
            # Block 2: Target reshaping and masking
            initial_word_targets = targets
            _batch_targets_for_onehot = initial_word_targets
            if _batch_targets_for_onehot.ndim > 1: # Ensure 1D: (batch_size_orig,)
                _batch_targets_for_onehot = _batch_targets_for_onehot.squeeze(-1)
            if _batch_targets_for_onehot.ndim == 0: # Handle batch_size_orig=1 case after squeeze
                 _batch_targets_for_onehot = _batch_targets_for_onehot.unsqueeze(0)

            y_one_hot_batch_level = torch.zeros(
                batch_size_orig, 
                self.word_prediction_head.n_classes, 
                device=x_raw_eeg.device,
                dtype=torch.float # Match dtype of y_one_hot for consistency
            )
            
            # Create mask for valid targets (not -1)
            _valid_batch_targets_mask = _batch_targets_for_onehot != -1
            
            # Get row indices for valid targets
            _batch_row_indices = torch.arange(batch_size_orig, device=x_raw_eeg.device)[_valid_batch_targets_mask]
            # Get corresponding target values (class indices)
            _filtered_batch_target_values = _batch_targets_for_onehot[_valid_batch_targets_mask].long()
            
            if _batch_row_indices.numel() > 0: # If there are any valid targets
                # Filter out target values that are out of bounds for the number of classes
                _in_bounds_mask = (_filtered_batch_target_values >= 0) & \
                                  (_filtered_batch_target_values < self.word_prediction_head.n_classes)
                
                _batch_row_indices_final = _batch_row_indices[_in_bounds_mask]
                _filtered_batch_target_values_final = _filtered_batch_target_values[_in_bounds_mask]
                
                if _batch_row_indices_final.numel() > 0:
                     y_one_hot_batch_level[_batch_row_indices_final, _filtered_batch_target_values_final] = 1.0
            
            # Set up one-hot targets for channel_mean and total_mean predictions
            y_one_hot_channel_mean = y_one_hot_batch_level[targets != -1].repeat_interleave(num_patches_seq, dim=0)
            y_one_hot_total_mean = y_one_hot_batch_level[targets != -1]
   
            if initial_word_targets.ndim == 1:
                expanded_targets_per_item = initial_word_targets.unsqueeze(1)
            else:
                expanded_targets_per_item = initial_word_targets
            repetition_factor = num_bm // batch_size_orig
            # Shape: (batch_size_orig, 1) -> (num_bm, 1)
            targets_repeated_bm = expanded_targets_per_item.repeat_interleave(repetition_factor, dim=0)
            targets_bm_seq = targets_repeated_bm.expand(-1, num_patches_seq)
            unmasked_word_targets = targets_bm_seq.reshape(-1)
            num_unmasked_elements = unmasked_word_targets.size(0)
                
            if num_unmasked_elements > 0:
                # Create a mask for valid targets (not equal to -1, which often denotes padding/ignore)
                valid_target_mask = unmasked_word_targets != -1
                
                # Get row indices for y_one_hot (0 to num_unmasked_elements - 1)
                row_indices = torch.arange(num_unmasked_elements, device=unmasked_word_targets.device)

                # Filter row indices and target values based on the valid_target_mask
                filtered_row_indices = row_indices[valid_target_mask]
                # Target values (class indices) must be long type for indexing
                filtered_target_values = unmasked_word_targets[valid_target_mask].long()

                # Populate y_one_hot: for each valid target, set the corresponding class index to 1.0
                # This check ensures we only attempt to index if there are valid targets.
                if filtered_row_indices.numel() > 0:
                    y_one_hot[filtered_row_indices, filtered_target_values] = 1.0
            # targets = unmasked_word_targets 

            # Block 3: Loss and accuracy computation
            # 1. Per-token level (word_pred)
            y_one_hot_filtered = y_one_hot[unmasked_word_targets != -1]
            word_pred_filtered = word_pred[unmasked_word_targets != -1]
            loss_word = self.loss_fn_ce(word_pred_filtered, y_one_hot_filtered)

            loss_word = loss_word.mean()
            
            accuracy = (word_pred_filtered.argmax(dim=1) == y_one_hot_filtered.argmax(dim=1)).float()
            
            # 2. Channel-mean level (channel_mean_word_pred)
            loss_channel_mean = self.loss_fn_ce(channel_mean_word_pred, y_one_hot_channel_mean)
            loss_channel_mean = loss_channel_mean.mean()
            mean_channel_accuracy = (channel_mean_word_pred.argmax(dim=-1) == y_one_hot_channel_mean.argmax(dim=-1)).float()
            
            # 3. Total-mean level (total_mean_word_pred)
            loss_total_mean = self.loss_fn_ce(total_mean_word_pred, y_one_hot_total_mean)
            loss_total_mean = loss_total_mean.mean()
            total_mean_accuracy = (total_mean_word_pred.argmax(dim=-1) == y_one_hot_total_mean.argmax(dim=-1)).float()
            
            # 4. Averaged softmax prediction accuracy
            averaged_accuracy = (channel_mean_word_pred.reshape(valid_batch_size_orig, num_patches_seq, -1).float().mean(dim=(1)).argmax(dim=-1) == y_one_hot_total_mean.argmax(dim=-1))
            averaged_accuracy2 = (word_pred_filtered.reshape(valid_batch_size_orig, num_input_channels, num_patches_seq, -1).float().mean(dim=(1,2)).argmax(dim=-1) == y_one_hot_total_mean.argmax(dim=-1))


            # 5. Optimistic accuracy (any of the methods gets it right)
            optimistic_accuracy = ((averaged_accuracy.float()+ averaged_accuracy2.float() + total_mean_accuracy.float()) >= 1).float()
            valid_targets = targets[targets != -1]
            target_labels = sorted(list(set(valid_targets.cpu().numpy())))
            optimistic_accuracy = optimistic_accuracy.view(-1)
            accuracy_detection = []
            accuracy_means_updown = []
            accuracy_means_our = []
            accuracy_means_tol = []
            for t in target_labels:
                if 0<t<5:
                    # accuracy_detection.append(optimistic_accuracy[valid_targets == t].float())
                    accuracy_means_our.append(optimistic_accuracy[valid_targets == t].float().mean())
                elif t<9:
                    print(optimistic_accuracy[valid_targets == t].float())
                    accuracy_means_updown.append(optimistic_accuracy[valid_targets == t].float().mean())
                else:
                    accuracy_means_tol.append(total_mean_accuracy[valid_targets == t].float().mean())
            balanced_accuracy_our = sum(accuracy_means_our) / len(accuracy_means_our)
            balanced_accuracy_tol = sum(accuracy_means_tol) / len(accuracy_means_tol) if accuracy_means_tol else 0.0
            balanced_accuracy_updown = sum(accuracy_means_updown) / len(accuracy_means_updown) if accuracy_means_updown else 0.0
            optimistic_accuracy = optimistic_accuracy.float().mean()
            # Convert to scalar values
            averaged_accuracy = averaged_accuracy.float().mean()
            averaged_accuracy2 = averaged_accuracy2.float().mean()
            total_mean_accuracy = total_mean_accuracy.float().mean()
            mean_channel_accuracy = mean_channel_accuracy.float().mean()
            optimistic_accuracy = optimistic_accuracy.float().mean()
            accuracy = accuracy.float().mean()


            # Combine word-level losses for the total loss calculation later
            loss_whole_word = loss_channel_mean + loss_total_mean
            correlation_loss = self.word_prediction_head.minimize_weight_correlation_loss(1)
        else:
            optimistic_accuracy = 0.0
            averaged_accuracy = 0.0
            averaged_accuracy2 = 0.0
            mean_channel_accuracy = 0.0
            total_mean_accuracy = 0.0
            loss_whole_word = 0.0
            loss_word = 0.0
            loss_channel_mean = 0.0
            loss_total_mean = 0.0
            accuracy = 0.0
            correlation_loss = 0.0
            balanced_accuracy_our = 0.0
            balanced_accuracy_tol = 0.0
            balanced_accuracy_updown = 0.0


        total_loss = loss_wave + self.lambda_amplitude * loss_amplitude + self.lambda_phase * loss_phase + self.lambda_word * (loss_total_mean + loss_channel_mean ) + loss_erp_total/5 + correlation_loss
        
        detach = lambda x: x.detach() if isinstance(x, torch.Tensor) else x
        log.update({
            'mstp/loss_wave': detach(loss_wave),
            'mstp/loss_amplitude': detach(loss_amplitude),
            'mstp/loss_phase': detach(loss_phase),
            'mstp/total_loss': detach(total_loss)})

        if do_mean_regularization:
            log.update({
            'mstp/loss_erp_total': detach(loss_erp_total),
            'mstp/loss_erp_wave': detach(loss_erp_wave),
            'mstp/loss_erp_amplitude': detach(loss_erp_amplitude),
            'mstp/loss_erp_phase': detach(loss_erp_phase)
        })
        if targets is not None and loss_word > 0.0:
            log.update({
            'mstp/accuracy': detach(accuracy),
            'mstp/averaged_accuracy': detach(averaged_accuracy),
            'mstp/mean_channel_accuracy': detach(mean_channel_accuracy),
            'mstp/optimistic_accuracy': detach(optimistic_accuracy),
            'mstp/loss_word': detach(loss_word),
            'mstp/loss_channel_mean': detach(loss_channel_mean),
            'mstp/loss_total_mean': detach(loss_total_mean),
            'mstp/correlation_loss': detach(correlation_loss),
            'mstp/balanced_accuracy_our': detach(balanced_accuracy_our),
            'mstp/balanced_accuracy_tol': detach(balanced_accuracy_tol),
            'mstp/balanced_accuracy_updown': detach(balanced_accuracy_updown),
            'mstp/total_loss': detach(total_loss) -  self.lambda_word *(loss_total_mean + loss_channel_mean) - correlation_loss
        })
        else:
            if 'mstp/accuracy' in log: log.pop('mstp/accuracy')
            if 'mstp/averaged_accuracy' in log: log.pop('mstp/averaged_accuracy')
            if 'mstp/mean_channel_accuracy' in log: log.pop('mstp/mean_channel_accuracy')
            if 'mstp/optimistic_accuracy' in log: log.pop('mstp/optimistic_accuracy')
            if 'mstp/loss_word' in log: log.pop('mstp/loss_word')
            if 'mstp/loss_channel_mean' in log: log.pop('mstp/loss_channel_mean')
            if 'mstp/loss_total_mean' in log: log.pop('mstp/loss_total_mean')
            if 'mstp/correlation_loss' in log: log.pop('mstp/correlation_loss')
            if 'mstp/balanced_accuracy_our' in log: log.pop('mstp/balanced_accuracy_our')
            if 'mstp/balanced_accuracy_tol' in log: log.pop('mstp/balanced_accuracy_tol')
            if 'mstp/balanced_accuracy_updown' in log: log.pop('mstp/balanced_accuracy_updown')

        return total_loss, (backbone_output, channel_mean_word_pred.argmax(dim=-1)-1), log

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type='cpu', optimizer_type='adamw', eps=1e-6, **kwargs):
        import inspect

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate}
        ]

        fused_available = False
        if hasattr(torch.optim, 'AdamW'):
            adam_sig = inspect.signature(torch.optim.AdamW)
            fused_available = 'fused' in adam_sig.parameters

        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        if optimizer_type.lower() == 'lamb':
            print(f"Configuring bitsandbytes LAMB optimizer with LR={learning_rate}, WD={weight_decay}, Betas={betas}, Eps={eps}")
            try:
                from bitsandbytes.optim import LAMB
            except ImportError:
                raise ImportError("bitsandbytes is required for LAMB optimizer. Please install with `pip install bitsandbytes`.")
            optimizer = LAMB(
                optim_groups,
                lr=learning_rate,
                betas=betas,
                weight_decay=weight_decay,
                eps=eps
            )
        else:
            optimizer = torch.optim.AdamW(optim_groups, betas=betas, **extra_args)
        return optimizer

    # def log_plot(self, log, plot_name):

# Example usage (for testing if run directly)
if __name__ == '__main__':
    print("Testing InnerSpeech_LBLM_MSTP model structure...")
    
    # Define hyperparameters
    model_params = {
        "patch_length": 25,
        "patch_stride": 6,
        "num_subjects": 12, 
        "subject_embed_dim": 1,
        "mask_ratio": 0.15, 
        "use_rev_in": True,
        "eeg_channels": 3, # Smaller for quick test

        "conformer_input_dim": 64, 
        "conformer_num_layers": 2, # Fewer layers for quick test
        "conformer_d_model": 64,
        "conformer_n_head": 4, 
        "conformer_d_ff": 128, 
        "conformer_dropout_rate": 0.1,
        "conformer_conv_kernel_size": 7, # Smaller kernel for test

        "huber_delta": 1.0,
        "lambda_amplitude": 0.1,
        "lambda_phase": 0.1
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = InnerSpeech_LBLM_MSTP(**model_params).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # Example dummy input
    batch_size = 2
    num_channels = model_params["eeg_channels"]
    # Timepoints to allow for a few patches: P + (N-1)*S = 25 + (4-1)*6 = 25 + 18 = 43
    timepoints = 50 
    dummy_eeg = torch.randn(batch_size, num_channels, timepoints).to(device)
    dummy_subject_ids = None
    if model_params["num_subjects"] > 0:
        dummy_subject_ids = torch.randint(0, model_params["num_subjects"], (batch_size,)).to(device)

    print(f"Dummy EEG shape: {dummy_eeg.shape}")
    if dummy_subject_ids is not None:
        print(f"Dummy subject IDs shape: {dummy_subject_ids.shape}")

    # Test forward pass
    try:
        loss, _, log_output = model(dummy_eeg, subject_ids=dummy_subject_ids)
        print("Forward pass successful.")
        print("Loss:", loss.item())
        print("Log:", log_output)

        # Test backward pass
        if loss.requires_grad:
            loss.backward()
            print("Backward pass successful.")
        else:
            print("Loss does not require grad (e.g. no masked tokens). Skipping backward pass.")
        
        # Test optimizer configuration
        optimizer_params = {
            "weight_decay": 0.01,
            "learning_rate": 1e-4,
            "betas": (0.9, 0.999),
            "device_type": device.type
        }
        optimizer = model.configure_optimizers(**optimizer_params)
        print("Optimizer configured.")
        optimizer.step() # Dummy step
        optimizer.zero_grad()
        print("Optimizer step and zero_grad successful.")

    except Exception as e:
        print(f"Error during model testing: {e}")
        import traceback
        traceback.print_exc()

    print("Script finished.")
