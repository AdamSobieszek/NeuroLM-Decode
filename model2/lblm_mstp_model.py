import torch
import torch.nn as nn
import torch.fft
import inspect # For optimizer configuration

# Use relative imports if these files are in the same package/directory
from .lblm_input_processor import LBLMInputProcessor
from .lgtransformer import LBLMConformerBackbone
from .lblm_prediction_heads import SpectroTemporalPredictionHeads
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
                 eeg_channels=122,
                 conformer_input_dim=64,
                 conformer_num_layers=4,
                 conformer_d_model=64,
                 conformer_n_head=8,
                 conformer_d_ff=256,
                 conformer_dropout_rate=0.1,
                 conformer_conv_kernel_size=31,
                 huber_delta=1.0,
                 lambda_amplitude=0.1,
                 lambda_phase=0.1
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

        self.loss_fn_huber = HuberLoss(delta=huber_delta)
        self.lambda_amplitude = lambda_amplitude
        self.lambda_phase = lambda_phase

    def _calculate_fft_targets(self, original_patches_masked):
        # original_patches_masked: [TotalNumberOfMaskedPatches, PatchLength]
        if original_patches_masked.numel() == 0:
            # Return empty tensors with correct trailing dimensions if no patches
            return (torch.empty((0, self.fft_target_dim), device=original_patches_masked.device),
                    torch.empty((0, self.fft_target_dim), device=original_patches_masked.device))

        fft_result = torch.fft.rfft(original_patches_masked, dim=-1, norm="ortho") # Using ortho norm
        amplitude = torch.abs(fft_result)
        phase = torch.angle(fft_result)
        return amplitude, phase

    def forward(self, x_raw_eeg, subject_ids=None, input_mask_override=None):
        # x_raw_eeg: [batch_size, eeg_channels, num_timepoints]
        
        batch_size_orig, num_input_channels, _ = x_raw_eeg.shape

        # 1. Process Input
        processed_input = self.input_processor(x_raw_eeg, subject_ids)
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
        # print(pred_wave, target_wave_patches, loss_wave)
        loss_amplitude = self.loss_fn_huber(pred_amplitude, target_amplitude)
        loss_phase = self.loss_fn_huber(pred_phase, target_phase)

        total_loss = loss_wave + self.lambda_amplitude * loss_amplitude + self.lambda_phase * loss_phase
        
        log.update({
            'mstp/loss_wave': loss_wave.detach(),
            'mstp/loss_amplitude': loss_amplitude.detach(),
            'mstp/loss_phase': loss_phase.detach(),
            'mstp/total_loss': total_loss.detach()
        })
        
        return total_loss, None, log

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type='cpu', optimizer_type='adamw', eps=1e-6, **kwargs):
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
            print(f"Configuring LAMB optimizer with LR={learning_rate}, WD={weight_decay}, Betas={betas}, Eps={kwargs.get('eps', 1e-6)}")
            optimizer = Lamb(param_dict.values(), lr=learning_rate, betas=betas, weight_decay=weight_decay, eps=kwargs.get('eps', 1e-6))
        else:
            optimizer = torch.optim.AdamW(optim_groups, betas=betas, **extra_args)
        return optimizer

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
