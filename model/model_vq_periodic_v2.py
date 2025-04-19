"""
VQ model with trainable alpha parameter and minimum frequency constraint for high frequencies
"""

import torch
from torch import nn
import torch.nn.functional as F
import inspect

from model.model_neural_transformer import NeuralTransformer
from model.model_periodic_transformer_old import PeriodicTransformerDecoder
from model.model_neural_transformer import NTConfig
from torch.autograd import Function
import numpy as np

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class ExpertAverage(torch.autograd.Function):
    """
    Custom autograd function for weighted average of two models' outputs.
    Takes direct weight 'a' (not alpha) and controls gradient flow.
    
    forward: output = a * output1 + (1-a) * output2
    """
    
    @staticmethod
    def forward(ctx, output1, output2, a, passthrough_fraction=0):
        # Compute weighted average directly
        result = (1 - a) * output1 + a * output2
        
        # Save for backward pass
        ctx.save_for_backward(output1, output2, a)
        ctx.passthrough_fraction = passthrough_fraction
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors and options
        output1, output2, a = ctx.saved_tensors
        passthrough_fraction = ctx.passthrough_fraction
        
        # Gradient for output1: grad_output * a
        grad_output1 = grad_output*(0.5*(1-passthrough_fraction)+(1-a)*passthrough_fraction)
        grad_output2 = grad_output*(0.5*(1-passthrough_fraction)+(a)*passthrough_fraction)
        
        # Gradient a is calculated as:
        # d_result/d_a = output1 - output2
        grad_a = torch.ones_like(a)*torch.sum(grad_output * (output2 - output1))

        return grad_output1, grad_output2, grad_a, None

class VQ(nn.Module):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 n_embed=8192, 
                 embed_dim=128,
                 latent_dim=32,
                 decay=0.99,
                 quantize_kmeans_init=False,
                 decoder_out_dim=200,
                 smooth_l1_loss=False,
                 **kwargs
                 ):
        super().__init__()
        print(kwargs)
        if decoder_config.in_chans != embed_dim:
            print(f"Rewrite the in_chans in decoder from {decoder_config.in_chans} to {embed_dim}")
            decoder_config.in_chans = embed_dim

        # encoder & decoder params
        print('Final encoder config', encoder_config)
        self.encoder = NeuralTransformer(encoder_config)

        print('Final decoder config', decoder_config)
        self.decoder_freq = NeuralTransformer(decoder_config)
        self.decoder_raw = NeuralTransformer(decoder_config)
        
        # Store decoder config for later use
        self.decoder_config = decoder_config

        self.decoder_out_dim = decoder_out_dim

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config.n_embd, encoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(encoder_config.n_embd, embed_dim) # for quantize
        )
        self.decode_task_layer_freq = nn.Sequential(
            nn.Linear(decoder_config.n_embd, decoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(decoder_config.n_embd, self.decoder_out_dim // 2),
        )
        self.decode_task_layer_raw = nn.Sequential(
            nn.Linear(decoder_config.n_embd, decoder_config.n_embd),
            nn.Tanh(),
            nn.Linear(decoder_config.n_embd, self.decoder_out_dim),
        )

        self.kwargs = kwargs
        
        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer_freq.apply(self._init_weights)
        self.decode_task_layer_raw.apply(self._init_weights)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.embed_dim = embed_dim
        self.n_embd = encoder_config.n_embd
        self.latent_dim = latent_dim
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def init_ae_layer(self):
        # Enhanced downcast layer with BatchNorm and GELU activations
        self.downcast_layer = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(self.embed_dim*2, self.latent_dim),
            nn.BatchNorm1d(1024),
            nn.Tanh(),
        )
        self.downcast_layer.apply(self._init_weights)
        
        # Symmetrical upcast layer for balanced upscaling
        self.upcast_layer = nn.Sequential(
            nn.Linear(self.latent_dim, self.embed_dim * 2),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim * 2),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.BatchNorm1d(1024),
            nn.Tanh(),
        )
        self.upcast_layer.apply(self._init_weights)
        
        # Input dimension adjustment layer for concatenated inputs
        self.sigmodule_input_proj = nn.Sequential(
            nn.Linear(self.embed_dim + self.decoder_config.n_embd, self.decoder_out_dim),
            nn.LayerNorm(self.decoder_out_dim),
            nn.GELU()
        )
        self.sigmodule_input_proj.apply(self._init_weights)
        
        # Add the optimized periodic transformer module for raw signal processing
        # Now with minimum frequency parameter for high-frequency focus
        self.sigmodule_periodic_decoder_raw = PeriodicTransformerDecoder(
            input_dim=self.decoder_out_dim,
            hidden_dim=768,
            num_layers=4,
            num_heads=12,
            dropout=0.1,
            freq_bands=6,
            min_alpha=258.0  # Set minimum frequency parameter to focus on high frequencies
        )
        self.sigmodule_periodic_decoder_raw.apply(self._init_weights)
        
        # Add trainable alpha parameter that starts at 0
        # This controls the contribution of the periodic decoder to the final output
        self.sigmodule_alpha = nn.Parameter(torch.zeros(1) - 5.0)  # Start with -5 for very low sigmoid value
        
        # self.sigmodule_alpha.register_hook(lambda grad: grad * 1000.0)
        
    def forward(self, x, y_freq, y_raw, input_chans=None, input_time=None, input_mask=None, return_reconstruction=False, return_partial_reconstructions = False, **kwargs):
        """
        Forward pass with weighted contribution from periodic decoder
        x: shape [B, N, T]
        """
        mask = input_mask.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1) if input_mask is not None else None
        
        # Encode the input signal
        quantize, _, _, encoder_features = self.encode(x, input_chans, input_time, mask)
        
        # Upcast the latent representation to the decoder's expected dimension
        upcast_output = self.upcast_layer(quantize)
        
        # Get decoder outputs, but not yet processed through task layers
        decoder_features_freq = self.decoder_freq(upcast_output, input_chans, input_time, mask, return_all_tokens=True)
        decoder_features_raw = self.decoder_raw(upcast_output, input_chans, input_time, mask, return_all_tokens=True)
        
        # Generate the base reconstructions through task layers
        rec_freq = self.decode_task_layer_freq(decoder_features_freq)
        rec_raw = self.decode_task_layer_raw(decoder_features_raw)
        
        # No enhancement for frequency
        final_freq = rec_freq
        
        # Create concatenated input for periodic decoder: upcast_output + decoder_features_raw
        # [B, N, embed_dim + decoder.n_embd]
        concat_input = torch.cat([upcast_output, decoder_features_raw], dim=-1)
        
        # Project the concatenated input to the expected dimension
        projected_input = self.sigmodule_input_proj(concat_input)
        
        # Process through the periodic transformer decoder
        # The decoder now focuses on high-frequency components due to min_alpha
        periodic_output = self.sigmodule_periodic_decoder_raw(
            projected_input, 
            mask=input_mask if input_mask is not None else None
        )
        
        # Get current value of alpha, clamped between 0 and 1
        alpha = torch.sigmoid(self.sigmodule_alpha)
        
        # Final raw output is a weighted combination:
        # At the beginning (alpha ≈ 0): final_raw ≈ rec_raw
        # A decoupled learning process updates alpha to suitable value with final_raw = rec_raw*(1-alpha) + periodic_output*alpha
        with torch.no_grad():
            is_it_bad = abs(alpha-0.5)*2
        final_raw = ExpertAverage.apply(rec_raw, periodic_output, alpha, is_it_bad)
        # final_raw = (1-alpha)*rec_raw + alpha*periodic_output
        # Apply mask and calculate losses
        if input_mask is not None:
            loss_freq_mask = input_mask.unsqueeze(-1).repeat(1, 1, final_freq.size(-1))
            loss_raw_mask = input_mask.unsqueeze(-1).repeat(1, 1, final_raw.size(-1))
            
            rec_freq_loss = self.calculate_rec_loss(final_freq * loss_freq_mask, y_freq)
            rec_raw_loss = self.calculate_rec_loss(final_raw * loss_raw_mask, y_raw)
        else:
            rec_freq_loss = self.calculate_rec_loss(final_freq, y_freq)
            rec_raw_loss = self.calculate_rec_loss(final_raw, y_raw)
        
        # Calculate total loss - higher weight on frequency loss as it's often more stable
        loss = rec_freq_loss + rec_raw_loss
        
        # Log metrics
        log = {}
        split = "train" if self.training else "val"
        
        log[f'{split}/rec_freq_loss'] = rec_freq_loss.detach().mean()
        log[f'{split}/rec_raw_loss'] = rec_raw_loss.detach().mean() 
        log[f'{split}/total_loss'] = loss.detach().mean()
        # log[f'{split}/alpha'] = alpha.detach()  # Track alpha value during training
    
        if return_partial_reconstructions:
            print((final_freq, final_raw, (1-alpha)*rec_raw, alpha*periodic_output))
            return (final_freq, final_raw, (1-alpha)*rec_raw, alpha*periodic_output), encoder_features, log
        elif not return_reconstruction:
            return loss, encoder_features, log
        else:
            return (final_freq, final_raw), encoder_features, log

    
    
    # Add to your VQ class
    def force_update_alpha(self, new_value=0.0):
        """Force-update the alpha parameter for debugging"""
        # Find alpha parameter
        for name, param in self.named_parameters():
            if 'alpha' in name.lower():
                # print(f"Forcing update to {name}")
                # print(f"  Old value: {param.data.item()}")
                with torch.no_grad():
                    param.data.fill_(new_value)
                # print(f"  New value: {param.data.item()}")
                return True
        
        print("Alpha parameter not found!")
        return False

    
    @property
    def device(self):
        return self.decoder_freq.cls_token.device if hasattr(self.decoder_freq, 'cls_token') else next(self.parameters()).device
    
    def encode(self, x, input_chans=None, input_time=None, mask=None):
        encoder_features = self.encoder(x, input_chans, input_time, mask, return_all_tokens=True)
        
        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))
        
        quantize = self.downcast_layer(to_quantizer_features)
        return quantize, None, None, encoder_features
    
    def calculate_rec_loss(self, rec, target):
        rec_loss = self.loss_fn(rec, target)
        return rec_loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # Categorize parameters
        ae_params = []
        sigmodule_params = []
        decay_params = []
        nodecay_params = []
        
        # Categorize parameters by name and dimension
        for name, p in param_dict.items():
            if 'cast_' in name:
                ae_params.append(p)
            elif 'sigmodule' in name:
                sigmodule_params.append(p)
            elif p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)
        
        # Configure optimization groups
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate},
            {'params': ae_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            # Give higher learning rate to the periodic transformer to accelerate its training
            {'params': sigmodule_params, 'weight_decay': weight_decay, 'lr': learning_rate},
        ]
            
        # Print parameter counts for debugging
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        num_ae_params = sum(p.numel() for p in ae_params)
        num_sigmodule_params = sum(p.numel() for p in sigmodule_params)
        print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        print(f"Num AE interface parameters: {len(ae_params)}, with {num_ae_params:,} parameters")
        print(f"Num sigmodule parameters: {len(sigmodule_params)}, with {num_sigmodule_params:,} parameters")
        
        # Create optimizer
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")
        
        return optimizer
    
        
def load_model(ckpt_path, device):
    """
    Load the VQ_Align model from checkpoint
    """
    print(f"Loading model from checkpoint: {ckpt_path}")
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Get model configuration
    encoder_args = checkpoint['encoder_args']
    decoder_args = checkpoint['decoder_args']
    
    if "dropout" in encoder_args:
        encoder_args["dropout"] = 0.1
        decoder_args["dropout"] = 0.1
    # Create model configuration
    encoder_conf = NTConfig(**encoder_args)
    decoder_conf = NTConfig(**decoder_args)
    encoder_conf

    
    # Initialize model
    model = VQ_Align(encoder_conf, decoder_conf)
    
    # Fix state dict keys if needed
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    state_dict = {k:v for k,v in state_dict.items() if k not in ["domain_classifier.0.weight", "domain_classifier.0.bias", "domain_classifier.2.weight", "domain_classifier.2.bias", "wte.weight", "VQ.quantize.cluster_size", "VQ.quantize.embedding.weight", "VQ.quantize.embedding.cluster_size", "VQ.quantize.embedding.embed_avg", "VQ.quantize.embedding.initted"]}
    
    # Load state dict
    model.load_state_dict(state_dict)
    model = model.VQ
    model.init_ae_layer()
    # print(model)
    return model

class VQ_Align(nn.Module):
    def __init__(self, 
                 encoder_config=None,
                 decoder_config=None,
                 checkpoint_path=None
                 ):
        super(VQ_Align, self).__init__()
        if checkpoint_path:
            print("LOADING VQ.pt CHECKPOINT\n\n\n\n-----------------")
            self.VQ = load_model(checkpoint_path, "cuda")
        else:
            self.VQ = VQ(encoder_config, decoder_config)
        
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, y_freq=None, y_raw=None, input_chans=None, input_time=None, input_mask=None, return_reconstruction=False):
        if y_freq is not None:
            loss, encoder_features, log = self.VQ(x, y_freq, y_raw, input_chans, input_time, input_mask, return_reconstruction=False)
            # reverse_x = ReverseLayerF.apply(encoder_features, alpha)
            # domain_out = self.domain_classifier(reverse_x)
            # target = torch.full((domain_out.size(0), domain_out.size(1)), fill_value=-1, device=x.device)
            # target[input_mask == True] = 0
            # domain_loss = F.cross_entropy(domain_out.view(-1, domain_out.size(-1)), target.view(-1), ignore_index=-1)
            split="train" if self.training else "val"
            # log[f'{split}/domain_loss'] = domain_loss.detach().item()
            return loss, None, log

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # Identify AE parameters more precisely - look for specific parameter names
        # that belong to the encoder-decoder interface layers
        ae_params = []
        sigmodule_params = []
        decay_params = []
        nodecay_params = []
        
        # Categorize parameters by name and dimension
        for name, p in param_dict.items():
            # Parameters for encoder output layer and decoder input layer
            if 'cast_' in name:
                ae_params.append(p)
            elif 'sigmodule' in name:
                sigmodule_params.append(p)
            # Standard weight decay for 2D+ parameters that aren't in the AE interfaces
            elif p.dim() >= 2:
                decay_params.append(p)
            # No decay for 1D parameters (biases, etc.)
            else:
                nodecay_params.append(p)
        
        # Configure optimization groups with appropriate learning rates and weight decay
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate},
            {'params': ae_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': sigmodule_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            ]
            
        # Print parameter counts for debugging
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        num_ae_params = sum(p.numel() for p in ae_params)
        num_sigmodule_params = sum(p.numel() for p in sigmodule_params)
        print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        print(f"Num AE interface parameters: {len(ae_params)}, with {num_ae_params:,} parameters")
        print(f"Num sigmodule parameters: {len(sigmodule_params)}, with {num_sigmodule_params:,} parameters")
        
        # Create AdamW optimizer and use the fused version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")
        
        return optimizer

    