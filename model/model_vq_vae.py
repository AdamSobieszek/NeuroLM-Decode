"""
VQ model with trainable alpha parameter and minimum frequency constraint for high frequencies
"""

import torch
from torch import nn
import torch.nn.functional as F
import inspect

from model.model_neural_transformer import NeuralTransformer, HybridUNetDecoder
from model.model_periodic_transformer import PeriodicTransformerDecoder
from model.model_neural_transformer import NTConfig
from torch.autograd import Function
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from scipy import signal
import math
def calculate_and_plot_fft_power(periodic_output, rec_raw, fs=200, save_plot=True):
    """
    Calculate and plot the FFT power spectrum for periodic_output and rec_raw
    
    Parameters:
    -----------
    periodic_output : torch.Tensor or numpy.ndarray
        The output from the periodic decoder
    rec_raw : torch.Tensor or numpy.ndarray
        The raw reconstruction output
    fs : int
        Sampling frequency in Hz (default: 200)
    save_plot : bool
        Whether to save the plot to a file (default: True)
    
    Returns:
    --------
    str
        Path to the saved plot if save_plot is True, otherwise None
    """
    # Convert to numpy if tensors
    if hasattr(periodic_output, 'detach'):
        periodic_output = periodic_output.detach().cpu().numpy()
    if hasattr(rec_raw, 'detach'):
        rec_raw = rec_raw.detach().cpu().numpy()
    
    # If inputs are multi-dimensional, flatten to 1D
    # Assuming first dimension is batch, second is channels, third is time
    if periodic_output.ndim > 1:
        # Take the first sample from batch, first channel
        periodic_output = periodic_output[0, 0] if periodic_output.ndim > 2 else periodic_output[0]
    if rec_raw.ndim > 1:
        rec_raw = rec_raw[0, 0] if rec_raw.ndim > 2 else rec_raw[0]
    
    # Calculate FFT
    n = len(periodic_output)
    
    # Calculate the PSD using Welch's method for better frequency resolution
    f_periodic, psd_periodic = signal.welch(periodic_output, fs=fs, nperseg=min(256, n), 
                                           scaling='spectrum')
    f_raw, psd_raw = signal.welch(rec_raw, fs=fs, nperseg=min(256, n), 
                                  scaling='spectrum')
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot power spectrum - only up to Nyquist frequency (fs/2)
    nyquist_idx_periodic = np.where(f_periodic <= fs/2)[0]
    nyquist_idx_raw = np.where(f_raw <= fs/2)[0]
    
    plt.semilogy(f_periodic[nyquist_idx_periodic], psd_periodic[nyquist_idx_periodic], 
                label='Periodic Output')
    plt.semilogy(f_raw[nyquist_idx_raw], psd_raw[nyquist_idx_raw], 
                label='Raw Reconstruction')
    
    # Customize plot
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title(f'FFT Power Spectrum (Sampling Rate: {fs} Hz)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlim(0, fs/2)  # Limit x-axis to Nyquist frequency
    plt.legend()
    
    # Save plot if requested
    if save_plot:
        # Create subfolder if it doesn't exist
        subfolder = "/workspace/fft_plots"
        os.makedirs(subfolder, exist_ok=True)
        
        # Get current timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(subfolder, f"fft_power_spectrum_{timestamp}.png")
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        
        return filename
    else:
        plt.show()
        return None

# Example usage in your forward method:
# Add this code at the end of the forward method to generate the FFT plot during inference

"""
with torch.no_grad():
    # When you want to generate and save the plot
    if not self.training:  # Only do this during validation/inference
        # Sample to use for FFT (adjust as needed)
        sample_idx = 0
        
        # Get a single sample/channel for FFT analysis
        p_output = periodic_output[sample_idx].cpu().numpy()
        r_raw = rec_raw[sample_idx].cpu().numpy()
        
        # Calculate and save plot
        plot_path = calculate_and_plot_fft_power(p_output, r_raw, fs=200)
        log[f'{split}/fft_plot_path'] = plot_path
"""

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
        grad_output1 = grad_output*0.5#*(1-passthrough_fraction)+(1-a)*passthrough_fraction)
        grad_output2 = grad_output*0.5#*(1-passthrough_fraction)+(a)*passthrough_fraction)
        
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
                 pl_weight=2.0,  # Weight for path length regularization
                 pl_decay=0.01,  # EMA decay for path length
                 periodic_decoder_config=None,
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
        self.decoder_raw = HybridUNetDecoder(decoder_config,
                                   unet_base_ch=64,
                                   unet_depth=4)

        
        # Store decoder config for later use
        self.decoder_config = decoder_config
        self.periodic_decoder_config = {} if periodic_decoder_config is None else periodic_decoder_config 

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
        
        self.pl_weight = pl_weight
        self.pl_decay = pl_decay
        self.register_buffer('pl_mean', torch.ones(1))
        self.pl_mean.requires_grad_()
    
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
            **({
            'hidden_dim':768//4,
            'num_layers':4,
            'num_heads':12,
            'dropout':0.1,
            'freq_bands':6}|self.periodic_decoder_config)
        )

        self.sigmodule_periodic_decoder_raw.apply(self._init_weights)

        # Add trainable alpha parameter that starts at 0
        # This controls the contribution of the periodic decoder to the final output
        self.sigmodule_alpha = nn.Parameter(torch.zeros(1) +1.0)  # Start with -5 for very low sigmoid value
        
        self.sigmodule_alpha.register_hook(lambda grad: grad * 100.0)
     

    def std_norm(self, x, dim=(0,1)):
            eps = 1e-10
            mean = torch.mean(x, dim=dim, keepdim=True)
            std = torch.std(x, dim=dim, keepdim=True)
            x = (x - mean) / (std+eps)
            return x

    
    def reshape_data_by_channels(self, data, ch_names, num_channels=23):
        batch_size, num_tokens, time_samples = data.shape
        
        # Calculate how many tokens per channel we have
        tokens_per_channel = num_tokens // num_channels
        
        # Create output tensor
        reshaped_data = data[:,:tokens_per_channel*num_channels].view(batch_size, num_channels, tokens_per_channel * time_samples)
        
        return reshaped_data
    
    def inverse_reshape_data_by_channels(self, reshaped_data, ch_names, time_samples=200, num_channels=23):
        batch_size, _, total_length = reshaped_data.shape
        tokens_per_channel = total_length // time_samples
        num_tokens = tokens_per_channel * num_channels
        
        # Create output tensor

        original_data = reshaped_data.view(batch_size, num_tokens, time_samples)
        
        return original_data
        
    def norm_whole_channels(self, data, input_mask, num_channels=23):
        """
        Fully differentiable implementation to normalize data only where mask is True (1)
        without using any Python loops
        
        Args:
            data: The data tensor with shape [B, N, T]
            input_mask: A boolean mask with shape [B, N] indicating which tokens to normalize
            num_channels: Number of channels in the data
            
        Returns:
            Normalized data tensor with the same shape as input
        """
        batch_size, num_tokens, time_samples = data.shape
        
        # If no mask is provided, normalize everything
        if input_mask is None:
            reshaped_data = self.reshape_data_by_channels(data, None, num_channels)
            reshaped_data = self.std_norm(reshaped_data, (1,2))
            normalized_data = self.inverse_reshape_data_by_channels(reshaped_data, None)
            return torch.cat([normalized_data, data[:, normalized_data.size(1):]], dim=1) if normalized_data.size(1) < data.size(1) else normalized_data
        
        normalized_data = data[input_mask].reshape(batch_size, num_channels, -1)
        
        # Calculate how many tokens we can reshape properly
        valid_tokens = normalized_data.size(2)//200*num_channels
        
        normalized_data = self.std_norm(normalized_data, (1,2))
        
        normalized_data = normalized_data.reshape(batch_size, valid_tokens, time_samples)
        
        # If the original data had more tokens than what we processed, concatenate them
        if valid_tokens < num_tokens:
            normalized_data = torch.cat([normalized_data, data[:, valid_tokens:]], dim=1)
        
        return normalized_data
     
    
    # Keep the original forward method without path length regularization
    def forward(self, x, y_freq, y_raw, input_chans=None, input_time=None, input_mask=None, 
               return_reconstruction=False, return_partial_reconstructions=False, **kwargs):
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
        final_freq = -rec_freq
        
        # Create concatenated input for periodic decoder
        concat_input = torch.cat([upcast_output, decoder_features_raw], dim=-1)
        projected_input = self.sigmodule_input_proj(concat_input)
        
        # Process through the periodic transformer decoder
        periodic_output = self.sigmodule_periodic_decoder_raw(
            projected_input, 
            mask=input_mask if input_mask is not None else None
        )
        
        # Get current value of alpha
        alpha = torch.abs(self.sigmodule_alpha)
        
        # Combine raw reconstruction with periodic output
        final_raw = rec_raw + alpha*periodic_output
        
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
        loss = 4*rec_freq_loss + rec_raw_loss
        
        # Log metrics
        log = {}
        split = "train" if self.training else "val"
        
        log[f'{split}/rec_freq_loss'] = rec_freq_loss.detach().mean()
        log[f'{split}/rec_raw_loss'] = rec_raw_loss.detach().mean() 
        log[f'{split}/total_loss'] = loss.detach().mean()
        
        if return_partial_reconstructions:
            return (final_freq, final_raw, rec_raw, alpha*periodic_output), quantize, log
        elif not return_reconstruction:
            return loss, quantize, log
        else:
            return (final_freq, final_raw), quantize, log
    
    
    # Add a separate method for path length regularization
    def path_length_regularization(self, quantize, y_freq, input_chans=None, input_time=None, input_mask=None):
        """
        Compute path length regularization separately to avoid double backward issues.
        This should be called separately from the main forward pass.
        
        Args:
            quantize: The latent representation from the encoder
            y_freq: Target frequency outputs for reference
            input_chans, input_time, input_mask: Additional inputs needed for model
            
        Returns:
            pl_loss: The path length regularization loss
            log: Dictionary with logging metrics
        """
        mask = input_mask.unsqueeze(1).repeat(1, y_freq.size(1), 1).unsqueeze(1) if input_mask is not None else None
        log = {}
        split = "train" if self.training else "val"
        with torch.no_grad():
            # Upcast the latent representation to the decoder's expected dimension
            upcast_output = self.upcast_layer(quantize)
            
            # Generate decoder features and output for frequency
            decoder_features_freq = self.decoder_freq(upcast_output, input_chans, input_time, mask, return_all_tokens=True)
            rec_freq = self.decode_task_layer_freq(decoder_features_freq)
            final_freq = rec_freq
            
            # Scale for noise addition
            batch_size, seq_len, feat_dim = final_freq.shape
            noise_scale = 1.0 / np.sqrt(seq_len * feat_dim)
            
            # Generate random directions for each sample
            pl_noise = torch.randn_like(final_freq) * noise_scale
        
        quantize = quantize.requires_grad_(True)
        # Compute paths for samples
        pl_lengths = []
        for i in range(batch_size):
            # Get individual sample
            q_sample = quantize[i:i+1].clone().detach().requires_grad_(True)
            
            # Forward through the network
            upcast_sample = self.upcast_layer(q_sample)
            decoder_features = self.decoder_freq(upcast_sample, 
                                               None if input_chans is None else input_chans[i:i+1], 
                                               None if input_time is None else input_time[i:i+1], 
                                               None if mask is None else mask[i:i+1], 
                                               return_all_tokens=True)
            output_sample = self.decode_task_layer_freq(decoder_features)
            
            # Apply noise
            output_with_noise = output_sample * pl_noise[i:i+1]
            
            # Compute gradient
            grads = torch.autograd.grad(outputs=output_with_noise.sum(), 
                                      inputs=q_sample,
                                      create_graph=False)[0]
            
            # Compute path length (L2 norm of gradients)
            path_length = torch.sqrt(torch.sum(grads.pow(2)))
            pl_lengths.append(path_length)
        
        # Stack path lengths
        pl_lengths = torch.stack(pl_lengths)
        
        # Update path length mean with EMA
        with torch.no_grad():
            pl_mean = self.pl_mean.lerp(pl_lengths.mean().detach(), self.pl_decay)
            self.pl_mean.copy_(pl_mean)
        
        # Compute penalty: squared difference from mean
        pl_penalty = (pl_lengths - pl_mean.detach()).pow(2).mean()
        
        # Apply weight to penalty
        pl_loss = 4* pl_penalty * self.pl_weight
        
        # Log metrics
        log[f'{split}/pl_penalty'] = pl_penalty.detach()
        log[f'{split}/pl_mean'] = pl_mean.detach()
        log[f'{split}/pl_loss'] = pl_loss.detach()
        
        return pl_loss, log

    
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
            {'params': ae_params, 'weight_decay': weight_decay*20, 'lr': learning_rate},
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
        
def load_model(ckpt_path, device, periodic_decoder_config=None):
    """
    Load the VQ_Align model from checkpoint
    """
    print(f"Loading model from checkpoint: {ckpt_path}")
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Get model configuration
    encoder_args = checkpoint['encoder_args']
    decoder_args = checkpoint['decoder_args']
    
    if "dropout" in encoder_args:
        encoder_args["dropout"] = 0.1
        decoder_args["dropout"] = 0.1
    # Create model configuration
    encoder_conf = NTConfig(**encoder_args)
    decoder_conf = NTConfig(**decoder_args)
 
    # Fix state dict keys if needed
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            k = k[len(unwanted_prefix):]
        if "sigmodule_periodic2_decoder_raw" in k: # TODO remove in the future
            state_dict[k.replace("sigmodule_periodic2_decoder_raw", "sigmodule_periodic_decoder_raw")] = state_dict.pop(k)
    state_dict = {k:v for k,v in state_dict.items() if k not in ["domain_classifier.0.weight", "domain_classifier.0.bias", "domain_classifier.2.weight", "domain_classifier.2.bias", "wte.weight", "VQ.quantize.cluster_size", "VQ.quantize.embedding.weight", "VQ.quantize.embedding.cluster_size", "VQ.quantize.embedding.embed_avg", "VQ.quantize.embedding.initted"]}
    
    # Initialize model
    
    model = VQ_Align(encoder_conf, decoder_conf, periodic_decoder_config=periodic_decoder_config)
    model.VQ.init_ae_layer()
    load_partial_state_dict(model,state_dict)
    # Load state dict
    model = model.VQ
    # print(model)
    return model

class VQ_Align(nn.Module):
    def __init__(self, 
                 encoder_config=None,
                 decoder_config=None,
                 checkpoint_path=None,
                 device="cuda",
                 periodic_decoder_config=None,
                 ):
        super(VQ_Align, self).__init__()
        if checkpoint_path:
            print("LOADING VQ.pt CHECKPOINT\n\n\n\n-----------------")
            self.VQ = load_model(checkpoint_path, device, periodic_decoder_config)
        else:
            self.VQ = VQ(encoder_config, decoder_config, periodic_decoder_config=periodic_decoder_config)
            self.VQ.init_ae_layer()
            self.to(device)
        self.domain_classifier = ThinkingOfLatents()
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
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
        domain_classifier_params = []
        print('Loaded optimizer with domain_classifier')
        
        # Categorize parameters by name and dimension
        for name, p in param_dict.items():
            # Parameters for encoder output layer and decoder input layer
            if 'domain_classifier' in name:
                domain_classifier_params.append(p)
            elif 'cast_' in name:
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
            {'params': ae_params, 'weight_decay': weight_decay*2, 'lr': learning_rate},
            {'params': sigmodule_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': domain_classifier_params, 'weight_decay': weight_decay*100, 'lr': learning_rate},
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
        
    def forward(self, x, y_freq=None, y_raw=None, input_chans=None, input_time=None, input_mask=None, return_reconstruction=False, target=None):
        if y_freq is not None:
            split="train" if self.training else "val"
            loss, encoder_features, log = self.VQ(x, y_freq, y_raw, input_chans, input_time, input_mask, return_reconstruction=False)
            
            # Initialize domain_loss and contrastive_loss
            domain_loss = 0
            contrastive_loss = 0
            
            if target is not None:
                # Apply mixup to encoder features
                batch_size = encoder_features.size(0)
                # Generate random indices for mixing
                indices = torch.randperm(batch_size, device=encoder_features.device)
                # Generate random mixing coefficient
                lam = torch.distributions.beta.Beta(0.2, 0.2).sample().to(encoder_features.device)
                
                # Mix the encoder features
                mixed_encoder_features = lam * encoder_features + (1 - lam) * encoder_features[indices]
                
                # Forward pass with mixed features
                domain_out = self.domain_classifier(mixed_encoder_features, add_noise=True)
                
                # Create a tensor filled with zeros as default
                target_hot = torch.full((domain_out.size(0), domain_out.size(-1)), fill_value=0, device=x.device)
                
                if all(target.isnan().flatten()):
                    print('here')
                    domain_loss = 0
                else:
                    try:
                        # Get the number of classes from domain_out
                        num_classes = domain_out.size(-1)
                        # Convert integer labels to one-hot encoding
                        with torch.no_grad():
                            target_hot = target_hot.unsqueeze(1).repeat((1,1024,1)).float()
                            input_mask[:,:62] = 0
                            for i in range(target.size(0)):
                                if not torch.isnan(target[i]):
                                    label_idx = int(target[i].item())
                                    # Set the corresponding class to 1 (one-hot)
                                    assert 0 <= label_idx < num_classes
                                    target_hot[i, :, label_idx] = 1
                                input_mask = input_mask.view(-1)
                        
                        # Mix the targets using the same lambda
                        mixed_target_hot = lam * target_hot.view(-1,4) + (1 - lam) * target_hot[indices].view(-1,4)
                        domain_out = domain_out.view(-1,4)
                        
                        # Calculate loss with mixed targets
                        domain_loss_expanded = F.cross_entropy(domain_out, mixed_target_hot, ignore_index=-1, reduction='none')
                        domain_loss = (domain_loss_expanded*input_mask).sum()/input_mask.sum()
                        
                        # Calculate accuracy on unmixed data
                        with torch.no_grad():
                            # Run classifier on original (unmixed) features for accuracy calculation
                            unmixed_domain_out = self.domain_classifier(encoder_features)
                            unmixed_domain_out = unmixed_domain_out.view(-1,4)
                            
                            local_mask = input_mask.clone().view(-1,1).repeat((1,4))
                            unmixed_domain_out = unmixed_domain_out[local_mask].view(-1,4)
                            target_hot_view = target_hot.view(-1,4)[local_mask].view(-1,4)
                            
                            predictions = torch.argmax(unmixed_domain_out, dim=-1)
                            correct = 0
                            total = 0
                            for i in range(predictions.size(0)):
                                    true_label = torch.argmax(target_hot_view[i]) if target_hot_view.dim() > 1 else target_hot_view[i]
                                    pred = predictions[i]
                                    correct += (pred == true_label).item()
                                    total += 1
                            accuracy = correct / total if total > 0 else 0
                        

                    except Exception as e:
                        print(e)
                # --- New contrastive learning loss ---
                try:
                    # raise Exception('Not implemented')
                    if not all(target.isnan().flatten()):
                        # Get valid indices (non-NaN targets)
                        valid_indices = ~torch.isnan(target).flatten()
                        if valid_indices.sum() > 1:  # Need at least 2 valid samples for contrastive learning
                            valid_targets = target[valid_indices].long()
                            
                            # Average the encoder features for each sample to get a single vector per sample
                            # Shape: [batch_size, feature_dim]
                            if input_mask is not None:
                                # Use masked average pooling to get sample embeddings
                                mask_expanded = input_mask.view(-1,1024,1).expand_as(encoder_features)
                                masked_features = encoder_features * mask_expanded
                                mask_sum = mask_expanded.sum(dim=1, keepdim=True).clamp(min=1e-9)
                                sample_embeddings = (masked_features.sum(dim=1) / mask_sum.squeeze(1))
                            else:
                                # Simple average pooling if no mask
                                sample_embeddings = encoder_features.mean(dim=1)
                            
                            valid_embeddings = sample_embeddings[valid_indices]
                            
                            # Normalize embeddings for cosine similarity
                            normalized_embeddings = F.normalize(valid_embeddings, p=2, dim=1)
                            
                            # Compute all pairwise similarities
                            similarities = torch.matmul(normalized_embeddings, normalized_embeddings.t())
                            
                            # Create mask for positive pairs (same class)
                            pos_mask = (valid_targets.unsqueeze(1) == valid_targets.unsqueeze(0)).float()
                            # Remove self-similarities from positive mask
                            pos_mask.fill_diagonal_(0)
                            
                            # Create mask for negative pairs (different class)
                            neg_mask = (valid_targets.unsqueeze(1) != valid_targets.unsqueeze(0)).float()
                            
                            # Temperature parameter for scaling
                            temperature = 0.2
                            
                            # Compute contrastive loss using InfoNCE / NT-Xent loss formulation
                            similarities = similarities / temperature
                            
                            # For each anchor, compute loss
                            contrastive_losses = []
                            for i in range(len(valid_embeddings)):
                                pos_similarities = similarities[i][pos_mask[i] > 0]
                                neg_similarities = similarities[i][neg_mask[i] > 0]
                                
                                if len(pos_similarities) > 0 and len(neg_similarities) > 0:
                                    # Compute log_prob: log(exp(pos_sim) / (exp(pos_sim) + sum(exp(neg_sim))))
                                    pos_exp = torch.exp(pos_similarities)/2
                                    neg_exp = torch.exp(neg_similarities).sum()
                                    
                                    # InfoNCE loss for this anchor
                                    anchor_loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()
                                    contrastive_losses.append(anchor_loss)
                            
                            if contrastive_losses:
                                contrastive_loss = torch.stack(contrastive_losses).mean()
                                
                                # Add to log
                                
                except Exception as e:
                    print(f"Contrastive loss error: {e}")
                    contrastive_loss = torch.tensor(0.0, device=x.device)
            
            similarity_matrix = (self.domain_classifier.weight_similarity_matrix().abs().sum()-4)/12
            # print(f"Domain loss: {domain_loss:.5f}, Accuracy: {accuracy:.4f}, Contrastive loss: {contrastive_loss:.5f}, Similarity matrix: {similarity_matrix:.5f}")
                     
            log[f'{split}/similarity_matrix'] = similarity_matrix
            log[f'{split}/contrastive_loss'] = contrastive_loss.detach()
            log[f'{split}/domain_loss'] = domain_loss.detach()
            log[f'{split}/accuracy'] = accuracy
            # Combine all losses with appropriate weighting
            # You can adjust these weights as needed
            lambda_domain = 0.3
            lambda_contrastive = 0.004 # more?
            lambda_domain_correlation = 1.0
            correlation_loss = self.domain_classifier.minimize_weight_correlation_loss(lambda_domain_correlation)
            total_loss = loss + lambda_domain * domain_loss + lambda_contrastive * contrastive_loss + correlation_loss
            
            return total_loss, encoder_features, log


def load_partial_state_dict(model, state_dict, verbose=False):
    """
    Load a state dictionary that doesn't fully align with the model.
    This will load all matching parameters and skip non-matching ones.
    
    Args:
        model: The target model to load parameters into
        state_dict: The state dictionary containing parameters to load
        verbose: Whether to print detailed loading information
    
    Returns:
        dict: Statistics about the loading process
    """
    model_state_dict = model.state_dict()
    
    # Track statistics
    stats = {
        'loaded': [],
        'skipped_missing_in_model': [],
        'skipped_missing_in_state_dict': [],
        'skipped_shape_mismatch': [],
    }
    
    # Try to load parameters
    for name, param in state_dict.items():
        if name in model_state_dict:
            if model_state_dict[name].shape == param.shape:
                model_state_dict[name].copy_(param)
                stats['loaded'].append(name)
            else:
                stats['skipped_shape_mismatch'].append({
                    'name': name, 
                    'model_shape': tuple(model_state_dict[name].shape),
                    'state_dict_shape': tuple(param.shape)
                })
        else:
            stats['skipped_missing_in_model'].append(name)
    
    # Track parameters in model that weren't loaded
    for name in model_state_dict:
        if name not in state_dict:
            stats['skipped_missing_in_state_dict'].append(name)
    
    # Update the model with the loaded parameters
    model.load_state_dict(model_state_dict, strict=False)
    
    # Print statistics if verbose
    if verbose:
        print(f"Loaded {len(stats['loaded'])}/{len(model_state_dict)} parameters")
        print(f"Skipped {len(stats['skipped_missing_in_model'])} parameters missing in model")
        print(f"Skipped {len(stats['skipped_missing_in_state_dict'])} parameters missing in state dict")
        print(f"Skipped {len(stats['skipped_shape_mismatch'])} parameters due to shape mismatch")
        
        if stats['skipped_shape_mismatch'] and verbose:
            print("\nShape mismatches:")
            for mismatch in stats['skipped_shape_mismatch']:
                print(f"  {mismatch['name']}: model={mismatch['model_shape']} vs state_dict={mismatch['state_dict_shape']}")
    
    return stats

class ThinkingOfLatents(torch.nn.Module):
    """
    A neural network module that transforms a hidden_dim-dimensional input into n_classes outputs
    using a single linear transformation with persistent, correlated rotations across forward passes.
    """
    def __init__(self, hidden_dim=32, n_classes=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        
        # Weight parameter, shape [hidden_dim, n_classes]
        self.weight = torch.nn.Parameter(torch.randn(hidden_dim, self.n_classes))
        # Bias parameter, shape [n_classes]
        self.bias = torch.nn.Parameter(torch.zeros(self.n_classes))
        
        # Initialize a persistent rotation direction
        self.initialize_rotation_direction()
        
        # Noise factor for updating rotation direction
        self.noise_factor = 0.1

    def initialize_rotation_direction(self):
        """Initialize a persistent rotation direction orthogonal to weight space and first 8 standard basis vectors"""
        with torch.no_grad():
            # Get the initial weight matrix
            weight_matrix = self.weight.detach()
            
            # Get orthogonal basis for the weight space
            Q, _ = torch.linalg.qr(weight_matrix)
            
            # Generate initial random vector
            random_vec = torch.randn(self.hidden_dim, 1, device=self.weight.device)
            
            # Project out components in the weight space
            for i in range(min(self.n_classes, self.hidden_dim)):
                q_i = Q[:, i:i+1]
                random_vec = random_vec - q_i * torch.matmul(q_i.T, random_vec)
            
            # Project out components from first 8 standard basis vectors
            components_to_zero = min(8, self.hidden_dim)
            if components_to_zero > 0:
                random_vec[:components_to_zero, 0] = 0.0
            
            # Normalize the direction
            self.rotation_direction = torch.nn.functional.normalize(random_vec, p=2, dim=0).to(self.weight.device)
            
    def update_rotation_direction(self):
        """Add small noise to rotation direction to create correlated rotations between forward passes"""
        with torch.no_grad():
            # Generate small noise
            noise = torch.randn_like(self.rotation_direction)
            noise = torch.nn.functional.normalize(noise, p=2, dim=0) * self.noise_factor
            
            # Add noise to current direction
            noisy_direction = self.rotation_direction + noise.to(self.rotation_direction.device)
            
            # Re-project out components from weight space
            weight_matrix = self.weight.detach()

            # Re-project out components from first 8 standard basis vectors
            components_to_zero = min(8, self.hidden_dim)
            if components_to_zero > 0:
                noisy_direction[:components_to_zero, 0] = 0.0
            
            # Re-normalize
            self.rotation_direction = torch.nn.functional.normalize(noisy_direction, p=2, dim=0)

    def forward(self, x, add_noise=False):
        """
        Forward pass that rotates all linear classification heads using a persistent, 
        slowly evolving rotation direction.
        
        Args:
            x (torch.Tensor): Input tensor of shape [..., hidden_dim]
            add_noise (bool): Whether to apply rotation to the weights
            
        Returns:
            torch.Tensor: Output tensor of shape [..., n_classes] (squeezed)
        """
        weights = self.weight  # [hidden_dim, n_classes]
        
        if add_noise:
            # Use the persistent rotation direction
            rotation_direction = self.rotation_direction.to(x.device)
            
            # Generate random rotation angle
            angle = torch.rand(1, device=weights.device) * math.pi/2
            cos_theta = torch.cos(angle)
            sin_theta = torch.sin(angle)
            
            # Create rotation matrix in the plane spanned by each weight vector and the rotation direction
            rotated_weights = torch.zeros_like(weights)
            
            for j in range(self.n_classes):
                weight_j = weights[:, j:j+1]  # [hidden_dim, 1]
                weight_j_norm = weight_j.norm()
                
                # Get the component of weight_j orthogonal to rotation_direction
                proj = torch.matmul(rotation_direction.T, weight_j)
                weight_orthogonal = weight_j - rotation_direction * proj
                weight_orthogonal_norm = weight_orthogonal.norm() + 1e-8  # Avoid division by zero
                weight_orthogonal_normalized = weight_orthogonal / weight_orthogonal_norm
                
                # Apply rotation in the plane while maintaining differentiability
                rotated_weights[:, j] = (weight_orthogonal_normalized * cos_theta * weight_j_norm + 
                                        rotation_direction * sin_theta * weight_j_norm).squeeze()
            
            weights = rotated_weights
            
            # Update the rotation direction for the next forward pass
            self.update_rotation_direction()
        
        output = torch.matmul(x, weights) + self.bias
        return output.squeeze()

    def compute_weight_correlation(self):
        """
        Computes the mean absolute correlation between the weight vectors of the linear layer.
        This is fully differentiable and can be used as a regularization term.
        
        Returns:
            torch.Tensor: Mean absolute correlation between all pairs of weight vectors
        """
        # self.weight has shape [hidden_dim, n_classes].
        # Transpose to get weight vectors as rows: shape [n_classes, hidden_dim].
        weights_matrix = self.weight.T # Keep gradients
        
        # Normalize each weight vector (row) using L2 norm
        normalized_weights = torch.nn.functional.normalize(weights_matrix, p=2, dim=1)
        
        # Compute cosine similarity matrix. Shape: [n_classes, n_classes]
        similarity_matrix = torch.matmul(normalized_weights, normalized_weights.t())
        
        # Create a mask to exclude self-correlations (diagonal elements)
        mask = torch.ones_like(similarity_matrix) - torch.eye(self.n_classes, device=similarity_matrix.device)
        
        # Calculate mean absolute correlation (excluding diagonal)
        # Take absolute values to penalize both positive and negative correlations
        mean_abs_correlation = (torch.abs(similarity_matrix) * mask).sum() / (mask.sum().clamp(min=1e-9))
        
        return mean_abs_correlation
    
    def minimize_weight_correlation_loss(self, lambda_corr=1.0):
        """
        Computes a loss term to minimize correlation between classifier weights.
        This can be added to the main loss function during training.
        
        Args:
            lambda_corr (float): Weight for the correlation penalty
            
        Returns:
            torch.Tensor: Weighted correlation loss
        """
        correlation = self.compute_weight_correlation()
        return lambda_corr * correlation

    def weight_similarity_matrix(self):
        """
        Computes the matrix of cosine similarities between the weight vectors of the linear layer.
        Each column of self.weight is a weight vector for a class.
        The weights are detached from the computation graph for this analysis.
        """
        # self.weight has shape [hidden_dim, n_classes].
        # Transpose to get weight vectors as rows: shape [n_classes, hidden_dim].
        weights_matrix = self.weight.T.detach()
        
        # Normalize each weight vector (row) using L2 norm.
        normalized_weights = torch.nn.functional.normalize(weights_matrix, p=2, dim=1)
        
        # Compute cosine similarity matrix: W_norm @ W_norm.T
        # Resulting shape: [n_classes, n_classes].
        similarity_matrix = torch.matmul(normalized_weights, normalized_weights.t())
        
        return similarity_matrix

        
class OrthogonalThinkingOfLatents(torch.nn.Module):
    """
    A classifier with 4 output heads, where the weight vectors for these heads
    are constrained to be orthogonal. Orthogonality is achieved via QR decomposition
    of a learnable parameter matrix.
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        if hidden_dim < 4:
            raise ValueError("hidden_dim must be at least 4 to support 4 orthogonal directions.")
        
        # Parameter matrix from which orthogonal weights will be derived.
        # Shape: [hidden_dim, 4] so that its columns form a basis for the weights.
        self.weight_basis_params = torch.nn.Parameter(torch.randn(hidden_dim, 4))
        # Individual biases for each of the 4 outputs
        self.biases = torch.nn.Parameter(torch.zeros(4))

    def get_orthogonal_weights(self):
        """
        Derives 4 orthogonal weight vectors from self.weight_basis_params using QR decomposition.
        Returns:
            torch.Tensor: Orthogonal weight vectors of shape [4, hidden_dim].
        """
        # Q will have shape [hidden_dim, 4] with orthonormal columns.
        # R will have shape [4, 4].
        q, r = torch.linalg.qr(self.weight_basis_params)
        
        # Transpose Q so that rows are the orthogonal weight vectors. Shape: [4, hidden_dim].
        orthogonal_weights = q.T
        
        # Adjust signs based on the diagonal of R for a more canonical representation,
        # which can help stabilize training.
        signs = torch.sign(torch.diag(r))
        orthogonal_weights = orthogonal_weights * signs.unsqueeze(1) # signs: [4] -> [4,1]
        
        return orthogonal_weights

    def forward(self, x):
        """
        Forward pass using orthogonal weight vectors.
        Args:
            x (torch.Tensor): Input tensor of shape [..., hidden_dim].
        Returns:
            torch.Tensor: Output tensor of shape [..., 4].
        """
        weights = self.get_orthogonal_weights() # Shape [4, hidden_dim]
        # x: [..., hidden_dim], weights.T: [hidden_dim, 4]
        # Output: [..., 4]
        output = torch.matmul(x, weights.T) + self.biases
        return output.squeeze() # Matches original ThinkingOfLatents behavior

    def weight_similarity_matrix(self):
        """
        Computes the cosine similarity matrix of the (orthogonal) weight vectors.
        Ideally, this should be an identity matrix. For analysis purposes (detached).
        """
        weights = self.get_orthogonal_weights().detach()
        normalized_weights = torch.nn.functional.normalize(weights, p=2, dim=1)
        similarity_matrix = torch.matmul(normalized_weights, normalized_weights.t())
        return similarity_matrix

    def compute_mean_abs_off_diagonal_correlation(self):
        """
        Computes the mean absolute cosine similarity of the off-diagonal elements
        of the orthogonal weight vectors. This should be close to 0 due to QR.
        This version uses weights with gradients for potential regularization.
        """
        weights = self.get_orthogonal_weights() # Gradients are preserved
        normalized_weights = torch.nn.functional.normalize(weights, p=2, dim=1)
        similarity_matrix = torch.matmul(normalized_weights, normalized_weights.t())
        
        num_outputs = 4
        mask = torch.ones_like(similarity_matrix) - torch.eye(num_outputs, device=similarity_matrix.device)
        mean_abs_correlation = (torch.abs(similarity_matrix) * mask).sum() / (mask.sum().clamp(min=1e-9))
        return mean_abs_correlation

    def minimize_weight_correlation_loss(self, lambda_corr=1.0):
        """
        Computes a loss term based on the correlation of the (nominally) orthogonal weights.
        Since weights are made orthogonal by QR, this loss should be near zero and primarily
        penalizes numerical inaccuracies or acts as a regularizer on weight_basis_params.
        """
        correlation = self.compute_mean_abs_off_diagonal_correlation()
        return lambda_corr * correlation

class EnsembleOfOrthogonalThinkers(torch.nn.Module):
    """
    An ensemble of `n_classifiers` instances of `OrthogonalThinkingOfLatents`.
    Each instance maintains 4 internally orthogonal classifier heads.
    The forward pass averages the outputs of these ensemble members.
    A regularization loss can be applied to encourage decorrelation *among all*
    `4 * n_classifiers` direction vectors.
    """
    def __init__(self, hidden_dim=32, n_classifiers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_classifiers = n_classifiers

        if self.n_classifiers <= 0:
            raise ValueError("n_classifiers must be positive.")

        # The condition n_classifiers * 4 < hidden_dim relates to the possibility of
        # making all 4 * n_classifiers directions globally mutually orthogonal.
        # This class structure doesn't enforce global orthogonality by construction,
        # but the regularization term will aim to reduce correlations.
        if self.n_classifiers * 4 >= hidden_dim:
            print(f"Warning: Total directions ({self.n_classifiers * 4}) "
                  f"is not strictly less than hidden_dim ({hidden_dim}). "
                  f"Achieving full global orthogonality for all {self.n_classifiers*4} directions via "
                  f"regularization might be challenging.")

        self.classifiers = torch.nn.ModuleList(
            [OrthogonalThinkingOfLatents(hidden_dim) for _ in range(n_classifiers)]
        )

    def forward(self, x):
        """
        Forward pass through the ensemble.
        Args:
            x (torch.Tensor): Input tensor of shape [..., hidden_dim].
        Returns:
            torch.Tensor: Averaged output tensor of shape [..., 4].
        """
        if self.n_classifiers == 0:
             # Should match expected output shape, perhaps zeros or handle upstream
            return torch.zeros(*x.shape[:-1], 4, device=x.device).squeeze()


        all_outputs = [classifier(x) for classifier in self.classifiers] # List of [..., 4] tensors
        
        # Stack along a new dimension (dim 0) and then average over this dimension.
        # stacked_outputs shape: [n_classifiers, ..., 4]
        stacked_outputs = torch.stack(all_outputs, dim=0)
        avg_output = torch.mean(stacked_outputs, dim=0) # Shape [..., 4]
        
        return avg_output.squeeze() # Matches original ThinkingOfLatents behavior

    def get_all_weight_vectors(self, detach=False):
        """
        Collects all 4 * n_classifiers weight vectors from all ensemble members.
        Args:
            detach (bool): Whether to detach the weight vectors from the computation graph.
        Returns:
            torch.Tensor: Tensor of all weight vectors, shape [4 * n_classifiers, hidden_dim].
        """
        all_weights_list = []
        for clf in self.classifiers:
            weights = clf.get_orthogonal_weights() # Shape [4, hidden_dim]
            if detach:ń
                weights = weights.detach()
            all_weights_list.append(weights)
        
        if not all_weights_list: # Should not happen if n_classifiers > 0
            # Attempt to get a device if possible, otherwise default to CPU
            device = "cpu"
            if len(self.classifiers) > 0 and hasattr(self.classifiers[0], 'weight_basis_params'):
                 device = self.classifiers[0].weight_basis_params.device
            return torch.empty(0, self.hidden_dim, device=device)

        return torch.cat(all_weights_list, dim=0)

    def compute_average_correlation_all_directions(self):
        """
        Computes the mean absolute cosine similarity among all 4 * n_classifiers
        weight vectors from all ensemble members. This is used for regularization.
        """
        if self.n_classifiers == 0:
            device = "cpu"
            if len(self.classifiers) > 0 and hasattr(self.classifiers[0], 'weight_basis_params'):
                 device = self.classifiers[0].weight_basis_params.device
            return torch.tensor(0.0, device=device)

        all_weights = self.get_all_weight_vectors(detach=False) # Keep gradients
        
        num_total_directions = all_weights.shape[0]
        if num_total_directions <= 1: # Need at least 2 vectors for correlation
            return torch.tensor(0.0, device=all_weights.device)

        normalized_weights = torch.nn.functional.normalize(all_weights, p=2, dim=1)
        # similarity_matrix has shape [4*n_classifiers, 4*n_classifiers]
        similarity_matrix = torch.matmul(normalized_weights, normalized_weights.t())
        
        mask = torch.ones_like(similarity_matrix) - torch.eye(num_total_directions, device=similarity_matrix.device)
        
        # Sum of absolute off-diagonal elements / number of off-diagonal elements
        mean_abs_correlation = (torch.abs(similarity_matrix) * mask).sum() / (mask.sum().clamp(min=1e-9))
        return mean_abs_correlation

    def minimize_weight_correlation_loss(self, lambda_corr=1.0):
        """
        Computes a loss term to minimize correlation between all classifier directions
        across the entire ensemble.
        """
        correlation = self.compute_average_correlation_all_directions()
        return lambda_corr * correlation

    def weight_similarity_matrix_overall(self):
        """
        For analysis: returns the cosine similarity matrix of all 4 * n_classifiers weight vectors.
        """
        all_weights = self.get_all_weight_vectors(detach=True)
        if all_weights.shape[0] == 0:
            device = "cpu"
            if len(self.classifiers) > 0 and hasattr(self.classifiers[0], 'weight_basis_params'):
                 device = self.classifiers[0].weight_basis_params.device
            return torch.empty(0,0, device=device)

        normalized_weights = torch.nn.functional.normalize(all_weights, p=2, dim=1)
        similarity_matrix = torch.matmul(normalized_weights, normalized_weights.t())
        return similarity_matrix