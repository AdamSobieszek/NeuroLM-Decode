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
            'hidden_dim':768//2,
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
        
def load_model(ckpt_path, device, periodic_decoder_config=None):
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
 
    # Fix state dict keys if needed
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    state_dict = {k:v for k,v in state_dict.items() if k not in ["domain_classifier.0.weight", "domain_classifier.0.bias", "domain_classifier.2.weight", "domain_classifier.2.bias", "wte.weight", "VQ.quantize.cluster_size", "VQ.quantize.embedding.weight", "VQ.quantize.embedding.cluster_size", "VQ.quantize.embedding.embed_avg", "VQ.quantize.embedding.initted"]}
    
    
    # Initialize model
    
    model = VQ_Align(encoder_conf, decoder_conf, periodic_decoder_config=periodic_decoder_config)
   
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
                 checkpoint_path=None,
                 periodic_decoder_config=None,
                 ):
        super(VQ_Align, self).__init__()
        if checkpoint_path:
            print("LOADING VQ.pt CHECKPOINT\n\n\n\n-----------------")
            self.VQ = load_model(checkpoint_path, "cuda", periodic_decoder_config)
        else:
            self.VQ = VQ(encoder_config, decoder_config, periodic_decoder_config=periodic_decoder_config)
            self.VQ.init_ae_layer()
        
    
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
            return loss, encoder_features, log

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
    