"""
Periodic Transformer module with Snake activation for signal processing
Updated to use dilated convolutions and transposed convolutions inspired by BigVGAN
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SnakeActivation(nn.Module):
    """Snake activation with learnable frequency parameter"""
    def __init__(self, in_features, alpha_init=1.0):
        super().__init__()
        # Initialize alpha directly with alpha_init
        self.alpha = nn.Parameter(torch.ones(in_features) * alpha_init)
        
    def forward(self, x):
        # Standard Snake activation without minimum constraint
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x) ** 2


class LowPassFilter(nn.Module):
    """Low-pass filter for anti-aliasing"""
    def __init__(self, cutoff=0.5, kernel_size=6):
        super().__init__()
        self.kernel_size = kernel_size
        self.cutoff = cutoff
        self.register_buffer('filter', self._build_filter())
        
    def _build_filter(self):
        # Kaiser window sinc filter
        half_size = self.kernel_size // 2
        t = torch.arange(-half_size, half_size + 1, dtype=torch.float32)
        window = torch.kaiser_window(2 * half_size + 1)
        sinc = torch.sin(2 * math.pi * self.cutoff * t) / (math.pi * t + 1e-8)  # Avoid division by zero
        sinc[half_size] = 2 * self.cutoff  # Handle division by zero at center
        filter_kernel = window * sinc
        return filter_kernel.view(1, 1, -1) / filter_kernel.sum()
    
    def forward(self, x):
        # Ensure we have the right shape for 1D convolution [B, C, T]
        orig_shape = x.shape
        if len(orig_shape) == 3 and orig_shape[2] > orig_shape[1]:  # [B, C, T]
            x_reshaped = x
        else:  # [B, T, C]
            x_reshaped = x.transpose(1, 2)
        
        # Apply filter to each channel independently
        x_filtered = F.conv1d(
            x_reshaped, 
            self.filter.repeat(x_reshaped.size(1), 1, 1), 
            padding=self.kernel_size//2, 
            groups=x_reshaped.size(1)
        )
        
        # Return to original shape if needed
        if len(orig_shape) == 3 and orig_shape[2] > orig_shape[1]:  # Was already [B, C, T]
            return x_filtered
        else:  # Was [B, T, C]
            return x_filtered.transpose(1, 2)


class AMPFeedForward(nn.Module):
    """Anti-aliased Multi-Periodicity Feed Forward Network"""
    def __init__(self, dim, hidden_dim=None, dropout=0.0, snake_alpha_init=1.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        # Network with Snake activation
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            SnakeActivation(hidden_dim, alpha_init=snake_alpha_init),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Anti-aliasing filter
        self.low_pass = LowPassFilter(cutoff=0.5, kernel_size=6)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Process through MLP with Snake activation
        h = self.net(x)
        
        # Apply low-pass filter (anti-aliasing)
        h_filtered = self.low_pass(h)
        
        # Residual connection and normalization
        x = x + h_filtered
        x = self.norm(x)
        
        return x


class AMPBlock(nn.Module):
    """AMP Block with dilated convolutions inspired by BigVGAN"""
    def __init__(self, channels, kernel_size=3, dilation_rates=(1, 3, 5), snake_alpha_init=1.0):
        super().__init__()
        
        # Dilated convolutions
        self.convs = nn.ModuleList([
            nn.Conv1d(
                channels, 
                channels, 
                kernel_size=kernel_size, 
                stride=1, 
                dilation=d,
                padding=(kernel_size-1)//2 * d  # Padding to maintain sequence length
            )
            for d in dilation_rates
        ])
        
        # Snake activations for each conv layer
        self.activations = nn.ModuleList([
            SnakeActivation(channels, alpha_init=snake_alpha_init * (1.0 + 0.25 * i))
            for i in range(len(dilation_rates))
        ])
        
        # Low-pass filters for anti-aliasing
        self.low_pass_filters = nn.ModuleList([
            LowPassFilter(cutoff=0.5, kernel_size=6)
            for _ in range(len(dilation_rates))
        ])
        
    def forward(self, x):
        """
        Forward pass for AMPBlock
        x shape: [B, C, T] - batch, channels, time
        """
        # Store input for residual connection
        residual = x
        
        # Process through each dilated conv with activation
        for conv, activation, low_pass in zip(self.convs, self.activations, self.low_pass_filters):
            # Apply activation and then convolution
            x_act = activation(x.transpose(1, 2)).transpose(1, 2)  # [B, T, C] -> [B, C, T]
            x_conv = conv(x_act)
            # Apply low-pass filter
            x_filtered = low_pass(x_conv)
            # Add residual connection
            x = x_filtered + residual
            # Update residual for next layer
            residual = x
            
        return x


class PeriodicTransformerLayer(nn.Module):
    """Transformer layer with periodic inductive bias"""
    def __init__(self, dim, num_heads, dropout=0.0, snake_alpha_init=1.0):
        super().__init__()
        # Self-attention with pre-normalization
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # AMP feed-forward with varying initialization
        self.amp_ffn = AMPFeedForward(
            dim, 
            hidden_dim=dim*4, 
            dropout=dropout,
            snake_alpha_init=snake_alpha_init
        )
        
    def forward(self, x, mask=None):
        # Self-attention with pre-norm and residual connection
        attn_out = self.norm1(x)
        attn_out, _ = self.self_attn(attn_out, attn_out, attn_out, key_padding_mask=mask)
        x = x + attn_out
        
        # AMP feed-forward 
        x = self.amp_ffn(x)
        
        return x


class PeriodicTransformerDecoder(nn.Module):
    """
    Transformer decoder with periodic inductive bias for signal generation.
    Includes transposed convolutions and dilated convolutions inspired by BigVGAN.
    
    Parameters:
    -----------
    input_dim : int
        Dimension of the input signal
    hidden_dim : int
        Internal dimension of the transformer
    num_layers : int
        Number of transformer layers
    num_heads : int
        Number of attention heads per layer
    dropout : float
        Dropout probability
    freq_bands : int
        Number of frequency bands for the output
    """
    def __init__(self, 
                input_dim=200,
                hidden_dim=512,
                num_layers=4,
                num_heads=12,
                dropout=0.1,
                freq_bands=6):
        super().__init__()
        
        # Store parameters
        self.input_dim = input_dim
        self.freq_bands = freq_bands
        
        # Initial transposed 1D convolution (upsampling step from BigVGAN)
        self.initial_transpose_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),  # Project to hidden_dim
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),  # Upsample
            SnakeActivation(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)  # Adjust channels
        )
        
        # AMP Block for periodic signal processing
        self.amp_block = AMPBlock(
            channels=hidden_dim,
            kernel_size=3,
            dilation_rates=(1, 3, 5, 7),  # Multiple dilation rates for different frequency ranges
            snake_alpha_init=1.0
        )
        
        # Project back to transformer format
        self.conv_to_transformer = nn.Linear(hidden_dim, hidden_dim)
        
        # Periodic transformer layers with progressively varying frequency initialization
        self.layers = nn.ModuleList([
            PeriodicTransformerLayer(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                snake_alpha_init=1.0 + 0.5 * i  # Progressive increase in frequency
            ) for i in range(num_layers)
        ])
        
        # Multi-band frequency decomposition
        # Each band uses a different initialization for the Snake activation
        self.freq_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                SnakeActivation(hidden_dim // 2, alpha_init=1.0 + 0.5 * i),  # Different frequencies
                nn.Linear(hidden_dim // 2, input_dim // self.freq_bands + (1 if i < input_dim % self.freq_bands else 0))
            ) for i in range(self.freq_bands)
        ])
        
        # Band mixer with attention to learn optimal combination of frequency bands
        self.band_mixer = nn.Sequential(
            nn.Linear(hidden_dim, self.freq_bands),
            nn.Softmax(dim=-1)
        )
        
        # Final output projection
        self.output_proj = nn.Linear(input_dim, input_dim)
        
    def forward(self, x, mask=None):
        # Store original shape for verification
        original_shape = x.shape
        batch_size, seq_len, feat_dim = x.shape
        
        # Process through initial transposed convolution
        # First transpose to [B, D, T] for convolutional layers
        x_conv = x.transpose(1, 2)  # [B, T, D] -> [B, D, T]
        x_conv = self.initial_transpose_conv(x_conv)  # Apply transposed conv
        
        # Apply AMP Block with dilated convolutions
        x_conv = self.amp_block(x_conv)
        
        # Transpose back to transformer format [B, T', D]
        # Note: The sequence length might be different due to upsampling
        x_conv = x_conv.transpose(1, 2)
        
        # Downsample back to original sequence length if needed
        if x_conv.size(1) != seq_len:
            x_conv = F.interpolate(x_conv.transpose(1, 2), size=seq_len, mode='linear', align_corners=False)
            x_conv = x_conv.transpose(1, 2)
            
        # Project back to transformer hidden dimension
        x = self.conv_to_transformer(x_conv)
        
        # Process through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Extract attention weights for frequency bands
        band_weights = self.band_mixer(x)  # [B, seq_len, freq_bands]
        
        # Generate multiple frequency bands
        freq_components = []
        for i, freq_extractor in enumerate(self.freq_extractors):
            # Extract this frequency band
            component = freq_extractor(x)
            
            # Weight by the attention weights
            weight = band_weights[:, :, i].unsqueeze(-1)
            weighted_component = component * weight
            
            freq_components.append(weighted_component)
        
        # Combine frequency bands into final output
        output = torch.cat(freq_components, dim=-1)
        
        # Always apply final projection to ensure exact dimension matching
        output = self.output_proj(output)
        
        # Verify output shape matches original input shape
        assert output.shape == original_shape, f"Output shape {output.shape} doesn't match input shape {original_shape}"
        
        return output