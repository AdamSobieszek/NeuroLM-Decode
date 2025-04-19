"""
Periodic Transformer module with Snake activation for signal processing
Updated to correctly handle different tensor layouts in convolutional and transformer components
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SnakeActivation(nn.Module):
    """Snake activation with learnable frequency parameter for transformer format [B, T, C]"""
    def __init__(self, in_features, alpha_init=1.0):
        super().__init__()
        # Initialize alpha directly with alpha_init
        self.alpha = nn.Parameter(torch.ones(in_features) * alpha_init)
        
    def forward(self, x):
        # Standard Snake activation for transformer format [B, T, C]
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x) ** 2  


class SnakeActivation1D(nn.Module):
    """Snake activation with learnable frequency parameter for CNN format [B, C, T]"""
    def __init__(self, channels, alpha_init=1.0):
        super().__init__()
        # Initialize alpha directly with alpha_init
        self.alpha = nn.Parameter(torch.ones(channels) * alpha_init)
        
    def forward(self, x):
        # For CNN format [B, C, T], alpha needs to be reshaped for proper broadcasting
        # Need to reshape alpha to [1, C, 1] to broadcast against [B, C, T]
        alpha_reshaped = self.alpha.view(1, -1, 1)
        return x + (1.0 / alpha_reshaped) * torch.sin(alpha_reshaped * x) ** 2


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
        # Assumes input is in CNN format [B, C, T]
        # Apply filter to each channel independently
        x_filtered = F.conv1d(
            x, 
            self.filter.repeat(x.size(1), 1, 1), 
            padding=self.kernel_size//2, 
            groups=x.size(1)
        )
        return x_filtered


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
        
        # Transpose for low-pass filtering [B, T, C] -> [B, C, T]
        h_conv = h.transpose(1, 2)
        h_filtered = self.low_pass(h_conv)
        h_filtered = h_filtered.transpose(1, 2)  # Back to [B, T, C]
        
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
        
        # Snake activations for convolutional format
        self.activations = nn.ModuleList([
            SnakeActivation1D(channels, alpha_init=snake_alpha_init * (1.0 + 0.25 * i))
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
            x_act = activation(x)  # Already in [B, C, T] format
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
    """
    def __init__(self, 
                input_dim=200,
                hidden_dim=384,
                num_layers=4,
                num_heads=6,
                dropout=0.1,
                freq_bands=6):
        super().__init__()
        
        # Store parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.freq_bands = freq_bands
        
        # Simple 1D convolution to adjust dimensions - more stable than transposed conv
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            SnakeActivation1D(hidden_dim, alpha_init=1.0)
        )
        
        # AMP Block for periodic signal processing
        self.amp_block = AMPBlock(
            channels=hidden_dim,
            kernel_size=3,
            dilation_rates=(1, 3, 5, 7),  # Multiple dilation rates for different frequency ranges
            snake_alpha_init=1.0
        )
        
        # Projection back to transformer format
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Periodic transformer layers with progressively varying frequency initialization
        self.layers = nn.ModuleList([
            PeriodicTransformerLayer(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                snake_alpha_init=1.0 + 0.5 * i  # Progressive increase in frequency
            ) for i in range(num_layers)
        ])
        
        # Final frequency band extractors
        # Calculate base size and remainder for even distribution
        base_size = input_dim // freq_bands
        remainder = input_dim % freq_bands
        
        # Create extractors with appropriate output sizes
        self.freq_extractors = nn.ModuleList()
        for i in range(freq_bands):
            # Add an extra feature to some bands if needed for exact dimension matching
            output_size = base_size + (1 if i < remainder else 0)
            
            self.freq_extractors.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                SnakeActivation(hidden_dim // 2, alpha_init=1.0 + 0.5 * i),
                nn.Linear(hidden_dim // 2, output_size)
            ))
        
        # Band mixer with attention to learn optimal combination
        self.band_mixer = nn.Sequential(
            nn.Linear(hidden_dim, freq_bands),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, mask=None):
        """
        Forward pass for the periodic transformer decoder
        x: tensor of shape [batch_size, seq_len, input_dim]
        mask: optional mask for the transformer
        """
        # Store original shape for verification
        original_shape = x.shape
        batch_size, seq_len, _ = x.shape
        
        # Convert to convolutional format [B, C, T]
        x_conv = x.transpose(1, 2)
        
        # Apply input projection
        x_conv = self.input_projection(x_conv)
        
        # Apply AMP block with dilated convolutions
        x_conv = self.amp_block(x_conv)
        
        # Convert back to transformer format [B, T, C]
        x_trans = x_conv.transpose(1, 2)
        
        # Apply projection to transformer hidden dim
        x_trans = self.projection(x_trans)
        
        # Process through transformer layers
        for layer in self.layers:
            x_trans = layer(x_trans, mask)
        
        # Extract band weights for frequency mixing
        band_weights = self.band_mixer(x_trans)  # [B, T, freq_bands]
        
        # Generate multiple frequency bands
        freq_components = []
        for i, extractor in enumerate(self.freq_extractors):
            # Extract this frequency band
            component = extractor(x_trans)  # [B, T, output_size]
            
            # Apply band weighting
            weight = band_weights[:, :, i].unsqueeze(-1)  # [B, T, 1]
            weighted_component = component * weight
            
            freq_components.append(weighted_component)
        
        # Concatenate frequency bands to form final output
        output = torch.cat(freq_components, dim=-1)
        
        # Verify shape matches original input
        assert output.shape == original_shape, f"Output shape {output.shape} doesn't match input shape {original_shape}"
        
        return output