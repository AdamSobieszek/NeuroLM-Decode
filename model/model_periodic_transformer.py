"""
Periodic Transformer module with Snake activation for signal processing
Modified to enforce a minimum frequency in the Snake activation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SnakeActivation(nn.Module):
    """Snake activation with learnable frequency parameter and minimum frequency constraint"""
    def __init__(self, in_features, alpha_init=1.0, min_alpha=1.0):
        super().__init__()
        # Initialize alpha with max(alpha_init, min_alpha) to ensure minimum frequency
        self.alpha = nn.Parameter(torch.ones(in_features) * max(alpha_init, min_alpha))
        self.min_alpha = min_alpha
        
    def forward(self, x):
        # Apply soft minimum constraint during forward pass
        # This ensures alpha doesn't go below minimum frequency even during training
        effective_alpha = self.alpha.abs() + self.min_alpha
        return x + (1.0 / effective_alpha) * torch.sin(effective_alpha * x) ** 2


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
    """Anti-aliased Multi-Periodicity Feed Forward Network with minimum frequency constraint"""
    def __init__(self, dim, hidden_dim=None, dropout=0.0, snake_alpha_init=1.0, min_alpha=1.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        # Multi-scale network for capturing different frequency ranges
        # Now with minimum frequency constraint
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            SnakeActivation(hidden_dim, alpha_init=snake_alpha_init, min_alpha=min_alpha),
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


class PeriodicTransformerLayer(nn.Module):
    """Transformer layer with periodic inductive bias and minimum frequency constraint"""
    def __init__(self, dim, num_heads, dropout=0.0, snake_alpha_init=1.0, min_alpha=1.0):
        super().__init__()
        # Self-attention with pre-normalization
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # AMP feed-forward with varying initialization and minimum frequency constraint
        self.amp_ffn = AMPFeedForward(
            dim, 
            hidden_dim=dim*4, 
            dropout=dropout,
            snake_alpha_init=snake_alpha_init,
            min_alpha=min_alpha
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
    Now with minimum frequency parameter to focus on high-frequency components.
    
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
    min_alpha : float
        Minimum frequency parameter for Snake activation
        Higher values focus more on higher frequency components
    """
    def __init__(self, 
                input_dim=200,
                hidden_dim=512,
                num_layers=4,
                num_heads=12,
                dropout=0.1,
                freq_bands=6,
                min_alpha=2.0):  # Default minimum alpha for high frequencies
        super().__init__()
        
        # Store parameters
        self.input_dim = input_dim
        self.freq_bands = freq_bands
        self.min_alpha = min_alpha
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # Periodic transformer layers with progressively varying frequency initialization
        # and minimum frequency constraint
        self.layers = nn.ModuleList([
            PeriodicTransformerLayer(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                snake_alpha_init=min_alpha * (1.0 + 0.5 * i),  # Progressive increase starting from min_alpha
                min_alpha=min_alpha
            ) for i in range(num_layers)
        ])
        
        # Multi-band frequency decomposition with minimum frequency constraint
        # Each band uses a different initialization for the Snake activation
        # to target different parts of the high-frequency spectrum
        self.freq_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                SnakeActivation(
                    hidden_dim // 2, 
                    alpha_init=min_alpha * (1.0 + 0.5 * i),  # Spread across high frequency spectrum
                    min_alpha=min_alpha
                ),
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
        
        # Input projection
        x = self.input_proj(x)
        
        # Process through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Extract attention weights for frequency bands
        batch_size, seq_len, _ = x.shape
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
        # Make sure components are concatenated to exactly match input_dim
        output = torch.cat(freq_components, dim=-1)
        
        # Always apply final projection to ensure exact dimension matching
        output = self.output_proj(output)
        
        # Verify output shape matches original input shape
        assert output.shape == original_shape, f"Output shape {output.shape} doesn't match input shape {original_shape}"
        
        return output