import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Swish(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConvolutionModule(nn.Module):
    """
    Convolution Module as typically used in Conformer blocks.
    Order: LayerNorm -> PointwiseConv (expand) -> GLU/Swish -> DepthwiseConv -> BatchNorm -> Swish -> PointwiseConv (project) -> Dropout.
    The LBLM paper's Appendix A.1 (Figure A.1) is the definitive source if it differs.
    This is a common interpretation.
    """
    def __init__(self, d_model, conv_kernel_size, expansion_factor=2, dropout_rate=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * expansion_factor, # Expand
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        # GLU is often used: one half for sigmoid gate, other half for value
        # Or simply an activation like Swish
        self.activation1 = Swish() # Or nn.GLU(dim=1) if using GLU

        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model * expansion_factor // (2 if isinstance(self.activation1, nn.GLU) else 1), # Halved if GLU
            out_channels=d_model * expansion_factor // (2 if isinstance(self.activation1, nn.GLU) else 1),
            kernel_size=conv_kernel_size,
            stride=1,
            padding=(conv_kernel_size - 1) // 2,
            groups=d_model * expansion_factor // (2 if isinstance(self.activation1, nn.GLU) else 1), # Depthwise
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(d_model * expansion_factor // (2 if isinstance(self.activation1, nn.GLU) else 1))
        self.activation2 = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model * expansion_factor // (2 if isinstance(self.activation1, nn.GLU) else 1),
            out_channels=d_model, # Project back
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x_norm = self.layer_norm(x)
        x_conv = x_norm.transpose(1, 2)  # [B, D, S] for Conv1d

        x_conv = self.pointwise_conv1(x_conv)
        x_conv = self.activation1(x_conv) # If GLU, x_conv is now [B, D_expanded/2, S]
        
        x_conv = self.depthwise_conv(x_conv)
        x_conv = self.batch_norm(x_conv)
        x_conv = self.activation2(x_conv)
        
        x_conv = self.pointwise_conv2(x_conv)
        x_conv = self.dropout(x_conv)
        
        return x_conv.transpose(1, 2) # [B, S, D]

class FeedForwardModule(nn.Module):
    """Standard FeedForward module: LayerNorm -> Linear -> Swish -> Dropout -> Linear -> Dropout"""
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = Swish()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x_ff = self.linear1(x_norm)
        x_ff = self.activation(x_ff)
        x_ff = self.dropout1(x_ff)
        x_ff = self.linear2(x_ff)
        x_ff = self.dropout2(x_ff)
        return x_ff

class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, d_model, n_head, dropout_rate=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attention_mask=None):
        # x: [batch_size, seq_len, d_model]
        # MHA expects query, key, value. For self-attention, they are the same.
        # MHA also expects mask: (N, S) where N is batch size, S is source sequence length.
        # Or (N*num_heads, S, S) for additive attention mask.
        # For padding mask, it should be [batch_size, seq_len] bool tensor (True where padded).
        x_norm = self.layer_norm(x)
        attn_output, _ = self.mha(x_norm, x_norm, x_norm, key_padding_mask=attention_mask, need_weights=False)
        attn_output = self.dropout(attn_output)
        return attn_output

class ZeroConv1d(nn.Module):
    """1x1 Conv1d with weights and bias initialized to zero."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # x: [batch_size, d_model, seq_len]
        return self.conv(x)

class LayerGatedConformerBlock(nn.Module):
    """
    Implements a single Layer-Gated Conformer Block as described.
    The standard Conformer structure: FFN -> MHSA -> Conv -> FFN, each with residual.
    Then, the entire block's output is gated with its input.
    """
    def __init__(self, d_model, n_head, d_ff, conv_kernel_size, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model

        # Standard Conformer modules
        # LBLM Appendix A.1: "Conformer block... consists four modules, two feed-forward layer,
        # a convolution module, and a multi-head self-attention (MHSA) layer."
        self.ffn1 = FeedForwardModule(d_model, d_ff, dropout_rate)
        self.mhsa = MultiHeadSelfAttentionModule(d_model, n_head, dropout_rate)
        self.conv_module = ConvolutionModule(d_model, conv_kernel_size, dropout_rate=dropout_rate)
        self.ffn2 = FeedForwardModule(d_model, d_ff, dropout_rate)
        self.final_layer_norm = nn.LayerNorm(d_model) # Applied after the four modules

        # Layer Gating mechanism components
        # "submodule comprising a zero convolution layer, a feed-forward layer, and a sigmoid function"
        self.zero_conv_gate = ZeroConv1d(d_model, d_model)
        # The "feed-forward layer" for gating could be simple. Let's use a single Linear.
        self.ffn_gate = nn.Linear(d_model, d_model)
        self.sigmoid_gate = nn.Sigmoid()

    def _conformer_transform(self, x_input, attention_mask=None):
        # x_input: [batch_size, seq_len, d_model]
        
        # Apply the four modules sequentially with residual connections
        # Note: Conformer often uses pre-norm. The modules above include their own LayerNorm.
        # The 0.5 factor for FFN residuals is common in some Conformer implementations.
        
        # 1. First FeedForward
        x_ffn1 = self.ffn1(x_input)
        x_residual = x_input + 0.5 * x_ffn1 # Or x_input + x_ffn1

        # 2. Multi-Head Self-Attention
        x_mhsa = self.mhsa(x_residual, attention_mask=attention_mask)
        x_residual = x_residual + x_mhsa
        
        # 3. Convolution Module
        x_conv = self.conv_module(x_residual)
        x_residual = x_residual + x_conv

        # 4. Second FeedForward
        x_ffn2 = self.ffn2(x_residual)
        x_residual = x_residual + 0.5 * x_ffn2 # Or x_residual + x_ffn2

        # 5. Final LayerNorm for the block's transformation output
        transformed_x = self.final_layer_norm(x_residual)
        return transformed_x

    def forward(self, x_prev_layer, attention_mask=None):
        # x_prev_layer (e_l_minus_1): [batch_size, seq_len, d_model]
        
        # 1. Get the Conformer transformation output (e_prime_l)
        conformer_transformed_x = self._conformer_transform(x_prev_layer, attention_mask)

        # 2. Calculate gating values g(e_l_minus_1)
        # Input to Conv1d needs to be [B, D, S]
        gate_input = x_prev_layer.transpose(1, 2) # [B, d_model, seq_len]
        g_val = self.zero_conv_gate(gate_input)   # [B, d_model, seq_len]
        
        # Input to Linear needs to be [B, S, D]
        g_val = g_val.transpose(1, 2)             # [B, seq_len, d_model]
        g_val = self.ffn_gate(g_val)              # [B, seq_len, d_model]
        g_val = self.sigmoid_gate(g_val)          # [B, seq_len, d_model], token-wise gates

        # 3. Apply gating
        # e_l = g(e_l_minus_1) * e_prime_l + (1 - g(e_l_minus_1)) * e_l_minus_1
        output_x = g_val * conformer_transformed_x + (1 - g_val) * x_prev_layer
        
        return output_x

class LBLMConformerBackbone(nn.Module):
    """
    Multi-layer Conformer backbone using LayerGatedConformerBlocks.
    Takes embedded patches as input.
    """
    def __init__(self, 
                 input_dim, # Dimension of the input patch embeddings
                 num_layers, 
                 d_model,   # Dimension used throughout the Conformer blocks
                 n_head, 
                 d_ff, 
                 conv_kernel_size, 
                 dropout_rate=0.1):
        super().__init__()

        if input_dim != d_model:
            self.input_projection = nn.Linear(input_dim, d_model)
            print(f"LBLMConformerBackbone: Projecting input_dim {input_dim} to d_model {d_model}")
        else:
            self.input_projection = nn.Identity()

        self.layers = nn.ModuleList([
            LayerGatedConformerBlock(
                d_model=d_model,
                n_head=n_head,
                d_ff=d_ff,
                conv_kernel_size=conv_kernel_size,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x_embedded_patches, attention_mask=None):
        # x_embedded_patches: [batch_size * num_channels, num_patches_per_channel, input_dim]
        # attention_mask (key_padding_mask for MHA): [batch_size * num_channels, num_patches_per_channel]
        #                                            (True where padded/masked, False otherwise)
        
        x = self.input_projection(x_embedded_patches)
        
        for i in range(self.num_layers):
            x = self.layers[i](x, attention_mask=attention_mask)
            
        return x # Output: [batch_size * num_channels, num_patches_per_channel, d_model]