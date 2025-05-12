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
    """
    def __init__(self, d_model, conv_kernel_size, expansion_factor=2, dropout_rate=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * expansion_factor,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False # Typically False before BatchNorm
        )
        self.activation1 = Swish() # Or nn.GLU(dim=1)

        # If using GLU, channels for depthwise_conv are halved.
        depthwise_channels = d_model * expansion_factor
        if isinstance(self.activation1, nn.GLU):
            # GLU halves the channel dimension (dim=1 for [B, C, T])
            # But pointwise_conv1 outputs d_model * expansion_factor.
            # GLU's input is (N, Cin, Lin) and output is (N, Cin/2, Lin).
            # So, depthwise_conv input channels remain d_model * expansion_factor for GLU.
            # The actual activation splits into two halves.
            # Let's assume activation1 is Swish for simpler channel math here.
            # If GLU: self.glu = nn.GLU(dim=1) instead of self.activation1
            pass


        self.depthwise_conv = nn.Conv1d(
            in_channels=depthwise_channels,
            out_channels=depthwise_channels,
            kernel_size=conv_kernel_size,
            stride=1,
            padding=(conv_kernel_size - 1) // 2,
            groups=depthwise_channels, # Depthwise
            bias=False # Typically False before BatchNorm
        )
        self.batch_norm = nn.BatchNorm1d(depthwise_channels)
        self.activation2 = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=depthwise_channels,
            out_channels=d_model, # Project back
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False # Often False
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x_norm = self.layer_norm(x)
        x_conv = x_norm.transpose(1, 2)  # [B, D, S] for Conv1d

        x_conv = self.pointwise_conv1(x_conv)
        x_conv = self.activation1(x_conv) # If GLU: x_conv = self.glu(x_conv)
        
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
        if d_model % n_head != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_head ({n_head})")
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attention_mask=None):
        # x: [batch_size, seq_len, d_model]
        # attention_mask (key_padding_mask): [batch_size, seq_len] boolean (True for masked)
        
        x_norm = self.layer_norm(x)
        
        # Ensure contiguity before MHA
        query = x_norm.contiguous()
        key = x_norm.contiguous()
        value = x_norm.contiguous()
        
        # attention_mask for nn.MultiheadAttention should be key_padding_mask:
        # (N, S) where N is batch size, S is source sequence length.
        # If True, the corresponding key will be ignored for attention.
        # It doesn't need to be made contiguous in the same way usually, but no harm.
        if attention_mask is not None:
            attention_mask = attention_mask.contiguous()

        attn_output, _ = self.mha(query, key, value, key_padding_mask=attention_mask, need_weights=False)
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
        # x: [batch_size, in_channels, seq_len]
        return self.conv(x)

class LayerGatedConformerBlock(nn.Module):
    """
    Implements a single Layer-Gated Conformer Block.
    Order: FFN -> MHSA -> Conv -> FFN, each with residual. Then, gating.
    """
    def __init__(self, d_model, n_head, d_ff, conv_kernel_size, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model

        self.ffn1 = FeedForwardModule(d_model, d_ff, dropout_rate)
        self.mhsa = MultiHeadSelfAttentionModule(d_model, n_head, dropout_rate)
        self.conv_module = ConvolutionModule(d_model, conv_kernel_size, dropout_rate=dropout_rate)
        self.ffn2 = FeedForwardModule(d_model, d_ff, dropout_rate)
        self.final_layer_norm = nn.LayerNorm(d_model)

        self.zero_conv_gate = ZeroConv1d(d_model, d_model)
        self.ffn_gate_linear = nn.Linear(d_model, d_model) # Simplified FFN for gate
        self.sigmoid_gate = nn.Sigmoid()

    def _conformer_transform(self, x_input, attention_mask=None):
        x = x_input
        
        # FFN1 + residual
        x = x + 0.5 * self.ffn1(x) # Common to scale FFN residual

        # MHSA + residual
        x = x + self.mhsa(x, attention_mask=attention_mask)
        
        # ConvModule + residual
        x = x + self.conv_module(x)

        # FFN2 + residual
        x = x + 0.5 * self.ffn2(x) # Common to scale FFN residual

        transformed_x = self.final_layer_norm(x)
        return transformed_x

    def forward(self, x_prev_layer, attention_mask=None):
        # x_prev_layer (e_l_minus_1): [batch_size, seq_len, d_model]
        
        conformer_transformed_x = self._conformer_transform(x_prev_layer, attention_mask)

        gate_input_conv = x_prev_layer.transpose(1, 2) # [B, d_model, seq_len]
        g_val_conv = self.zero_conv_gate(gate_input_conv)   # [B, d_model, seq_len]
        
        g_val_linear_input = g_val_conv.transpose(1, 2) # [B, seq_len, d_model]
        g_val_linear = self.ffn_gate_linear(g_val_linear_input) # [B, seq_len, d_model]
        g_val = self.sigmoid_gate(g_val_linear) # [B, seq_len, d_model]

        output_x = g_val * conformer_transformed_x + (1 - g_val) * x_prev_layer
        
        return output_x

class LBLMConformerBackbone(nn.Module):
    """
    Multi-layer Conformer backbone using LayerGatedConformerBlocks.
    """
    def __init__(self, 
                 input_dim,
                 num_layers, 
                 d_model,
                 n_head, 
                 d_ff, 
                 conv_kernel_size, 
                 dropout_rate=0.1):
        super().__init__()

        if input_dim != d_model:
            self.input_projection = nn.Linear(input_dim, d_model)
            # print(f"LBLMConformerBackbone: Projecting input_dim {input_dim} to d_model {d_model}")
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
        # x_embedded_patches: [batch_size_conform, seq_len, input_dim]
        # attention_mask (key_padding_mask): [batch_size_conform, seq_len]
        
        x = self.input_projection(x_embedded_patches)
        
        for i in range(self.num_layers):
            x = self.layers[i](x, attention_mask=attention_mask)
            
        return x
