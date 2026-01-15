# st_classifier_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence


# -----------------------------------------------------------
# 1-D depth-wise-separable convolution (used by Inception)
# -----------------------------------------------------------
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_c, out_c, k, stride=1, bias=False):
        super().__init__()
        p = (k - 1) // 2
        self.dw = nn.Conv1d(in_c, in_c, k, stride=stride,
                            padding=p, groups=in_c, bias=bias)
        self.pw = nn.Conv1d(in_c, out_c, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor):
        return self.pw(self.dw(x))


# -----------------------------------------------------------
# Channel-wise self-attention
# -----------------------------------------------------------
class ChannelAttention(nn.Module):
    """
    Learns a graph over EEG channels for every patch-position separately.
    Input  : [B, D, M, N]
    Output : [B, D, M, N]  (same shape, channels mixed)
    """
    def __init__(self, d_model: int, heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, heads, batch_first=True)

    def forward(self, x: torch.Tensor):
        B, D, M, N = x.shape
        # Treat “channel” as the sequence dimension of the attention module.
        x = x.permute(0, 3, 2, 1).reshape(B * N, M, D)   # [B·N, M, D]
        out, _ = self.mha(x, x, x)                       # channel-attention
        out = out.reshape(B, N, M, D).permute(0, 3, 2, 1)
        return out                                       # [B, D, M, N]


# -----------------------------------------------------------
# Multi-scale 1-D Inception
# -----------------------------------------------------------
class MultiScaleInception1d(nn.Module):
    def __init__(self, in_c, out_per_branch, use_dw_separable=True, stem_k=9):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_c, in_c, stem_k,
                      padding=(stem_k - 1) // 2,
                      groups=in_c, bias=False),
            nn.BatchNorm1d(in_c),
            nn.GELU(),
        )

        Conv = DepthwiseSeparableConv1d if use_dw_separable else nn.Conv1d
        self.b1 = nn.Conv1d(in_c, out_per_branch, 1, bias=False)
        self.b3 = Conv(in_c, out_per_branch, 3, bias=False)
        self.b5 = Conv(in_c, out_per_branch, 5, bias=False)
        self.b7 = Conv(in_c, out_per_branch, 7, bias=False)
        self.bp = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_c, out_per_branch, 1, bias=False),
        )
        self.bn = nn.BatchNorm1d(out_per_branch * 5)
        self.act = nn.GELU()

    def forward(self, x):                                # [B, C, N]
        x = self.stem(x)
        y = torch.cat([self.b1(x), self.b3(x),
                       self.b5(x), self.b7(x), self.bp(x)], dim=1)
        return self.act(self.bn(y))                      # [B, 5·out, N]


# -----------------------------------------------------------
# Pyramidal temporal pooling
# -----------------------------------------------------------
class PyramidPool1d(nn.Module):
    """
    Three parallel AvgPool1d paths with strides {1, 2, 4},
    each re-interpolated back to the longest path length and concatenated.
    """
    def __init__(self, scales: Sequence[int] = (1, 2, 4)):
        super().__init__()
        self.scales = scales
        self.pools = nn.ModuleList(
            [nn.Identity() if k == 1 else nn.AvgPool1d(k, stride=k)
             for k in scales]
        )

    def forward(self, x):                                # [B, C, N]
        outs = []
        for pool in self.pools:
            o = pool(x)                                  # [B, C, N/k]
            if o.shape[-1] != x.shape[-1]:               # upsample to max-len
                o = F.interpolate(o, size=x.shape[-1],
                                   mode="linear", align_corners=False)
            outs.append(o)
        return torch.cat(outs, dim=1)                    # [B, C·len(scales), N]


# -----------------------------------------------------------
# Spatio-Temporal Classifier v2
# -----------------------------------------------------------
class SpatioTemporalClassifier(nn.Module):
    """
    Spatial channel-attention  →  Inception  →  PyramidPool
    →  TemporalConv  →  Learned-(attention) pooling  →  FFN.
    """
    def __init__(
        self,
        num_classes: int,
        d_model_from_backbone: int,
        num_eeg_channels: int,
        *,
        spatial_out: int = 64,
        inception_out_per_branch: int = 32,
        temporal_out: int = 128,
        temporal_kernel: int = 15,
        ffn_hidden: int = 256,
        dropout: float = 0.3,
        attn_heads: int = 4,
    ):
        super().__init__()
        self.M = num_eeg_channels
        self.D = d_model_from_backbone

        # 0. Channel-wise attention over raw backbone features
        self.chan_attn = ChannelAttention(d_model_from_backbone, heads=attn_heads)

        # 1. Spatial aggregation (still use (M,1) conv but *after* attention)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(self.D, spatial_out,
                      kernel_size=(self.M, 1), bias=False),
            nn.BatchNorm2d(spatial_out),
            nn.GELU(),
        )

        # 2. Inception (1-D)
        self.inception = MultiScaleInception1d(
            spatial_out, inception_out_per_branch, use_dw_separable=True
        )

        # 3. Pyramidal pooling
        self.pyramid = PyramidPool1d(scales=(1, 2, 4))

        # 4. Temporal convolution
        in_c_temporal = inception_out_per_branch * 5 * 3  # ×3 for pyramid concat
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_c_temporal, temporal_out,
                      kernel_size=temporal_kernel,
                      padding=(temporal_kernel - 1) // 2, bias=False),
            nn.BatchNorm1d(temporal_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 5. Learned pooling = single “CLS” token attention
        self.cls_token = nn.Parameter(torch.zeros(1, 1, temporal_out))
        self.cls_attn = nn.MultiheadAttention(temporal_out, 4, batch_first=True)

        # 6. Feed-forward classifier
        self.mlp = nn.Sequential(
            nn.Linear(temporal_out, ffn_hidden),
            nn.BatchNorm1d(ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, num_classes),
        )

    # -------------------------------------------------------
    def forward(self, tokens: torch.Tensor):             # [B·M, N, D]
        B_M, N, D = tokens.shape
        assert D == self.D and B_M % self.M == 0
        B = B_M // self.M

        # reshape for channel attention
        x = tokens.view(B, self.M, N, D).permute(0, 3, 1, 2)     # [B,D,M,N]
        x = self.chan_attn(x)                                    # [B,D,M,N]

        # spatial aggregation
        x = self.spatial_conv(x).squeeze(2)                      # [B,F_s,N]

        # inception + pyramid pooling
        x = self.inception(x)                                    # [B,5F,N]
        x = self.pyramid(x)                                      # [B,15F,N]

        # temporal conv
        x = self.temporal_conv(x)                                # [B,F_t,N]
        x = x.transpose(1, 2)                                    # [B,N,F_t]

        # learned “CLS” pooling
        cls = self.cls_token.expand(B, -1, -1)                   # [B,1,F_t]
        z, _ = self.cls_attn(torch.cat([cls, x], dim=1),
                             torch.cat([cls, x], dim=1),
                             torch.cat([cls, x], dim=1))
        cls_emb = z[:, 0]                                        # [B,F_t]

        return self.mlp(cls_emb)                                 # [B,C]
    