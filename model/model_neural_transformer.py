"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

import math
import torch.nn as nn
from model.model import Block
from einops import rearrange
from dataclasses import dataclass


class TemporalConv(nn.Module):
    """ EEG to Patch Embedding
    """
    def __init__(self, in_chans=1, out_chans=8):
        '''
        in_chans: in_chans of nn.Conv2d()
        out_chans: out_chans of nn.Conv2d(), determing the output dimension
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()
        self.l = nn.Sequential(
            nn.Linear(400, 768),
            nn.GELU()
        )

    def forward(self, x, **kwargs):
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        x = self.l(x)
        return x


@dataclass
class NTConfig:
    block_size: int = 1024
    patch_size: int = 200
    num_classes: int = 0
    in_chans: int = 1
    out_chans: int = 16
    use_mean_pooling: bool = True
    init_scale: float = 0.001
    n_layer: int = 12
    n_head: int = 10
    n_embd: int = 400
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class NeuralTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.num_classes = config.num_classes

        # To identify whether it is neural tokenizer or neural decoder. 
        # For the neural decoder, use linear projection (PatchEmbed) to project codebook dimension to hidden dimension.
        # Otherwise, use TemporalConv to extract temporal features from EEG signals.
        self.patch_embed = TemporalConv(out_chans=config.out_chans) if config.in_chans == 1 else nn.Linear(config.in_chans, config.n_embd)
        self.patch_size = config.patch_size

        self.pos_embed = nn.Embedding(256, config.n_embd)
        self.time_embed = nn.Embedding(64, config.n_embd)

        self.rel_pos_bias = None

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = nn.Identity() if config.use_mean_pooling else nn.LayerNorm(config.n_embd, eps=1e-6)
        self.fc_norm = nn.LayerNorm(config.n_embd, eps=1e-6) if config.use_mean_pooling else None
        self.head = nn.Linear(config.n_embd, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.pos_drop = nn.Dropout(p=config.dropout)

        if isinstance(self.head, nn.Linear):
            nn.init.trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(config.init_scale)
            self.head.bias.data.mul_(config.init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.c_proj.weight.data, layer_id + 1)
            rescale(layer.mlp.c_proj.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, input_chans=None, input_times=None, mask=None, return_all_tokens=False, **kwargs):
        batch_size, n, t = x.shape
        x = self.patch_embed(x)

        # add position and temporal embeddings
        pos_embed_used = self.pos_embed(input_chans)
        x = x + pos_embed_used
        time_embed = self.time_embed(input_times)
        x = x + time_embed

        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x, mask)
        
        x = self.norm(x)
        if self.fc_norm is not None:
            if return_all_tokens:
                return self.fc_norm(x)
            else:
                return self.fc_norm(x.mean(1))
        else:
            return x

    def forward(self, x, input_chans=None, input_times=None, mask=None, return_all_tokens=False, **kwargs):
        '''
        x: [batch size, sequence length, patch size]
        '''
        x = self.forward_features(x, input_chans, input_times, mask, return_all_tokens=return_all_tokens, **kwargs)
        x = self.head(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

# your standard small 1D U‑Net
class UNet1DDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, depth=4):
        super().__init__()
        # down path
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for d in range(depth):
            out_ch = base_channels * (2**d)
            self.downs.append(nn.Sequential(
                nn.Conv1d(ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch), nn.GELU(),
                nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch), nn.GELU(),
            ))
            self.pools.append(nn.MaxPool1d(2))
            ch = out_ch

        # up path
        self.ups = nn.ModuleList()
        self.ups_convs = nn.ModuleList()
        for d in reversed(range(depth)):
            in_ch = ch
            out_ch = base_channels * (2**d)
            self.ups.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2))
            # after concat with skip, channels = out_ch*2
            self.ups_convs.append(nn.Sequential(
                nn.Conv1d(out_ch*2, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch), nn.GELU(),
                nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch), nn.GELU(),
            ))
            ch = out_ch

        self.final = nn.Conv1d(ch, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [B, T, C] -> [B, C, T]
        x = x.transpose(1,2)
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        for up, upconv, skip in zip(self.ups, self.ups_convs, reversed(skips)):
            x = up(x)
            # pad if needed
            if x.size(-1) != skip.size(-1):
                x = F.pad(x, (0, skip.size(-1)-x.size(-1)))
            x = torch.cat([x, skip], dim=1)
            x = upconv(x)
        x = self.final(x)
        # back to [B, T, C_out]
        return x.transpose(1,2)

class HybridUNetDecoder(NeuralTransformer):
    def __init__(self, config: NTConfig,
                 unet_base_ch: int = 64,
                 unet_depth: int = 4):
        super().__init__(config)
        # kill the old head, we’ll do our own reconstruction
        self.head = nn.Identity()
        self.unet = UNet1DDecoder(
            in_channels  = config.n_embd,
            out_channels = config.patch_size,
            base_channels= unet_base_ch,
            depth        = unet_depth
        )

    def forward(self,
                x,                   # [B, seq_len, patch_size]
                input_chans=None,
                input_times=None,
                mask=None,
                return_all_tokens: bool = False,
                **kwargs):
        # 1) run the standard embedding + transformer encoder
        #    but always ask for full-token output
        feats = self.forward_features(
            x,
            input_chans,
            input_times,
            mask,
            return_all_tokens=True
        )  # => [B, seq_len, n_embd]

        # 2) reconstruct waveform with U‑Net
        recon = self.unet(feats)  # => [B, seq_len, patch_size]

        # 3) exactly mirror the original return logic:
        #    if they wanted all tokens, hand back the feature‐matrix,
        #    otherwise hand back the reconstruction.
        if return_all_tokens:
            return feats       # exactly one tensor
        else:
            return recon       # exactly one tensor

