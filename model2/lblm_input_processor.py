import torch
import torch.nn as nn
import math
import numpy as np

class LBLMInputProcessor(nn.Module):
    def __init__(self, patch_length, patch_stride, embed_dim, num_subjects, subject_embed_dim,
                 mask_ratio=0.10, use_rev_in=True, eeg_channels=None):
        super().__init__()
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.use_rev_in = use_rev_in
        self.eeg_channels = eeg_channels

        self.patch_embedder = nn.Linear(patch_length, embed_dim)

        # Max patches: (L_max - P)/S + 1. Assume L_max like e.g. 2s * 250Hz = 500 samples
        # (500 - 25) / 6 + 1 = 475 / 6 + 1 = 79 + 1 = 80.
        # Let's set a reasonable upper bound, adjust if actual sequences are longer.
        max_patches_per_channel = (500 - patch_length) // patch_stride + 200 # Heuristic, or pass as arg
        if max_patches_per_channel <= 0: max_patches_per_channel = 300 # Fallback
        
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_patches_per_channel, embed_dim), requires_grad=False)
        self._init_positional_encoding(max_patches_per_channel, embed_dim)
        self.max_patches_per_channel = 30

        if num_subjects > 0 and subject_embed_dim > 0:
            self.subject_gain = nn.Embedding(num_subjects, subject_embed_dim) # Learnable scalar gain
            nn.init.ones_(self.subject_gain.weight) # Initialize to ones
            # assert subject_embed_dim in [1, embed_dim, embed_dim//2 ], "subject_embed_dim must be 1 or embed_dim"
        else:
            self.subject_gain = None
    def cut_subject_gain(self): # TODO: This is a hack to change stride, remove
        """Properly resize the subject gain embedding by creating a new Embedding layer with reduced size"""
        if self.subject_gain is not None:
            old_weight = self.subject_gain.weight[:,:-1]  # Remove last dimension
            new_embed_dim = old_weight.shape[1]
            num_subjects = self.subject_gain.num_embeddings
            
            # Create new embedding layer with reduced size
            new_subject_gain = nn.Embedding(num_subjects, new_embed_dim).to(self.subject_gain.weight.device)
            
            # Copy over the weights
            with torch.no_grad():
                new_subject_gain.weight.copy_(old_weight)
                
            # Replace old embedding
            self.subject_gain = new_subject_gain

    def _init_positional_encoding(self, max_len, d_model):
        if max_len <=0:
            raise ValueError("max_len for positional encoding must be positive.")
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1 : # Handle d_model = 1 case for cos
            pe[:, 1::2] = torch.cos(position * div_term)
        self.positional_encoding.data.copy_(pe.unsqueeze(0))

    def rev_in_norm(self, x, eps=1e-5):
        # x shape: [B, M, NumPatches, PatchLength]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized_x = (x - mean) / (std + eps)
        return normalized_x, mean, std

    def rev_in_denorm(self, x_norm, mean, std):
        # x_norm, mean, std can be [TotalMasked, PatchLen] or [TotalMasked, 1]
        return x_norm * std + mean

    def forward(self, x_raw_eeg, subject_ids=None):
        # x_raw_eeg: [batch_size, num_channels, num_timepoints]
        batch_size, num_channels, num_timepoints = x_raw_eeg.shape

        if self.eeg_channels is None:
            self.eeg_channels = num_channels
        elif self.eeg_channels != num_channels:
            raise ValueError(f"Input x_raw_eeg has {num_channels} channels, but processor configured for {self.eeg_channels}")

        # 1. Patching
        # Ensure num_timepoints is sufficient for at least one patch
        if num_timepoints < self.patch_length:
             raise ValueError(f"num_timepoints ({num_timepoints}) is less than patch_length ({self.patch_length}). Cannot create patches.")
        patches = x_raw_eeg.unfold(dimension=-1, size=self.patch_length, step=self.patch_stride)
        # patches: [B, M, NumPatchesDerived, PatchLength]
        num_patches_seq = patches.shape[2]
        if num_patches_seq == 0:
            raise ValueError(f"Unfolding resulted in 0 patches. num_timepoints={num_timepoints}, patch_length={self.patch_length}, stride={self.patch_stride}")

        original_patches_for_target = patches.clone() 

        # 2. Reversible Instance Normalization (Optional)
        rev_in_mean, rev_in_std = None, None
        if self.use_rev_in:
            patches, rev_in_mean, rev_in_std = self.rev_in_norm(patches) # mean/std: [B,M,NumPatches,1]

        # 3. Embed patches
        embedded_patches = self.patch_embedder(patches.reshape(-1, self.patch_length))
        # embedded_patches: [B*M*NumPatches, embed_dim] -> [B*M, NumPatches, embed_dim]
        embedded_patches = embedded_patches.reshape(batch_size * num_channels, num_patches_seq, self.embed_dim)

        # 4. Add Positional Encoding
        if num_patches_seq > self.positional_encoding.shape[1]:
            # Dynamically extend positional encoding if needed (or error out)
            # For now, error out as LBLM implies fixed size or pre-calculated max.
            raise ValueError(f"Actual num_patches_seq ({num_patches_seq}) exceeds pre-calculated max_len for positional encoding ({self.positional_encoding.shape[1]}). Adjust max_patches_per_channel in init.")

        embedded_patches = embedded_patches + self.positional_encoding[:, :num_patches_seq, :]

        # 5. Apply Subject Embedding
        if self.subject_gain is not None and subject_ids is not None:
            if subject_ids.max() >= self.subject_gain.num_embeddings:
                raise ValueError(f"subject_id out of range for subject_gain embedding. Got {subject_ids.max()}, max allowed is {self.subject_gain.num_embeddings-1}")
 # [B*M, 1, 1]
            try:
                gains = self.subject_gain(subject_ids) # [B, 1]
                gains_expanded = gains.repeat_interleave(num_channels, dim=0).unsqueeze(-2) # [B*M, 1, 1]
                embedded_patches = embedded_patches * gains_expanded
            except:
                self.cut_subject_gain()
                gains = self.subject_gain(subject_ids) # [B, 1]
                gains_expanded = gains.repeat_interleave(num_channels, dim=0).unsqueeze(-2) # [B*M, 1, 1]
                embedded_patches = embedded_patches * gains_expanded
        elif self.subject_gain is not None and subject_ids is None:
            print("Warning: LBLMInputProcessor subject_gain is enabled but subject_ids not provided in forward pass.")

        # 6. Masking for MSTP
        token_mask_prob = torch.rand(batch_size * num_channels, num_patches_seq, device=x_raw_eeg.device)*(1+np.random.rand() if self.training else 1) # TODO Remove if not needed
        # True where token is masked (to be predicted)
        masked_indices_bool = token_mask_prob < self.mask_ratio


        def smooth_mask(masked_indices_bool, p, i):
            if np.random.rand() < p:
                masked_indices_bool[i,1:] = masked_indices_bool[i,1:]|masked_indices_bool[i,:-1]
            if np.random.rand() < p:
                masked_indices_bool[i,:-1] = masked_indices_bool[i,:-1]|masked_indices_bool[i,1:]
            if np.random.rand() < p/2:
                masked_indices_bool[i,:-1] = masked_indices_bool[i,:-1]|masked_indices_bool[i,1:]
            return masked_indices_bool


        if not self.training:
            for i in range(masked_indices_bool.shape[0]):
                masked_indices_bool = smooth_mask(masked_indices_bool, 0.5, i)
        else:
            i = [*range(masked_indices_bool.shape[0])]
            masked_indices_bool = smooth_mask(masked_indices_bool, 0.5, i)
        return embedded_patches, original_patches_for_target, masked_indices_bool, rev_in_mean, rev_in_std
