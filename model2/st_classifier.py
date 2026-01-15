# File: st_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MultiScaleInceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels_per_branch, use_depthwise_separable=True):
        super().__init__()
        self.use_depthwise_separable = use_depthwise_separable
        ConvModule = DepthwiseSeparableConv1d if use_depthwise_separable else nn.Conv1d

        # Branches for 1x1, 1x3, 1x5, 1x7
        # Padding: (kernel_size - 1) // 2 for 'same' padding
        self.branch1x1 = nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=1)
        
        self.branch1x3 = ConvModule(in_channels, out_channels_per_branch, kernel_size=3, padding=1)
        
        self.branch1x5 = ConvModule(in_channels, out_channels_per_branch, kernel_size=5, padding=2)
        
        self.branch1x7 = ConvModule(in_channels, out_channels_per_branch, kernel_size=7, padding=3)

        # The "Pool" in the diagram for Inception is ambiguous.
        # Standard Inception has a MaxPool -> 1x1 Conv projection branch.
        # LBLM diagram shows "Pool" after the 1xK convs, which is unusual.
        # Let's implement a standard MaxPool+Projection branch.
        self.branch_pool_conv = nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=1)
        self.branch_pool_max = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)


        # The diagram's zoom-in has "Depth-wise Conv." at the top, then parallel convs and a pool.
        # If that "Depth-wise Conv." is an initial stem:
        # self.stem_dw_conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        # self.stem_pw_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False) # Or some F_intermediate
        # Then branches operate on output of stem.

        # For now, interpreting "Depth-wise Conv." in the diagram as meaning the branches themselves
        # (1x3, 1x5, 1x7) are depthwise-separable, which is handled by ConvModule.
        # And the 1x1 is inherently pointwise.
        # The "Pool" branch is interpreted as a parallel MaxPool -> 1x1 Conv like standard Inception.

    def forward(self, x):
        # x: [B, C_in, T_in]
        
        # Optional stem based on diagram interpretation (if "Depth-wise Conv." is initial)
        # x_stem = self.stem_pw_conv(self.stem_dw_conv(x)) 
        # x_in = F.relu(x_stem) # Or just x

        x_in = x # Assuming no separate stem for now, branches operate on input 'x'

        out1x1 = self.branch1x1(x_in)
        out1x3 = self.branch1x3(x_in)
        out1x5 = self.branch1x5(x_in)
        out1x7 = self.branch1x7(x_in)
        
        out_pool = self.branch_pool_max(x_in)
        out_pool = self.branch_pool_conv(out_pool)
        
        # Filter concatenation
        outputs = [out1x1, out1x3, out1x5, out1x7, out_pool]
        return torch.cat(outputs, dim=1) # Concatenate along channel dimension

class STClassifier(nn.Module):
    def __init__(self, num_classes, 
                 d_model_from_backbone, # Feature dimension from LBLM backbone output per patch
                 num_eeg_channels,      # Number of EEG channels
                 # Spatial Conv params
                 spatial_conv_out_features=64, 
                 # Multi-scale Inception params
                 inception_out_channels_per_branch=32, 
                 inception_use_depthwise_separable=True,
                 # Temporal Conv params
                 temporal_conv_out_features=128,
                 temporal_conv_kernel_size=5,
                 # Feed Forward params
                 fc_hidden_dim=256,
                 dropout_rate=0.5):
        super().__init__()
        self.num_eeg_channels = num_eeg_channels
        self.d_model_from_backbone = d_model_from_backbone

        # 1. Spatial Convolution
        # Input after reshape: [B, d_model, num_eeg_channels, num_patches]
        # We want to aggregate across num_eeg_channels.
        # This layer effectively learns spatial filters.
        # Output: [B, spatial_conv_out_features, 1, num_patches] -> squeeze -> [B, spatial_conv_out_features, num_patches]
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=d_model_from_backbone, 
                out_channels=spatial_conv_out_features, 
                kernel_size=(self.num_eeg_channels, 1), # Kernel spans all EEG channels
                padding='valid', # No padding for channel dimension
                bias=False
            ),
            nn.BatchNorm2d(spatial_conv_out_features),
            nn.ELU(), # Or ReLU
            # nn.Dropout2d(dropout_rate) # Optional spatial dropout
        )
        # The output of spatial_conv will be [B, spatial_conv_out_features, 1, num_patches]
        # We will squeeze the dimension of size 1.

        # 2. Multi-scale Inception
        # Input: [B, spatial_conv_out_features, num_patches]
        self.multi_scale_inception = MultiScaleInceptionModule(
            in_channels=spatial_conv_out_features,
            out_channels_per_branch=inception_out_channels_per_branch,
            use_depthwise_separable=inception_use_depthwise_separable
        )
        # Output channels: 5 * inception_out_channels_per_branch (if 5 branches)
        inception_total_out_channels = 5 * inception_out_channels_per_branch
        self.bn_after_inception = nn.BatchNorm1d(inception_total_out_channels)
        self.act_after_inception = nn.ELU() # Or ReLU
        
        # 3. Pooling after Inception
        # The diagram shows pooling. Let's use AdaptiveAvgPool1d to reduce sequence length.
        # Or a fixed MaxPool1d. Let's use MaxPool1d with stride to downsample.
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) # Example downsampling
        # self.pool1 = nn.AdaptiveAvgPool1d(output_size= new_seq_len_target) # Alternative

        # 4. Temporal Convolution
        # Input: [B, inception_total_out_channels, S_after_pool1]
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=inception_total_out_channels,
                out_channels=temporal_conv_out_features,
                kernel_size=temporal_conv_kernel_size,
                padding=(temporal_conv_kernel_size - 1) // 2, # 'same' padding
                bias=False
            ),
            nn.BatchNorm1d(temporal_conv_out_features),
            nn.ELU(), # Or ReLU
            nn.Dropout(dropout_rate)
        )

        # 5. Pooling after Temporal Conv
        # Use AdaptiveAvgPool1d to get a fixed-size output for the FFN, regardless of input sequence length.
        self.pool2 = nn.AdaptiveAvgPool1d(output_size=1)
        # Output: [B, temporal_conv_out_features, 1] -> squeeze -> [B, temporal_conv_out_features]

        # 6. Feed Forward Layer
        self.feed_forward = nn.Sequential(
            nn.Linear(temporal_conv_out_features, fc_hidden_dim),
            nn.BatchNorm1d(fc_hidden_dim), # Optional, but often good
            nn.ELU(), # Or ReLU
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim, num_classes)
        )
        # 7. Softmax is typically applied outside during loss calculation (e.g., with nn.CrossEntropyLoss)
        # If you need raw logits, don't add Softmax here. If you need probabilities, add nn.Softmax(dim=-1).

    def forward(self, x_backbone_features):
        # x_backbone_features (output from LBLM backbone): 
        # [batch_size * num_eeg_channels, num_patches, d_model]
        
        # Reshape and Permute for Spatial Conv
        # Current shape: [B*M, S, D]
        # Target for Conv2D: [B, D, M, S] (Batch, InChannels, Height=num_eeg_channels, Width=num_patches)
        B_times_M, S, D = x_backbone_features.shape
        B = B_times_M // self.num_eeg_channels
        
        if B_times_M % self.num_eeg_channels != 0:
            raise ValueError(f"First dimension of backbone features ({B_times_M}) "
                             f"is not divisible by num_eeg_channels ({self.num_eeg_channels}). "
                             f"Ensure batch processing aligns with num_eeg_channels.")

        x = x_backbone_features.reshape(B, self.num_eeg_channels, S, D) # [B, M, S, D]
        x = x.permute(0, 3, 1, 2) # [B, D, M, S] - D is in_channels for Conv2D

        # 1. Spatial Conv
        x = self.spatial_conv(x) # Output: [B, F_spatial, 1, S]
        x = x.squeeze(2) # Output: [B, F_spatial, S] (temporal sequences of spatially aggregated features)

        # 2. Multi-scale Inception
        x = self.multi_scale_inception(x)
        x = self.bn_after_inception(x)
        x = self.act_after_inception(x)
        
        # 3. Pooling1
        x = self.pool1(x)
        
        # 4. Temporal Conv
        x = self.temporal_conv(x)
        
        # 5. Pooling2 (Global Average Pooling over time)
        x = self.pool2(x) # Output: [B, F_temporal, 1]
        x = x.squeeze(2) # Output: [B, F_temporal] (flattened features)
        
        # 6. Feed Forward
        logits = self.feed_forward(x) # Output: [B, num_classes]
        
        return logits

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MultiScaleInceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels_per_branch, use_depthwise_separable=True):
        super().__init__()
        self.use_depthwise_separable = use_depthwise_separable
        ConvModule = DepthwiseSeparableConv1d if use_depthwise_separable else nn.Conv1d

        # Branches for 1x1, 1x3, 1x5, 1x7
        # Padding: (kernel_size - 1) // 2 for 'same' padding
        self.branch1x1 = nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=1)
        
        self.branch1x3 = ConvModule(in_channels, out_channels_per_branch, kernel_size=3, padding=1)
        
        self.branch1x5 = ConvModule(in_channels, out_channels_per_branch, kernel_size=5, padding=2)
        
        self.branch1x7 = ConvModule(in_channels, out_channels_per_branch, kernel_size=7, padding=3)

        # The "Pool" in the diagram for Inception is ambiguous.
        # Standard Inception has a MaxPool -> 1x1 Conv projection branch.
        # LBLM diagram shows "Pool" after the 1xK convs, which is unusual.
        # Let's implement a standard MaxPool+Projection branch.
        self.branch_pool_conv = nn.Conv1d(in_channels, out_channels_per_branch, kernel_size=1)
        self.branch_pool_max = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)


        # The diagram's zoom-in has "Depth-wise Conv." at the top, then parallel convs and a pool.
        # If that "Depth-wise Conv." is an initial stem:
        # self.stem_dw_conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        # self.stem_pw_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False) # Or some F_intermediate
        # Then branches operate on output of stem.

        # For now, interpreting "Depth-wise Conv." in the diagram as meaning the branches themselves
        # (1x3, 1x5, 1x7) are depthwise-separable, which is handled by ConvModule.
        # And the 1x1 is inherently pointwise.
        # The "Pool" branch is interpreted as a parallel MaxPool -> 1x1 Conv like standard Inception.

    def forward(self, x):
        # x: [B, C_in, T_in]
        
        # Optional stem based on diagram interpretation (if "Depth-wise Conv." is initial)
        # x_stem = self.stem_pw_conv(self.stem_dw_conv(x)) 
        # x_in = F.relu(x_stem) # Or just x

        x_in = x # Assuming no separate stem for now, branches operate on input 'x'

        out1x1 = self.branch1x1(x_in)
        out1x3 = self.branch1x3(x_in)
        out1x5 = self.branch1x5(x_in)
        out1x7 = self.branch1x7(x_in)
        
        out_pool = self.branch_pool_max(x_in)
        out_pool = self.branch_pool_conv(out_pool)
        
        # Filter concatenation
        outputs = [out1x1, out1x3, out1x5, out1x7, out_pool]
        return torch.cat(outputs, dim=1) # Concatenate along channel dimension

class STClassifier(nn.Module):
    def __init__(self, num_classes, 
                 d_model_from_backbone, # Feature dimension from LBLM backbone output per patch
                 num_eeg_channels,      # Number of EEG channels
                 # Spatial Conv params
                 spatial_conv_out_features=64, 
                 # Multi-scale Inception params
                 inception_out_channels_per_branch=62, 
                 inception_use_depthwise_separable=True,
                 # Temporal Conv params
                 temporal_conv_out_features=128,
                 temporal_conv_kernel_size=5,
                 # Feed Forward params
                 fc_hidden_dim=256,
                 dropout_rate=0.2):
        super().__init__()
        self.num_eeg_channels = num_eeg_channels
        self.d_model_from_backbone = d_model_from_backbone

        # 1. Spatial Convolution
        # Input after reshape: [B, d_model, num_eeg_channels, num_patches]
        # We want to aggregate across num_eeg_channels.
        # This layer effectively learns spatial filters.
        # Output: [B, spatial_conv_out_features, 1, num_patches] -> squeeze -> [B, spatial_conv_out_features, num_patches]
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=d_model_from_backbone, 
                out_channels=spatial_conv_out_features, 
                kernel_size=(self.num_eeg_channels, 1), # Kernel spans all EEG channels
                padding='valid', # No padding for channel dimension
                bias=False
            ),
            nn.BatchNorm2d(spatial_conv_out_features),
            nn.ELU(), # Or ReLU
            # nn.Dropout2d(dropout_rate) # Optional spatial dropout
        )
        # The output of spatial_conv will be [B, spatial_conv_out_features, 1, num_patches]
        # We will squeeze the dimension of size 1.

        # 2. Multi-scale Inception
        # Input: [B, spatial_conv_out_features, num_patches]
        self.multi_scale_inception = MultiScaleInceptionModule(
            in_channels=spatial_conv_out_features,
            out_channels_per_branch=inception_out_channels_per_branch,
            use_depthwise_separable=inception_use_depthwise_separable
        )
        # Output channels: 5 * inception_out_channels_per_branch (if 5 branches)
        inception_total_out_channels = 5 * inception_out_channels_per_branch
        self.bn_after_inception = nn.BatchNorm1d(inception_total_out_channels)
        self.act_after_inception = nn.ELU() # Or ReLU
        
        # 3. Pooling after Inception
        # The diagram shows pooling. Let's use AdaptiveAvgPool1d to reduce sequence length.
        # Or a fixed MaxPool1d. Let's use MaxPool1d with stride to downsample.
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) # Example downsampling
        # self.pool1 = nn.AdaptiveAvgPool1d(output_size= new_seq_len_target) # Alternative

        # 4. Temporal Convolution
        # Input: [B, inception_total_out_channels, S_after_pool1]
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=inception_total_out_channels,
                out_channels=temporal_conv_out_features,
                kernel_size=temporal_conv_kernel_size,
                padding=(temporal_conv_kernel_size - 1) // 2, # 'same' padding
                bias=False
            ),
            nn.BatchNorm1d(temporal_conv_out_features),
            nn.ELU(), # Or ReLU
            nn.Dropout(dropout_rate)
        )

        # 5. Pooling after Temporal Conv
        # Use AdaptiveAvgPool1d to get a fixed-size output for the FFN, regardless of input sequence length.
        self.pool2 = nn.AdaptiveAvgPool1d(output_size=1)
        # Output: [B, temporal_conv_out_features, 1] -> squeeze -> [B, temporal_conv_out_features]

        # 6. Feed Forward Layer
        self.feed_forward = nn.Sequential(
            nn.Linear(temporal_conv_out_features, fc_hidden_dim),
            nn.BatchNorm1d(fc_hidden_dim), # Optional, but often good
            nn.ELU(), # Or ReLU
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim, num_classes)
        )
        # 7. Softmax is typically applied outside during loss calculation (e.g., with nn.CrossEntropyLoss)
        # If you need raw logits, don't add Softmax here. If you need probabilities, add nn.Softmax(dim=-1).

    def forward(self, x_backbone_features):
        # x_backbone_features (output from LBLM backbone): 
        # [batch_size * num_eeg_channels, num_patches, d_model]
        
        # Reshape and Permute for Spatial Conv
        # Current shape: [B*M, S, D]
        # Target for Conv2D: [B, D, M, S] (Batch, InChannels, Height=num_eeg_channels, Width=num_patches)
        B_times_M, S, D = x_backbone_features.shape
        B = B_times_M // self.num_eeg_channels
        
        if B_times_M % self.num_eeg_channels != 0:
            raise ValueError(f"First dimension of backbone features ({B_times_M}) "
                             f"is not divisible by num_eeg_channels ({self.num_eeg_channels}). "
                             f"Ensure batch processing aligns with num_eeg_channels.")

        x = x_backbone_features.reshape(B, self.num_eeg_channels, S, D) # [B, M, S, D]
        x = x.permute(0, 3, 1, 2) # [B, D, M, S] - D is in_channels for Conv2D

        # 1. Spatial Conv
        x = self.spatial_conv(x) # Output: [B, F_spatial, 1, S]
        x = x.squeeze(2) # Output: [B, F_spatial, S] (temporal sequences of spatially aggregated features)

        # 2. Multi-scale Inception
        x = self.multi_scale_inception(x)
        x = self.bn_after_inception(x)
        x = self.act_after_inception(x)
        
        # 3. Pooling1
        x = self.pool1(x)
        
        # 4. Temporal Conv
        x = self.temporal_conv(x)
        
        # 5. Pooling2 (Global Average Pooling over time)
        x = self.pool2(x) # Output: [B, F_temporal, 1]
        x = x.squeeze(2) # Output: [B, F_temporal] (flattened features)
        
        # 6. Feed Forward
        logits = self.feed_forward(x) # Output: [B, num_classes]
        
        return logits
# Example Usage (for testing the classifier standalone)
if __name__ == '__main__':
    # Parameters matching LBLM backbone output and STClassifier needs
    batch_size = 4
    num_eeg_channels_val = 60 # Example
    d_model_val = 64          # Output feature dim from LBLM patch
    num_patches_val = 80      # Example sequence length of patches

    # Dummy LBLM backbone output
    dummy_backbone_output = torch.randn(batch_size * num_eeg_channels_val, num_patches_val, d_model_val)

    num_classes_val = 24 # LBLM paper word-level classification

    st_classifier = STClassifier(
        num_classes=num_classes_val,
        d_model_from_backbone=d_model_val,
        num_eeg_channels=num_eeg_channels_val,
        spatial_conv_out_features=32,
        inception_out_channels_per_branch=16, # Total 5*16 = 80 channels from inception
        temporal_conv_out_features=64,
        fc_hidden_dim=128,
        dropout_rate=0.3
    )

    print(f"STClassifier initialized with {sum(p.numel() for p in st_classifier.parameters()):,} parameters.")

    # Test forward pass
    st_classifier.train() # To ensure dropout is active if used
    output_logits = st_classifier(dummy_backbone_output)
    print("Output logits shape:", output_logits.shape) # Expected: [batch_size, num_classes_val]
    assert output_logits.shape == (batch_size, num_classes_val)

    st_classifier.eval()
    output_logits_eval = st_classifier(dummy_backbone_output)
    print("Output logits shape (eval):", output_logits_eval.shape)
    assert output_logits_eval.shape == (batch_size, num_classes_val)

    print("STClassifier standalone test completed.")