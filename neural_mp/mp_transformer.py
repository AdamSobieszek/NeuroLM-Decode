# File: model_mp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assume MPAttentionLayer is defined in this file or imported
class MPAttentionLayer(nn.Module):
    """
    Neural Matching Pursuit Attention Layer.
    Approximates one step of MP in a differentiable, parallelized manner.
    """
    def __init__(self, d_model: int, num_atoms: int, temperature: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.num_atoms = num_atoms
        self.dictionary = nn.Parameter(torch.empty(num_atoms, d_model))
        nn.init.xavier_uniform_(self.dictionary)
        self.register_buffer('temperature', torch.tensor(temperature))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: [B, D_model]
        # dictionary shape: [N_atoms, D_model]
        # mask shape: [B, N_atoms] - 0 for masked atoms, 1 for available atoms
        scores = torch.matmul(x, self.dictionary.t()) # -> [B, N_atoms]
        
        # Apply mask to scores if provided
        if mask is not None:
            scores = scores * mask + (1 - mask) * (-1e9)  # Set masked atoms to very negative values
        
        weights = F.softmax(scores / self.temperature, dim=-1) # -> [B, N_atoms]
        reconstruction = torch.matmul(weights, self.dictionary)#/self.dictionary.norm(dim=-1, keepdim=True).clamp(min=1e-9)) # -> [B, D_model]
        
        # Return both reconstruction and the argmax indices for masking
        argmax_indices = torch.argmax(scores, dim=-1)  # -> [B]
        
        return reconstruction, argmax_indices

class MP_Model(nn.Module):
    """
    A model that uses recurrent invocation of MPAttentionLayer to perform
    Matching Pursuit on an input signal.
    """
    def __init__(self, num_channels: int, signal_length: int, num_atoms: int, mp_iterations: int, attention_temp: float = 0.001):
        super().__init__()
        self.num_channels = num_channels
        self.signal_length = signal_length
        self.mp_iterations = mp_iterations
        
        # The MP layer operates on flattened signals
        self.d_model = num_channels * signal_length
        
        self.mp_layer = MPAttentionLayer(
            d_model=self.d_model,
            num_atoms=num_atoms,
            temperature=attention_temp
        )

    def forward(self, x_eeg: torch.Tensor):
        """
        Args:
            x_eeg (torch.Tensor): The input EEG signal.
                                  Shape: [batch_size, num_channels, signal_length]

        Returns:
            Tuple[torch.Tensor, torch.Tensor, dict]:
            - loss: The norm of the final residual.
            - final_residual: The signal remaining after all iterations.
            - log: A dictionary for logging.
        """
        # Flatten the input signal for the MP layer
        # [B, C, T] -> [B, C*T]
        B = x_eeg.shape[0]
        residual = x_eeg.view(B, -1)
        original_signal_flat = residual.clone()

        # Initialize mask - all atoms are available initially
        # Shape: [B, num_atoms]
        atom_mask = torch.ones(B, self.mp_layer.num_atoms, device=x_eeg.device, dtype=torch.float32)
        

        for _ in range(self.mp_iterations):
            # Find the next component to subtract based on the current residual
            component_to_remove, argmax_indices = self.mp_layer(residual, atom_mask)
            
            # Update the residual
            residual = residual - component_to_remove
            
            # Update mask to prevent reusing the selected atoms
            # Set the selected atoms to 0 in the mask for each batch element
            batch_indices = torch.arange(B, device=x_eeg.device)
            # atom_mask = atom_mask.clone()
            # atom_mask[batch_indices, argmax_indices] = 0

        # The objective is to make the final residual as close to zero as possible.
        # We use Mean Squared Error, which is proportional to the L2 norm.
        loss = F.mse_loss(residual, torch.zeros_like(residual))
        
        

        log = {"mp/final_residual_mse": loss}
        
        # Return the final residual for plotting/analysis
        final_residual_unflattened = residual.view(B, self.num_channels, self.signal_length)
        
        return loss, final_residual_unflattened, log

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, optimizer_type='adamw', **kwargs):
        # A simple AdamW optimizer is sufficient here
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
        return optimizer