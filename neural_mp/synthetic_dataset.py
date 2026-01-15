# File: synthetic_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import time

class SyntheticMPDataset(Dataset):
    """
    A synthetic dataset for testing Matching Pursuit models.

    This dataset generates signals by sparsely combining a set of predefined
    "ground truth" atoms and adding Gaussian noise. This version uses a
    vectorized implementation for high-performance signal generation.
    """
    def __init__(self,
                 num_samples: int = 10000,
                 num_channels: int = 62,
                 signal_length: int = 600,
                 num_true_atoms: int = 512,
                 sparsity: int = 5,
                 noise_level: float = 0.1):
        """
        Initializes the synthetic dataset.

        Args:
            num_samples (int): Total number of synthetic signals to generate.
            num_channels (int): Number of channels for each signal.
            signal_length (int): Number of time points for each signal.
            num_true_atoms (int): The size of the ground truth dictionary of atoms.
            sparsity (int): The *maximum* number of atoms used to construct each signal.
            noise_level (float): Standard deviation of Gaussian noise added to signals.
        """
        super().__init__()
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.signal_length = signal_length
        self.num_true_atoms = num_true_atoms
        self.sparsity = sparsity
        self.noise_level = noise_level

        print("Generating ground truth atoms for synthetic dataset...")
        self.true_atoms = self._generate_ground_truth_atoms()
        print(f"Generated {self.num_true_atoms} ground truth atoms.")
        
        print(f"Generating {num_samples} synthetic signals (vectorized)...")
        start_time = time.time()
        # Pre-generate all signals using the vectorized method
        self.signals = self._generate_all_signals_vectorized()
        end_time = time.time()
        print(f"Generated {num_samples} signals in {end_time - start_time:.2f} seconds.")

    def _generate_ground_truth_atoms(self) -> torch.Tensor:
        """Creates the dictionary of atoms used to generate all signals."""
        atoms = torch.zeros(self.num_true_atoms, self.num_channels, self.signal_length)
        t = torch.linspace(0, 4 * torch.pi, self.signal_length).unsqueeze(0).repeat(self.num_channels, 1)

        for i in range(self.num_true_atoms):
            freq = torch.rand(1) * 5.0 + 1.0
            phase = torch.rand(1) * 2 * torch.pi
            channel_amplitudes = torch.randn(self.num_channels, 1) * 0.5 + 1.0
            atom = torch.sin(freq * t + phase) * channel_amplitudes
            norm = torch.linalg.vector_norm(atom.flatten())
            if norm > 1e-6:
                atom /= norm
            atoms[i] = atom
        
        return atoms

    def _generate_all_signals_vectorized(self) -> torch.Tensor:
        """
        Pre-generates all synthetic signals using vectorized PyTorch operations.
        """
        # 1. Determine the number of active atoms for each sample (between 1 and sparsity)
        # Shape: [num_samples]
        num_active_atoms = torch.randint(1, self.sparsity + 1, (self.num_samples,))

        # 2. For all samples, select `sparsity` random atom indices. We'll use a mask
        # later to select a subset of these. Using topk on random values is an
        # efficient way to get unique random indices per sample.
        # Shape: [num_samples, sparsity]
        rand_indices = torch.rand(self.num_samples, self.num_true_atoms).topk(self.sparsity, dim=1).indices

        # 3. For all samples, generate `sparsity` random coefficients.
        # Shape: [num_samples, sparsity]
        rand_coeffs = torch.randn(self.num_samples, self.sparsity) * 1.5 + torch.sign(torch.randn(self.num_samples, self.sparsity))

        # 4. Create a mask to enforce variable sparsity. We'll zero out the
        # coefficients for atoms that are not meant to be active in each sample.
        # Shape: [num_samples, sparsity]
        mask = torch.arange(self.sparsity).expand(self.num_samples, self.sparsity) < num_active_atoms.unsqueeze(1)
        
        # Apply the mask to the coefficients
        final_coeffs = rand_coeffs * mask

        # 5. Gather the selected atoms for all samples using advanced indexing.
        # `true_atoms` shape: [num_true_atoms, num_channels, signal_length]
        # `rand_indices` shape: [num_samples, sparsity]
        # `gathered_atoms` shape: [num_samples, sparsity, num_channels, signal_length]
        gathered_atoms = self.true_atoms[rand_indices]

        # 6. Scale the gathered atoms by their corresponding coefficients.
        # We need to reshape coefficients for broadcasting: [S, P] -> [S, P, 1, 1]
        # so it aligns with the shape of `gathered_atoms`.
        scaled_atoms = gathered_atoms * final_coeffs.unsqueeze(-1).unsqueeze(-1)

        # 7. Sum over the `sparsity` dimension to create the final clean signals.
        # This collapses the [S, P, C, L] tensor into a [S, C, L] tensor.
        signals = torch.sum(scaled_atoms, dim=1)

        # 8. Add noise to the entire batch of signals at once.
        noise = torch.randn_like(signals) * self.noise_level
        signals += noise
        
        return signals.to(torch.float32)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a pre-generated synthetic signal from the stored tensor.
        """
        return {'x_raw_eeg': self.signals[idx]}