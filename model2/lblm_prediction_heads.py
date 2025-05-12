import torch
import torch.nn as nn

class SpectroTemporalPredictionHeads(nn.Module):
    def __init__(self, d_model, patch_length):
        super().__init__()
        self.d_model = d_model
        self.patch_length = patch_length
        # For rFFT, output size is N//2 + 1 complex numbers.
        # Amplitude and Phase will each have this many components.
        self.fft_output_dim = self.patch_length // 2 + 1

        self.head_wave = nn.Linear(d_model, patch_length)
        self.head_amplitude = nn.Linear(d_model, self.fft_output_dim)
        self.head_phase = nn.Linear(d_model, self.fft_output_dim)

    def forward(self, conformer_output_tokens):
        # conformer_output_tokens: [TotalNumberOfMaskedPatches, d_model]
        pred_wave = self.head_wave(conformer_output_tokens)         # [TotalMasked, patch_length]
        pred_amplitude = self.head_amplitude(conformer_output_tokens) # [TotalMasked, fft_output_dim]
        pred_phase = self.head_phase(conformer_output_tokens)       # [TotalMasked, fft_output_dim]
        return pred_wave, pred_amplitude, pred_phase
