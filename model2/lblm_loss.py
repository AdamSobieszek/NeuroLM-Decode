import torch
import torch.nn as nn
import torch.nn.functional as F

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
        if delta <= 0:
            raise ValueError("HuberLoss delta must be positive.")

    def forward(self, predictions, targets):
        # Ensure predictions and targets have the same shape
        if predictions.shape != targets.shape:
            raise ValueError(f"Predictions shape {predictions.shape} must match targets shape {targets.shape}")
            
        error = targets - predictions
        abs_error = torch.abs(error)
        
        quadratic_part_condition = (abs_error <= self.delta)
        linear_part_condition = (abs_error > self.delta) # Same as ~quadratic_part_condition
        
        # Initialize loss tensor
        loss_values = torch.zeros_like(predictions, dtype=predictions.dtype)
        
        # Quadratic part for elements where |error| <= delta
        loss_values[quadratic_part_condition] = 0.5 * error[quadratic_part_condition].pow(2)
        
        # Linear part for elements where |error| > delta
        loss_values[linear_part_condition] = self.delta * (abs_error[linear_part_condition] - 0.5 * self.delta)
        
        return loss_values.mean()
