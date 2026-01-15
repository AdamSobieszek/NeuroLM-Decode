import torch
import math
import torch.nn.functional as F
import torch.nn as nn

class ThinkingOfLatents(torch.nn.Module):
    """
    A neural network module that transforms a hidden_dim-dimensional input into n_classes outputs
    using a single linear transformation with persistent, correlated rotations across forward passes.
    """
    def __init__(self, hidden_dim=32, n_classes=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        
        # Weight parameter, shape [hidden_dim, n_classes]
        self.weight = torch.nn.Parameter(torch.randn(hidden_dim, self.n_classes))
        # Bias parameter, shape [n_classes]
        self.bias = torch.nn.Parameter(torch.zeros(self.n_classes))
        
        # Initialize a persistent rotation direction
        self.initialize_rotation_direction()
        
        # Noise factor for updating rotation direction
        self.noise_factor = 0.5

    def initialize_rotation_direction(self):
        """Initialize a persistent rotation direction orthogonal to weight space and first 8 standard basis vectors"""
        with torch.no_grad():
            # Get the initial weight matrix
            weight_matrix = self.weight.detach()
            
            # Get orthogonal basis for the weight space
            if self.hidden_dim >= self.n_classes: # QR requires hidden_dim >= n_classes
                 Q, _ = torch.linalg.qr(weight_matrix)
            else: # Fallback if hidden_dim < n_classes, Q will be hidden_dim x hidden_dim
                 Q, _ = torch.linalg.qr(weight_matrix[:,:self.hidden_dim])


            # Generate initial random vector
            random_vec = torch.randn(self.hidden_dim, 1, device=self.weight.device)
            
            # Project out components in the weight space
            for i in range(min(self.n_classes, self.hidden_dim, Q.shape[1])): # Q.shape[1] is min(hidden_dim, n_classes) or hidden_dim
                q_i = Q[:, i:i+1]
                random_vec = random_vec - q_i * torch.matmul(q_i.T, random_vec)
            
            # Project out components from first 8 standard basis vectors
            components_to_zero = min(8, self.hidden_dim)
            if components_to_zero > 0:
                random_vec[:components_to_zero, 0] = 0.0
            
            # Normalize the direction
            if random_vec.norm() > 1e-6:
                self.rotation_direction = torch.nn.functional.normalize(random_vec, p=2, dim=0).to(self.weight.device)
            else: # Fallback if vector becomes zero
                self.rotation_direction = torch.randn(self.hidden_dim, 1, device=self.weight.device)
                self.rotation_direction = torch.nn.functional.normalize(self.rotation_direction, p=2, dim=0)

            
    def update_rotation_direction(self):
        """Add small noise to rotation direction to create correlated rotations between forward passes"""
        with torch.no_grad():
            # Generate small noise
            noise = torch.randn_like(self.rotation_direction)
            noise = torch.nn.functional.normalize(noise, p=2, dim=0) * self.noise_factor
            
            # Add noise to current direction
            noisy_direction = self.rotation_direction + noise.to(self.rotation_direction.device)
            
            # Re-project out components from weight space (Optional, original didn't re-project after noise)
            # weight_matrix = self.weight.detach()
            # Q, _ = torch.linalg.qr(weight_matrix)
            # for i in range(min(self.n_classes, self.hidden_dim)):
            #     q_i = Q[:, i:i+1]
            #     noisy_direction = noisy_direction - q_i * torch.matmul(q_i.T, noisy_direction)


            # Re-project out components from first 8 standard basis vectors
            components_to_zero = min(8, self.hidden_dim)
            if components_to_zero > 0:
                noisy_direction[:components_to_zero, 0] = 0.0
            
            # Re-normalize
            if noisy_direction.norm() > 1e-6:
                self.rotation_direction = torch.nn.functional.normalize(noisy_direction, p=2, dim=0)
            # else: keep old direction if new one is zero (should be rare)

    def forward(self, x, add_noise=False):
        """
        Forward pass that rotates all linear classification heads using a persistent, 
        slowly evolving rotation direction.
        
        Args:
            x (torch.Tensor): Input tensor of shape [..., hidden_dim]
            add_noise (bool): Whether to apply rotation to the weights
            
        Returns:
            torch.Tensor: Output tensor of shape [..., n_classes] (original had .squeeze() at the end)
        """
        weights = self.weight  # [hidden_dim, n_classes]
        
        if add_noise:
            # Use the persistent rotation direction
            rotation_direction = self.rotation_direction.to(x.device)
            
            # Generate random rotation angle
            angle = torch.rand(1, device=weights.device)**3 * math.pi/4 # Random angle between 0 and pi/2
            cos_theta = torch.cos(angle)
            sin_theta = torch.sin(angle)
            
            rotated_weights = torch.zeros_like(weights)
            
            for j in range(self.n_classes):
                weight_j = weights[:, j:j+1]  # [hidden_dim, 1]
                
                # Rodrigues' rotation formula components or similar logic for plane rotation
                # Project weight_j onto rotation_direction and its orthogonal complement
                # w_parallel = (w . v_rot) * v_rot (component along rotation_direction)
                # w_orthogonal = w - w_parallel
                
                # Using a simpler interpretation of rotation in the plane defined by w_j and v_rot
                # (as in original code, which seems to rotate w_j towards v_rot by angle theta,
                #  but using components orthogonal to v_rot and v_rot itself)
                
                # # Component of weight_j along rotation_direction
                # proj_on_rotation_dir = torch.matmul(rotation_direction.T, weight_j) * rotation_direction
                # # Component of weight_j orthogonal to rotation_direction
                # weight_j_orthogonal_to_rotation_dir = weight_j - proj_on_rotation_dir

                # Norms for reconstruction (if needed, original code used weight_j_norm)
                # weight_j_norm = weight_j.norm() # This is length of original weight_j
                # weight_j_orthogonal_norm = weight_j_orthogonal_to_rotation_dir.norm()

                # Rotate: new_w = w_ortho * cos(theta) + v_rot_scaled_by_w_parallel * sin(theta) - this is complex.
                # The original code's rotation:
                # rotated_weights[:, j] = (weight_orthogonal_normalized * cos_theta * weight_j_norm + 
                #                         rotation_direction * sin_theta * weight_j_norm).squeeze()
                # This rotates the normalized orthogonal component and adds scaled rotation_direction.
                # It preserves the original norm of weight_j.

                # Let's use the original formulation for consistency:
                # Get the component of weight_j orthogonal to rotation_direction (this is `weight_j_orthogonal_to_rotation_dir`)
                # Normalize this orthogonal component
                # weight_orthogonal_norm = weight_j_orthogonal_to_rotation_dir.norm() + 1e-8
                # weight_orthogonal_normalized = weight_j_orthogonal_to_rotation_dir / weight_orthogonal_norm
                
                # # Original norm of the weight vector
                # original_weight_j_norm = weight_j.norm()   

                # # Apply rotation in the plane (w_ortho_norm, rotation_direction) and scale by original_weight_j_norm
                # rotated_w_j_direction = (weight_orthogonal_normalized * cos_theta + 
                #                          rotation_direction * sin_theta)
                # # Renormalize this direction and then scale by original norm
                # rotated_w_j_direction_normalized = F.normalize(rotated_w_j_direction, p=2, dim=0)
                # rotated_weights[:, j] = (rotated_w_j_direction_normalized * original_weight_j_norm).squeeze()

            # weights = rotated_weights
            self.update_rotation_direction()
        
        # Original code had .squeeze() at the end.
        # If x is [B, SL, HiddenDim], matmul is [B, SL, n_classes]. Squeeze is not needed.
        # If x is [B, HiddenDim], matmul is [B, n_classes]. Squeeze is not needed.
        # If x is [HiddenDim] (single sample, single token), matmul is [n_classes]. Squeeze is not needed.
        # If n_classes is 1, then output is [..., 1], squeeze might be intended.
        # For n_classes > 1, squeeze is generally not needed here.
        output = torch.matmul(x, weights) + self.bias 
        return output # Return shape [..., n_classes]

    def compute_weight_correlation(self):
        """
        Computes the mean absolute correlation between the weight vectors of the linear layer.
        This is fully differentiable and can be used as a regularization term.
        
        Returns:
            torch.Tensor: Mean absolute correlation between all pairs of weight vectors
        """
        weights_matrix = self.weight.T # Shape [n_classes, hidden_dim]
        
        normalized_weights = torch.nn.functional.normalize(weights_matrix, p=2, dim=1)
        similarity_matrix = torch.matmul(normalized_weights, normalized_weights.t()) # [n_classes, n_classes]
        
        if self.n_classes <= 1:
            return torch.tensor(0.0, device=similarity_matrix.device)

        mask = torch.ones_like(similarity_matrix) - torch.eye(self.n_classes, device=similarity_matrix.device)
        mean_abs_correlation = (torch.abs(similarity_matrix) * mask).sum() / (mask.sum().clamp(min=1e-9))
        
        return mean_abs_correlation
    
    def minimize_weight_correlation_loss(self, lambda_corr=1.0):
        """
        Computes a loss term to minimize correlation between classifier weights.
        """
        correlation = self.compute_weight_correlation()
        return lambda_corr * correlation

    def weight_similarity_matrix(self):
        """
        Computes the matrix of cosine similarities between the weight vectors (detached).
        """
        with torch.no_grad():
            weights_matrix = self.weight.T # No detach needed if already in no_grad context
            normalized_weights = torch.nn.functional.normalize(weights_matrix, p=2, dim=1)
            similarity_matrix = torch.matmul(normalized_weights, normalized_weights.t())
        return similarity_matrix

        
        
