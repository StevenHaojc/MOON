import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class GCCALoss(nn.Module):
    def __init__(self, outdim_size):
        super(GCCALoss, self).__init__()
        self.outdim_size = outdim_size

    def forward(self, H1, H2, H3):
        """
        H1, H2, H3 are of the same shape [batch_size, features_dim]
        """
        r = 1e-4
        eps = 1e-12

        H1, H2, H3 = [H - H.mean(dim=0) for H in [H1, H2, H3]]
        H1, H2, H3 = [H / (H.std(dim=0) + eps) for H in [H1, H2, H3]]

        C1 = torch.mm(H1.T, H1) + r * torch.eye(H1.size(1), device=H1.device)
        C2 = torch.mm(H2.T, H2) + r * torch.eye(H2.size(1), device=H2.device)
        C3 = torch.mm(H3.T, H3) + r * torch.eye(H3.size(1), device=H3.device)

        e1, v1 = torch.linalg.eigh(C1, UPLO='U')
        e2, v2 = torch.linalg.eigh(C2, UPLO='U')
        e3, v3 = torch.linalg.eigh(C3, UPLO='U')

        # Use the smallest eigenvectors to form the common space
        d = min([e1.size(0), e2.size(0), e3.size(0), self.outdim_size])

        H1, H2, H3 = [torch.mm(H, v[:, -d:]) for H, v in zip([H1, H2, H3], [v1, v2, v3])]

        # Generalized cca loss is mean of correlation of all pairs
        gcca_loss = (torch.norm(torch.mm(H1.T, H2)) + 
                     torch.norm(torch.mm(H2.T, H3)) + 
                     torch.norm(torch.mm(H1.T, H3))) / (3 * H1.size(1))
        
        return -gcca_loss  # Negating the loss to maximize correlation

# Example usage
batch_size = 32
feature_dim = 256
outdim_size = 128

# Dummy data representing features from three modalities
H1 = torch.randn(batch_size, feature_dim)
H2 = torch.randn(batch_size, feature_dim)
H3 = torch.randn(batch_size, feature_dim)

# Initialize GCCA Loss
gcca_loss = GCCALoss(outdim_size=outdim_size)

# Calculate loss
loss = gcca_loss(H1, H2, H3)
print(loss)
# Backpropagate and update model weights
# loss.backward()
