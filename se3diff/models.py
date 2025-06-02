import torch
import torch.nn as nn
from torch._prims_common import DeviceLikeType

from bioemu.models import SinusoidalPositionEmbedder
from bioemu.so3_sde import DiGSO3SDE, rotmat_to_rotvec


class ScoreNet(nn.Module):
    """Neural network to predict the score (3-vector) from a rotation and time."""

    def __init__(
        self, rot_embed_dim: int = 32, time_embed_dim: int = 32, hidden_dim: int = 128
    ):
        super().__init__()
        # Rotation embedding using a linear layer followed by LayerNorm and ReLU activation.
        self.rot_embed = nn.Sequential(
            nn.Linear(3, rot_embed_dim),
            nn.LayerNorm(rot_embed_dim),
            nn.ReLU(),
        )
        # Time embedding using sinusoidal positional encoding.
        self.time_embed = SinusoidalPositionEmbedder(time_embed_dim)

        # Define an MLP with two hidden layers and ReLU activation to predict the 3D score.
        self.net = nn.Sequential(
            nn.Linear(rot_embed_dim + time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # Output the 3D score vector.
        )

        # Initialize weights of the network via Xavier uniform distribution.
        for layer in self.parameters():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, rot_mat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param rot_mat: Tensor of shape (batch, 3, 3) representing rotation as a rotation matrix.
        :param t: Tensor of shape (batch,) representing diffusion time.
        :return: Tensor of shape (batch, 3) with the predicted score vectors.
        """
        # Convert rotation matrix to rotation vector.
        rot_vec = rotmat_to_rotvec(rot_mat)  # (batch, 3)
        # Embed the rotation vector.
        rot_emb = self.rot_embed(rot_vec)  # (batch, rot_embed_dim)
        # Embed the time value.
        t_emb = self.time_embed(t)  # (batch, time_embed_dim)
        # Concatenate along the feature dimension.
        x = torch.cat(
            torch.broadcast_tensors(rot_emb, t_emb), dim=-1
        )  # (batch, rot_embed_dim + time_embed_dim)
        # Compute the score prediction.
        score_pred = self.net(x)

        return score_pred


class DiGMixSO3SDE(DiGSO3SDE):
    def sample_multiple_igso3(
        self,
        mus: torch.Tensor,
        sigmas: torch.Tensor,
        weights: torch.Tensor,
        num_samples: int,
        device: DeviceLikeType | None = None,
    ) -> torch.Tensor:
        # ensure all tensors are on the right device
        mus = mus.to(device)
        sigmas = sigmas.to(device)
        weights = weights.to(device)

        # draw mixture components
        k = torch.multinomial(weights, num_samples, replacement=True)  # (B,)
        sigma = sigmas[k]  # (B,)
        mu = mus[k]  # (B,3,3)

        # sample random rotation from IGSO3 noise
        r = self.igso3.sample(sigma, num_samples=1).squeeze(-3)  # (B,3,3)

        # apply mean rotation
        x0 = mu @ r  # (B,3,3)

        return x0
