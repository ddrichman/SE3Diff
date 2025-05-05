import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._prims_common import DeviceLikeType

from bioemu.models import SinusoidalPositionEmbedder
from bioemu.so3_sde import DiGSO3SDE


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

    def forward(self, rot_vec: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param rot_vec: Tensor of shape (batch, 3) representing rotation in axis-angle form.
        :param t: Tensor of shape (batch,) representing diffusion time.
        :return: Tensor of shape (batch, 3) with the predicted score vectors.
        """
        # Embed the rotation vector.
        rot_emb = self.rot_embed(rot_vec)  # (batch, rot_embed_dim)
        # Embed the time value.
        t_emb = self.time_embed(t)  # (batch, time_embed_dim)
        # Concatenate along the feature dimension.
        x = torch.cat(
            [rot_emb, t_emb], dim=-1
        )  # (batch, rot_embed_dim + time_embed_dim)
        # Compute the score prediction.
        score_pred = self.net(x)

        return score_pred


class SO3EquivScoreNet(nn.Module):
    """
    SO(3)-equivariant score network.
    Takes an axis-angle rotation vector `rot_vec` and timestep `t`,
    extracts the rotation magnitude as an invariant, and outputs
    a scaled vector preserving equivariance: f(R rot_vec R^T) = R f(rot_vec).
    """

    def __init__(
        self,
        time_embed_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        # Time embedding
        self.time_embed = SinusoidalPositionEmbedder(time_embed_dim)
        # MLP to predict scalar multiplier alpha(|rot_vec|, t)
        self.scalar_net = nn.Sequential(
            nn.Linear(1 + time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # outputs alpha
        )
        # Initialize weights
        for layer in self.scalar_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, rot_vec: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        :param rot_vec: Tensor of shape (batch, 3) axis-angle vectors.
        :param t: Tensor of shape (batch,) diffusion timesteps.
        :return: Tensor of shape (batch, 3) predicted score vectors, equivariant under SO(3).
        """
        # Compute rotation magnitude |q|
        mag = rot_vec.norm(dim=-1, keepdim=True)  # (batch, 1)
        # Embed time
        t_emb = self.time_embed(t)  # (batch, time_embed_dim)
        # Concatenate magnitude (invariant) and time embedding
        features = torch.cat([mag, t_emb], dim=-1)  # (batch, 1+time_embed_dim)
        # Predict scalar multiplier alpha
        alpha = self.scalar_net(features)  # (batch, 1)
        # Equivariant output: scale original vector
        return alpha * rot_vec


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
