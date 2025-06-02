import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._prims_common import DeviceLikeType
from tqdm.auto import trange

from bioemu.denoiser import EulerMaruyamaPredictor
from bioemu.so3_sde import (
    SO3SDE,
    angle_from_rotmat,
    igso3_marginal_pdf,
    rotmat_to_rotvec,
)
from se3diff.models import DiGMixSO3SDE


def _get_so3_score(
    x_t: torch.Tensor,
    sde: SO3SDE,
    score_model: nn.Module,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate predicted score for the batch.

    Args:
        x_t: Current rotation matrix at time t. Shape [batch_size, 3, 3].
        sde: SDE.
        score_model: Score model.
        t: Diffusion timestep. Shape [batch_size,]
    """
    tmp = score_model(x_t, t)  # (B, 3)
    score = tmp * sde.get_score_scaling(t).unsqueeze(-1)

    return score


@torch.no_grad()
def reverse_diffusion(
    sde: SO3SDE,
    score_model: nn.Module,
    *,
    device: DeviceLikeType | None = None,
    batch_size: int = 4096,
    num_steps: int = 200,
) -> tuple[torch.Tensor, torch.Tensor]:

    sde.to(device)
    score_model.to(device)

    x_t = sde.prior_sampling((batch_size, 3, 3), device=device)
    predictor = EulerMaruyamaPredictor(
        corruption=sde, noise_weight=1.0, marginal_concentration_factor=1.0
    )
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    dts = torch.diff(timesteps)

    xs_list = [x_t]

    for i in trange(num_steps, desc="Reverse diffusion", leave=False):
        timestep = timesteps[i].item()
        t = torch.full((batch_size,), timestep, device=device)

        score = _get_so3_score(x_t, sde, score_model, t)  # (B, 3)

        x_t = predictor.update_given_score(
            x=x_t, t=t, dt=dts[i], score=score  # type: ignore
        )[0]
        xs_list.append(x_t)  # type: ignore

    xs = torch.stack(xs_list, dim=0)  # (T+1, B, 3, 3)

    return xs, timesteps


def igso3_mixture_marginal_pdf(
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    weights: torch.Tensor,
    l_max: int = 1000,
    num_points: int = 1000,
    tol: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor]:

    device = mus.device

    omega = torch.linspace(0, math.pi, num_points, device=device)  # (omega,)
    l_grid = torch.arange(l_max, device=device)  # (L,)
    omega_0 = angle_from_rotmat(mus)[0]  # (K,)

    pdfs = igso3_marginal_pdf(
        omega.unsqueeze(0),  # (1, omega)
        omega_0.unsqueeze(1),  # (K, 1)
        sigmas.unsqueeze(1),  # (K, 1)
        l_grid,
        tol=tol,
    )

    pdf = (weights.unsqueeze(-1) * pdfs).sum(dim=0)

    # Clamp to avoid negative values
    pdf = torch.clamp(pdf, min=0.0)

    return omega, pdf


def compute_train_loss(
    sde: DiGMixSO3SDE,
    score_model: nn.Module,
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    weights: torch.Tensor,
    device: DeviceLikeType | None = None,
    batch_size: int = 4096,
    tol: float = 1e-7,
) -> torch.Tensor:

    # 1) draw random samples x_0 from IGSO(3) mixture
    x_0 = sde.sample_multiple_igso3(
        mus, sigmas, weights, batch_size, device=device
    )  # (B,3,3)

    # 2) draw random times t from U[0,1]
    t = torch.rand(batch_size, device=device)

    # 3) get noisy rotations x_t from p_t(x | x_0)
    x_t = sde.sample_marginal(x_0, t)  # (B, 3, 3)

    # 4) get relative rotation vector q_t = Log(x_0^T x_t)
    q_t = rotmat_to_rotvec(torch.einsum("...ki,...kj->...ij", x_0, x_t))  # (B, 3)

    # 5) compute true score and model prediction
    true_score = sde.compute_score(q_t, t)  # (B, 3)
    pred_score = score_model(x_t, t)  # (B, 3)

    # 6) L2 denoising score‐matching loss
    loss = F.mse_loss(
        pred_score, true_score / (sde.get_score_scaling(t).unsqueeze(-1) + tol)
    )

    return loss
