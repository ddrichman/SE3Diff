import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._prims_common import DeviceLikeType
from tqdm.auto import trange

from bioemu.denoiser import EulerMaruyamaPredictor
from bioemu.so3_sde import DiGSO3SDE, angle_from_rotmat, rotmat_to_rotvec
from se3diff.models import DiGMixSO3SDE


def _get_so3_score(
    X_t: torch.Tensor,
    sde: DiGSO3SDE,
    score_model: nn.Module,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate predicted score for the batch.

    Args:
        X_t: Current rotation vector. Shape [batch_size, 3].
        sde: SDE.
        score_model: Score model.
        t: Diffusion timestep. Shape [batch_size,]
    """
    tmp = score_model(X_t, t)  # (B,3)
    score = tmp * sde.get_score_scaling(t).unsqueeze(-1)

    return score


def reverse_diffusion(
    sde: DiGSO3SDE,
    score_model: nn.Module,
    *,
    device: DeviceLikeType | None = None,
    batch_size: int = 4096,
    num_steps: int = 200,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: use heun denoiser

    sde.to(device)
    score_model.to(device).eval()

    x_t = sde.prior_sampling((batch_size, 3, 3), device=device)
    predictor = EulerMaruyamaPredictor(
        corruption=sde, noise_weight=1.0, marginal_concentration_factor=1.0
    )
    dt = -1.0 / torch.tensor(num_steps, device=device)
    t_vals = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    xs_list = [x_t]

    for i in trange(num_steps, desc="Reverse diffusion", leave=False):
        t_val = t_vals[i].item()
        t = torch.full((batch_size,), t_val, device=device)
        X_t = rotmat_to_rotvec(x_t)  # type: ignore

        with torch.no_grad():
            score = _get_so3_score(X_t, sde, score_model, t)  # (B,3)

        x_t, _ = predictor.update_given_score(
            x=x_t, t=t, dt=dt, batch_idx=None, score=score  # type: ignore
        )
        xs_list.append(x_t)  # type: ignore

    xs = torch.stack(xs_list, dim=0)  # (T+1,B,3,3)

    return xs, t_vals


def igso3_marginal_pdf(
    omega: torch.Tensor,
    omega_0: torch.Tensor,
    sigma: torch.Tensor,
    l_grid: torch.Tensor,
    tol: float = 1e-7,
) -> torch.Tensor:
    """Compute the marginal PDF of the IGSO(3) distribution.

    Args:
        omega: The angle between the two rotations.
        omega_0: The angle between the reference rotation and the first rotation.
        sigma: The standard deviation of the Gaussian noise.
        l_grid: The grid of irreducible representations.
        tol: A small tolerance value to avoid division by zero.
    Returns:
        The marginal PDF of the IGSO(3) distribution.
    """

    # assert omega and omega_0 are broadcastable

    denom_sin_0 = torch.sin(0.5 * omega_0)
    denom_sin = torch.sin(0.5 * omega)

    l_fac_1 = 2.0 * l_grid + 1.0
    l_fac_2 = -l_grid * (l_grid + 1.0)

    # Pre-compute numerator of expansion which only depends on angles.
    numerator_sin_0 = torch.sin((l_grid + 1 / 2) * omega_0.unsqueeze(-1))
    numerator_sin = torch.sin((l_grid + 1 / 2) * omega.unsqueeze(-1))

    exponential_term = torch.exp(l_fac_2 * sigma.unsqueeze(-1) ** 2 / 2)

    # Compute series expansion
    f_igso = torch.sum(exponential_term * numerator_sin * numerator_sin_0, dim=-1)
    # Finalize expansion. Offset for stability can be added since omega is [0,pi] and sin(omega/2)
    # is positive in this interval.
    f_igso = f_igso * denom_sin / (denom_sin_0 + tol)

    # For small omega, accumulate limit of sine fraction instead:
    # lim[x->0] sin((l+1/2)x) / sin(x/2) = 2l + 1
    f_limw = torch.sum(exponential_term * l_fac_1 * numerator_sin, dim=-1)
    f_limw = f_limw * denom_sin

    # Replace values at small omega with limit.
    f_igso = torch.where(omega_0 <= tol, f_limw, f_igso)

    # Remove remaining numerical problems
    f_igso = torch.where(
        torch.logical_or(torch.isinf(f_igso), torch.isnan(f_igso)),
        torch.zeros_like(f_igso),
        f_igso,
    )

    f_igso = f_igso * 2.0 / math.pi

    # Clamp to avoid negative values
    f_igso = torch.clamp(f_igso, min=0.0)

    return f_igso


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


def igso3_expansion(
    omega: torch.Tensor, sigma: torch.Tensor, l_grid: torch.Tensor, tol=1e-7
) -> torch.Tensor:
    """Arbitrary dimensional version of the IGSO(3) expansion in BioEmu."""

    # assert omega and sigma are broadcastable

    # Pre-compute sine in denominator and clamp for stability.
    denom_sin = torch.sin(0.5 * omega)

    # Pre-compute terms that rely only on expansion orders.
    l_fac_1 = 2.0 * l_grid + 1.0
    l_fac_2 = -l_grid * (l_grid + 1.0)

    # Pre-compute numerator of expansion which only depends on angles.
    numerator_sin = torch.sin((l_grid + 1 / 2) * omega.unsqueeze(-1))

    # Pre-compute exponential term with (2l+1) prefactor.
    exponential_term = l_fac_1 * torch.exp(l_fac_2 * sigma.unsqueeze(-1) ** 2 / 2)

    # Compute series expansion
    f_igso = torch.sum(exponential_term * numerator_sin, dim=-1)
    # For small omega, accumulate limit of sine fraction instead:
    # lim[x->0] sin((l+1/2)x) / sin(x/2) = 2l + 1
    f_limw = torch.sum(exponential_term * l_fac_1, dim=-1)

    # Finalize expansion. Offset for stability can be added since omega is [0,pi] and sin(omega/2)
    # is positive in this interval.
    f_igso = f_igso / (denom_sin + tol)

    # Replace values at small omega with limit.
    f_igso = torch.where(omega <= tol, f_limw, f_igso)

    # Remove remaining numerical problems
    f_igso = torch.where(
        torch.logical_or(torch.isinf(f_igso), torch.isnan(f_igso)),
        torch.zeros_like(f_igso),
        f_igso,
    )

    # Clamp to avoid negative values
    f_igso = torch.clamp(f_igso, min=0.0)

    return f_igso


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
    x_t = sde.sample_marginal(x_0, t)  # (B,3,3)

    # 4) get relative rotation vector q_t = Log(x_0^T x_t)
    # and rotation vector X_t = Log(x_t)
    q_t = rotmat_to_rotvec(torch.einsum("...ki,...kj->...ij", x_0, x_t))  # (B,3)
    X_t = rotmat_to_rotvec(x_t)  # (B,3)

    # 5) compute true score and model prediction
    true_score = sde.compute_score(q_t, t)  # (B,3)
    pred_score = score_model(X_t, t)  # (B,3)

    # 6) L2 denoising score‐matching loss
    loss = F.mse_loss(
        pred_score, true_score / (sde.get_score_scaling(t).unsqueeze(-1) + tol)
    )

    return loss
