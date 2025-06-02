import torch
import torch.nn as nn
from torch._prims_common import DeviceLikeType
from tqdm.auto import trange

from bioemu.denoiser import EulerMaruyamaPredictor
from bioemu.ppft import (
    compute_ev_loss_from_int_dws,
    compute_ev_loss_from_ws,
    compute_int_dws,
    compute_int_u_u_dt,
    compute_kl_loss_from_int_dws,
    compute_kl_loss_from_ws,
    compute_ws,
)
from bioemu.so3_sde import SO3SDE, angle_from_rotmat, igso3_expansion, rotmat_to_rotvec
from se3diff.train import _get_so3_score


@torch.no_grad()
def reverse_finetune_diffusion(
    sde: SO3SDE,
    score_model: nn.Module,
    finetune_model: nn.Module,
    *,
    device: DeviceLikeType | None = None,
    batch_size: int = 4096,
    num_steps: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    sde.to(device)
    score_model.to(device)
    finetune_model.to(device)

    x_t = sde.prior_sampling((batch_size, 3, 3), device=device)
    predictor = EulerMaruyamaPredictor(
        corruption=sde, noise_weight=1.0, marginal_concentration_factor=1.0
    )
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    dts = torch.diff(timesteps)  # (T,)

    xs_list = [x_t]
    dWs_list = []

    for i in trange(num_steps, desc="Reverse diffusion", leave=False):
        timestep = timesteps[i].item()
        t = torch.full((batch_size,), timestep, device=device)

        score = _get_so3_score(x_t, sde, score_model, t)  # (B, 3)
        finetune_score = finetune_model(x_t, t)  # (B, 3)

        x_t, _, dW_t = predictor.update_given_score(  # type: ignore
            x=x_t,
            t=t,
            dt=dts[i],
            score=score,
            finetune_score=finetune_score,
            with_noise=True,
        )
        xs_list.append(x_t)
        dWs_list.append(dW_t)

    xs = torch.stack(xs_list, dim=0)  # (T+1, B, 3, 3)
    dWs = torch.stack(dWs_list, dim=0)  # (T, B, 3)

    return xs, timesteps, dWs


def assign_igso3(
    x_0: torch.Tensor,
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    weights: torch.Tensor,
    l_max: int = 1000,
    tol: float = 1e-7,
) -> torch.Tensor:
    """Compute the the probability of each Gaussian component for each sample in x_0.
    Args:
        x_0: Tensor of shape (B, 3, 3) representing the rotation matrices.
        mus: Tensor of shape (K, 3, 3) representing the means of the Gaussian components.
        sigmas: Tensor of shape (K,) representing the standard deviations of the Gaussian components.
        weights: Tensor of shape (K,) representing the weights of the Gaussian components.
    Returns:
        Tensor of shape (B, K) representing the probabilities of each Gaussian component for each sample in x_0.
    """

    x_0_rel = torch.einsum("k...ij,b...il->bk...jl", mus, x_0)  # (B,K,3,3)
    x_0_rel_angle = angle_from_rotmat(x_0_rel)[0]  # (B,K)
    l_grid = torch.arange(l_max, device=x_0.device)
    pdf = igso3_expansion(x_0_rel_angle, sigmas, l_grid, tol=tol) * weights  # (B,K)
    hs = pdf / (torch.sum(pdf, dim=-1, keepdim=True) + tol)  # (B,K)

    return hs


def compute_finetune_loss(
    sde: SO3SDE,
    score_model: nn.Module,
    finetune_model: nn.Module,
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    h_stars: torch.Tensor,
    device: DeviceLikeType | None = None,
    lambda_: float = 0.1,
    batch_size: int = 4096,
    num_steps: int = 200,
    l_max: int = 1000,
    tol: float = 1e-7,
) -> torch.Tensor:
    """Compute the loss for the fine-tuning process."""
    xs, timesteps, dWs = reverse_finetune_diffusion(
        sde,
        score_model,
        finetune_model,
        device=device,
        batch_size=batch_size,
        num_steps=num_steps,
    )

    # Track the gradients here
    # us = finetune_model(xs[:-1], timesteps[:-1].unsqueeze(-1))  # (T, B, 3)
    us_list = []
    for i in trange(num_steps, desc="Reverse diffusion", leave=False):
        timestep = timesteps[i].item()
        t = torch.full((batch_size,), timestep, device=device)
        x_t = xs[i]  # (B, 3, 3)
        u_t = finetune_model(x_t, t)  # (B, 3)
        us_list.append(u_t)
    us = torch.stack(us_list, dim=0)  # (T, B, 3)
    hs = assign_igso3(xs[-1], mus, sigmas, h_stars, l_max=l_max, tol=tol)  # (B, K)
    dts = torch.diff(timesteps)  # (T,)
    int_u_u_dt = compute_int_u_u_dt(us=us, dts=dts)  # (B,)

    # ws = compute_ws(us=us, dWs=dWs, dts=dts)
    # loss_ev = compute_ev_loss_from_ws(ws=ws, hs=hs, h_stars=h_stars, tol=tol)
    # loss_kl = compute_kl_loss_from_ws(int_u_u_dt=int_u_u_dt, ws=ws)

    int_dws = compute_int_dws(us=us, dWs=dWs)  # (B,)
    loss_ev = compute_ev_loss_from_int_dws(
        int_dws=int_dws, hs=hs, h_stars=h_stars, tol=tol
    )
    loss_kl = compute_kl_loss_from_int_dws(int_u_u_dt=int_u_u_dt, int_dws=int_dws)

    loss = loss_ev + lambda_ * loss_kl

    return loss
