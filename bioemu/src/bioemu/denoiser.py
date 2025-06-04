# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from collections import defaultdict
from typing import NamedTuple, TypedDict, cast

import numpy as np
import torch
from torch import nn
from torch._prims_common import DeviceLikeType
from torch_geometric.utils import to_dense_batch
from tqdm.auto import trange

from bioemu.chemgraph import ChemGraph
from bioemu.sde_lib import SDE, CosineVPSDE
from bioemu.so3_sde import SO3SDE, apply_rotvec_to_rotmat, rotmat_to_rotvec


class SDEs(TypedDict):
    pos: CosineVPSDE
    node_orientations: SO3SDE


class DenoisedSDEPath(NamedTuple):
    batches: list[ChemGraph]
    timesteps: torch.Tensor
    us_batch: dict[str, torch.Tensor]
    dWs_batch: dict[str, torch.Tensor]


class EulerMaruyamaPredictor:
    """Euler-Maruyama predictor."""

    def __init__(
        self,
        *,
        corruption: SDE,
        noise_weight: float = 1.0,
        marginal_concentration_factor: float = 1.0,
    ):
        """
        Args:
            noise_weight: A scalar factor applied to the noise during each update. The parameter controls the stochasticity of the integrator. A value of 1.0 is the
            standard Euler Maruyama integration scheme whilst a value of 0.0 is the probability flow ODE.
            marginal_concentration_factor: A scalar factor that controls the concentration of the sampled data distribution. The sampler targets p(x)^{MCF} where p(x)
            is the data distribution. A value of 1.0 is the standard Euler Maruyama / probability flow ODE integration.

            See feynman/projects/diffusion/sampling/samplers_readme.md for more details.

        """
        self.corruption = corruption
        self.noise_weight = noise_weight
        self.marginal_concentration_factor = marginal_concentration_factor

    def reverse_drift_and_diffusion(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
        finetune_score: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        score_weight = 0.5 * self.marginal_concentration_factor * (1 + self.noise_weight**2)
        drift, diffusion = self.corruption.sde(x=x, t=t, batch_idx=batch_idx)
        drift = drift - diffusion**2 * score * score_weight
        if finetune_score is not None:
            drift = drift + diffusion * finetune_score * score_weight

        return drift, diffusion

    def update_given_drift_and_diffusion(
        self,
        *,
        x: torch.Tensor,
        dt: torch.Tensor,
        drift: torch.Tensor,
        diffusion: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.randn_like(drift)
        dW = self.noise_weight * torch.sqrt(dt.abs()) * z

        # Update to next step using either special update for SDEs on SO(3) or standard update.
        if isinstance(self.corruption, SO3SDE):
            mean = apply_rotvec_to_rotmat(x, drift * dt, tol=self.corruption.tol)
            sample = apply_rotvec_to_rotmat(
                mean,
                diffusion * dW,
                tol=self.corruption.tol,
            )
        elif isinstance(self.corruption, CosineVPSDE):
            mean = x + drift * dt
            sample = mean + diffusion * dW
        else:
            raise NotImplementedError(f"Update for {type(self.corruption)} not implemented.")

        return sample, mean, dW

    def update_given_score(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        score: torch.Tensor,
        finetune_score: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Set up different coefficients and terms.
        drift, diffusion = self.reverse_drift_and_diffusion(
            x=x, t=t, score=score, finetune_score=finetune_score, batch_idx=batch_idx
        )

        # Update to next step using either special update for SDEs on SO(3) or standard update.
        return self.update_given_drift_and_diffusion(x=x, dt=dt, drift=drift, diffusion=diffusion)

    def forward_sde_step(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update to next step using either special update for SDEs on SO(3) or standard update.
        Handles both SO(3) and Euclidean updates."""

        drift, diffusion = self.corruption.sde(x=x, t=t, batch_idx=batch_idx)
        # Update to next step using either special update for SDEs on SO(3) or standard update.
        return self.update_given_drift_and_diffusion(x=x, dt=dt, drift=drift, diffusion=diffusion)

    def traceback_brownian_motion(
        self,
        *,
        x_next: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        score: torch.Tensor,
        finetune_score: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Traceback the Brownian motion from the current state x to the previous state."""
        drift, diffusion = self.reverse_drift_and_diffusion(
            x=x,
            t=t,
            score=score,
            finetune_score=finetune_score,
            batch_idx=batch_idx,
        )
        mean = self.update_given_drift_and_diffusion(
            x=x,
            dt=dt,
            drift=drift,
            diffusion=0.0,  # type: ignore
        )[1]

        if isinstance(self.corruption, SO3SDE):
            dW = rotmat_to_rotvec(torch.einsum("...ji,...jk->...ik", mean, x_next)) / diffusion
        elif isinstance(self.corruption, CosineVPSDE):
            dW = (x_next - mean) / diffusion
        else:
            raise NotImplementedError(f"Update for {type(self.corruption)} not implemented.")

        return dW


def _get_score(
    batch: ChemGraph,
    sdes: SDEs,
    score_model: nn.Module,
    t: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Calculate predicted score for the batch.

    Args:
        batch: Batch of corrupted data.
        sdes: SDEs.
        score_model: Score model.  The score model is parametrized to predict a multiple of the score.
          This function converts the score model output to a score.
        t: Diffusion timestep. Shape [batch_size,]
    """

    batch_idx = batch.batch

    tmp = score_model(batch, t)
    # Score is in axis angle representation [N,3] (vector is along axis of rotation, vector length
    # is rotation angle in radians).
    node_orientations_score = tmp["node_orientations"] * sdes[
        "node_orientations"
    ].get_score_scaling(t, batch_idx=batch_idx).unsqueeze(-1)

    # Score model is trained to predict score * std, so divide by std to get the score.
    _, pos_std = sdes["pos"].marginal_prob(
        x=torch.ones_like(tmp["pos"]),
        t=t,
        batch_idx=batch_idx,
    )
    pos_score = tmp["pos"] / pos_std

    return {"node_orientations": node_orientations_score, "pos": pos_score}


def euler_maruyama_predictor(
    *,
    batch: ChemGraph,
    sdes: SDEs,
    score_model: nn.Module,
    num_steps: int,
    max_t: float,
    min_t: float,
    device: DeviceLikeType | None = None,
) -> ChemGraph:
    """Sample from prior and then denoise."""

    batch = batch.to(device)  # type: ignore
    if isinstance(score_model, nn.Module):
        # permits unit-testing with dummy model
        score_model = score_model.to(device)
    sdes["node_orientations"] = sdes["node_orientations"].to(device)

    batch = batch.replace(
        pos=sdes["pos"].prior_sampling(batch.pos.shape, device=device),
        node_orientations=sdes["node_orientations"].prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )

    timesteps = torch.linspace(max_t, min_t, num_steps + 1, device=device)
    dts = torch.diff(timesteps)
    fields = list(sdes.keys())  # ["pos", "node_orientations"]
    noisers = {
        name: EulerMaruyamaPredictor(
            corruption=cast(SDE, sde),
            noise_weight=1.0,
            marginal_concentration_factor=1.0,
        )
        for name, sde in sdes.items()
    }

    batch_idx = batch.batch

    for i in trange(num_steps, desc="Denoising", leave=False):
        # Set the timestep
        timestep = timesteps[i].item()
        t = torch.full((batch.num_graphs,), timestep, device=device)

        score = _get_score(batch=batch, sdes=sdes, score_model=score_model, t=t)

        # Apply noise.
        vals_hat = {}
        for field in fields:
            vals_hat[field] = noisers[field].update_given_score(  # type: ignore
                x=batch[field],
                t=t,
                dt=dts[i],
                score=score[field],
                batch_idx=batch_idx,
            )[0]
        batch = batch.replace(**vals_hat)

    return batch


def euler_maruyama_predictor_finetune(
    *,
    batch: ChemGraph,
    sdes: SDEs,
    score_model: nn.Module,
    finetune_model: nn.Module,
    num_steps: int,
    max_t: float,
    min_t: float,
    device: DeviceLikeType | None = None,
) -> DenoisedSDEPath:
    """Sample from prior and then denoise."""

    batch = batch.to(device)  # type: ignore
    if isinstance(score_model, nn.Module):
        # permits unit-testing with dummy model
        score_model = score_model.to(device)
    if isinstance(finetune_model, nn.Module):
        # permits unit-testing with dummy model
        finetune_model = finetune_model.to(device)
    sdes["node_orientations"] = sdes["node_orientations"].to(device)

    batch = batch.replace(
        pos=sdes["pos"].prior_sampling(batch.pos.shape, device=device),
        node_orientations=sdes["node_orientations"].prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )

    timesteps = torch.linspace(max_t, min_t, num_steps + 1, device=device)
    dts = torch.diff(timesteps)
    fields = list(sdes.keys())  # ["pos", "node_orientations"]
    noisers = {
        name: EulerMaruyamaPredictor(
            corruption=cast(SDE, sde),
            noise_weight=1.0,
            marginal_concentration_factor=1.0,
        )
        for name, sde in sdes.items()
    }

    batch_idx = batch.batch

    batches = [batch]
    us_list_batch = defaultdict(list)
    dWs_list_batch = defaultdict(list)

    for i in trange(num_steps, desc="Denoising", leave=False):
        # Set the timestep
        timestep = timesteps[i].item()
        t = torch.full((batch.num_graphs,), timestep, device=device)

        score = _get_score(batch=batch, sdes=sdes, score_model=score_model, t=t)
        finetune_score = finetune_model(batch, t)

        # Apply noise.
        vals_hat = {}
        for field in fields:
            vals_hat[field], _, dW_t = noisers[field].update_given_score(  # type: ignore
                x=batch[field],
                t=t,
                dt=dts[i],
                score=score[field],
                finetune_score=finetune_score[field],
                batch_idx=batch_idx,
            )
            u_t = finetune_score[field]
            us_list_batch[field].append(to_dense_batch(u_t, batch_idx)[0])  # (B, L, 3)
            dWs_list_batch[field].append(to_dense_batch(dW_t, batch_idx)[0])  # (B, L, 3)

        batch = batch.replace(**vals_hat)
        batches.append(batch)

    us_batch = {field: torch.stack(us_list_batch[field], dim=0) for field in fields}
    dWs_batch = {field: torch.stack(dWs_list_batch[field], dim=0) for field in fields}

    return DenoisedSDEPath(
        batches=batches,
        timesteps=timesteps,
        us_batch=us_batch,
        dWs_batch=dWs_batch,
    )


def heun_denoiser(
    *,
    batch: ChemGraph,
    sdes: SDEs,
    score_model: nn.Module,
    num_steps: int,
    max_t: float,
    min_t: float,
    noise: float,
    device: DeviceLikeType | None = None,
) -> ChemGraph:
    """Sample from prior and then denoise."""

    batch = batch.to(device)  # type: ignore
    if isinstance(score_model, nn.Module):
        # permits unit-testing with dummy model
        score_model = score_model.to(device)
    sdes["node_orientations"] = sdes["node_orientations"].to(device)

    batch = batch.replace(
        pos=sdes["pos"].prior_sampling(batch.pos.shape, device=device),
        node_orientations=sdes["node_orientations"].prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )

    ts_min = 0.0
    ts_max = 1.0
    timesteps = torch.linspace(max_t, min_t, num_steps + 1, device=device)
    dts = torch.diff(timesteps)
    fields = list(sdes.keys())  # ["pos", "node_orientations"]
    predictors = {
        name: EulerMaruyamaPredictor(
            corruption=cast(SDE, sde),
            noise_weight=0.0,
            marginal_concentration_factor=1.0,
        )
        for name, sde in sdes.items()
    }
    noisers = {
        name: EulerMaruyamaPredictor(
            corruption=cast(SDE, sde),
            noise_weight=1.0,
            marginal_concentration_factor=1.0,
        )
        for name, sde in sdes.items()
    }

    batch_idx = batch.batch

    for i in trange(num_steps, desc="Denoising", leave=False):
        # Set the timestep
        timestep = timesteps[i].item()
        t = torch.full((batch.num_graphs,), timestep, device=device)
        t_next = t + dts[i]  # dt is negative; t_next is slightly less noisy than t.

        # Select temporarily increased noise level t_hat.
        # To be more general than Algorithm 2 in Karras et al. we select a time step between the
        # current and the previous t.
        t_hat = t - noise * dts[i] if (i > 0 and ts_min < t[0] < ts_max) else t

        # Apply noise.
        vals_hat = {}
        for field in fields:
            vals_hat[field] = noisers[field].forward_sde_step(
                x=batch[field], t=t, dt=(t_hat - t)[0], batch_idx=batch_idx
            )[0]
        batch_hat = batch.replace(**vals_hat)

        score_hat = _get_score(batch=batch_hat, sdes=sdes, score_model=score_model, t=t_hat)

        # First-order denoising step from t_hat to t_next.
        drift_hat = {}
        for field in fields:
            drift_hat[field], _ = predictors[field].reverse_drift_and_diffusion(
                x=batch_hat[field],
                t=t_hat,
                score=score_hat[field],
                batch_idx=batch_idx,
            )
        for field in fields:
            batch[field] = predictors[field].update_given_drift_and_diffusion(
                x=batch_hat[field],
                dt=(t_next - t_hat)[0],
                drift=drift_hat[field],
                diffusion=0.0,  # type: ignore
            )[1]

        # Apply 2nd order correction.
        if t_next[0] > 0.0:
            score_next = _get_score(batch=batch, sdes=sdes, score_model=score_model, t=t_next)

            drifts_next = {}
            avg_drift = {}
            for field in fields:
                drifts_next[field], _ = predictors[field].reverse_drift_and_diffusion(
                    x=batch[field],
                    t=t_next,
                    score=score_next[field],
                    batch_idx=batch_idx,
                )
                avg_drift[field] = (drifts_next[field] + drift_hat[field]) / 2
            for field in fields:
                batch[field] = predictors[field].update_given_drift_and_diffusion(
                    x=batch_hat[field],
                    dt=(t_next - t_hat)[0],
                    drift=avg_drift[field],
                    diffusion=0.0,  # type: ignore
                )[1]

    return batch


def heun_denoiser_finetune(
    *,
    batch: ChemGraph,
    sdes: SDEs,
    score_model: nn.Module,
    finetune_model: nn.Module,
    num_steps: int,
    max_t: float,
    min_t: float,
    noise: float,
    device: DeviceLikeType | None = None,
) -> DenoisedSDEPath:
    """Sample from prior and then denoise."""

    batch = batch.to(device)  # type: ignore
    if isinstance(score_model, nn.Module):
        # permits unit-testing with dummy model
        score_model = score_model.to(device)
    if isinstance(finetune_model, nn.Module):
        # permits unit-testing with dummy model
        finetune_model = finetune_model.to(device)
    sdes["node_orientations"] = sdes["node_orientations"].to(device)

    batch = batch.replace(
        pos=sdes["pos"].prior_sampling(batch.pos.shape, device=device),
        node_orientations=sdes["node_orientations"].prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )

    ts_min = 0.0
    ts_max = 1.0
    timesteps = torch.linspace(max_t, min_t, num_steps + 1, device=device)
    dts = torch.diff(timesteps)
    fields = list(sdes.keys())  # ["pos", "node_orientations"]
    predictors = {
        name: EulerMaruyamaPredictor(
            corruption=cast(SDE, sde),
            noise_weight=0.0,
            marginal_concentration_factor=1.0,
        )
        for name, sde in sdes.items()
    }
    noisers = {
        name: EulerMaruyamaPredictor(
            corruption=cast(SDE, sde),
            noise_weight=1.0,
            marginal_concentration_factor=1.0,
        )
        for name, sde in sdes.items()
    }

    batch_idx = batch.batch

    batches = [batch]
    us_list_batch = defaultdict(list)
    dWs_list_batch = defaultdict(list)

    for i in trange(num_steps, desc="Denoising", leave=False):
        # Set the timestep
        timestep = timesteps[i].item()
        t = torch.full((batch.num_graphs,), timestep, device=device)
        t_next = t + dts[i]  # dt is negative; t_next is slightly less noisy than t.

        # Select temporarily increased noise level t_hat.
        # To be more general than Algorithm 2 in Karras et al. we select a time step between the
        # current and the previous t.
        t_hat = t - noise * dts[i] if (i > 0 and ts_min < t[0] < ts_max) else t

        # Apply noise.
        vals_hat = {}
        for field in fields:
            vals_hat[field] = noisers[field].forward_sde_step(
                x=batch[field], t=t, dt=(t_hat - t)[0], batch_idx=batch_idx
            )[0]
        batch_hat = batch.replace(**vals_hat)

        score_hat = _get_score(batch=batch_hat, sdes=sdes, score_model=score_model, t=t_hat)
        finetune_score_hat = finetune_model(batch_hat, t_hat)

        # Store the previous values.
        x = {field: batch[field].clone() for field in fields}
        if i > 0 and ts_min < t[0] < ts_max:
            score = _get_score(batch=batch, sdes=sdes, score_model=score_model, t=t)
            finetune_score = finetune_model(batch, t)
        else:
            score = {field: score_hat[field].clone() for field in fields}
            finetune_score = {field: finetune_score_hat[field].clone() for field in fields}

        # First-order denoising step from t_hat to t_next.
        drift_hat = {}
        for field in fields:
            drift_hat[field], _ = predictors[field].reverse_drift_and_diffusion(
                x=batch_hat[field],
                t=t_hat,
                score=score_hat[field],
                finetune_score=finetune_score_hat[field],
                batch_idx=batch_idx,
            )
        for field in fields:
            batch[field] = predictors[field].update_given_drift_and_diffusion(
                x=batch_hat[field],
                dt=(t_next - t_hat)[0],
                drift=drift_hat[field],
                diffusion=0.0,  # type: ignore
            )[1]

        # Apply 2nd order correction.
        if t_next[0] > 0.0:
            score_next = _get_score(batch=batch, sdes=sdes, score_model=score_model, t=t_next)
            finetune_score_next = finetune_model(batch, t_next)

            drifts_next = {}
            avg_drift = {}
            for field in fields:
                drifts_next[field], _ = predictors[field].reverse_drift_and_diffusion(
                    x=batch[field],
                    t=t_next,
                    score=score_next[field],
                    finetune_score=finetune_score_next[field],
                    batch_idx=batch_idx,
                )
                avg_drift[field] = (drifts_next[field] + drift_hat[field]) / 2
            for field in fields:
                batch[field] = predictors[field].update_given_drift_and_diffusion(
                    x=batch_hat[field],
                    dt=(t_next - t_hat)[0],
                    drift=avg_drift[field],
                    diffusion=0.0,  # type: ignore
                )[1]

        # Store the batch for the current timestep.
        batches.append(batch)
        # Store the noise increments.
        for field in fields:
            dW_t = noisers[field].traceback_brownian_motion(
                x_next=batch[field],
                x=x[field],
                t=t,
                dt=dts[i],
                score=score[field],
                finetune_score=finetune_score[field],
                batch_idx=batch_idx,
            )
            u_t = finetune_score[field]
            us_list_batch[field].append(to_dense_batch(u_t, batch_idx)[0])  # (B, L, 3)
            dWs_list_batch[field].append(to_dense_batch(dW_t, batch_idx)[0])  # (B, L, 3)

    us_batch = {field: torch.stack(us_list_batch[field], dim=0) for field in fields}
    dWs_batch = {field: torch.stack(dWs_list_batch[field], dim=0) for field in fields}

    return DenoisedSDEPath(
        batches=batches,
        timesteps=timesteps,
        us_batch=us_batch,
        dWs_batch=dWs_batch,
    )


def _t_from_lambda(sde: CosineVPSDE, lambda_t: torch.Tensor) -> torch.Tensor:
    """
    Used for DPMsolver. https://arxiv.org/abs/2206.00927 Appendix Section D.4
    """
    f_lambda = -1 / 2 * torch.log(torch.exp(-2 * lambda_t) + 1)
    exponent = f_lambda + torch.log(torch.cos(torch.tensor(np.pi * sde.s / 2 / (1 + sde.s))))
    t_lambda = 2 * (1 + sde.s) / np.pi * torch.acos(torch.exp(exponent)) - sde.s

    return t_lambda


def dpm_solver(
    *,
    batch: ChemGraph,
    sdes: SDEs,
    score_model: nn.Module,
    num_steps: int,
    max_t: float,
    min_t: float,
    device: DeviceLikeType | None = None,
) -> ChemGraph:
    """
    Implements the DPM solver for the VPSDE, with the Cosine noise schedule.
    Following this paper: https://arxiv.org/abs/2206.00927 Algorithm 1 DPM-Solver-2.
    DPM solver is used only for positions, not node orientations.
    """
    assert max_t < 1.0

    batch = batch.to(device)  # type: ignore
    if isinstance(score_model, nn.Module):
        # permits unit-testing with dummy model
        score_model = score_model.to(device)
    pos_sde = sdes["pos"]
    so3_sde = sdes["node_orientations"].to(device)

    batch = batch.replace(
        pos=pos_sde.prior_sampling(batch.pos.shape, device=device),
        node_orientations=so3_sde.prior_sampling(batch.node_orientations.shape, device=device),
    )

    timesteps = torch.linspace(max_t, min_t, num_steps + 1, device=device)
    dts = torch.diff(timesteps)

    batch_idx = batch.batch

    for i in trange(num_steps, desc="Denoising", leave=False):
        timestep = timesteps[i].item()
        t = torch.full((batch.num_graphs,), timestep, device=device)
        t_next = t + dts[i]

        # Evaluate score
        score = _get_score(batch=batch, sdes=sdes, score_model=score_model, t=t)

        # t_{i-1} in the algorithm is the current t
        alpha_t, sigma_t = pos_sde.mean_coeff_and_std(x=batch.pos, t=t, batch_idx=batch_idx)
        lambda_t = torch.log(alpha_t / sigma_t)
        alpha_t_next, sigma_t_next = pos_sde.mean_coeff_and_std(
            x=batch.pos, t=t_next, batch_idx=batch_idx
        )
        lambda_t_next = torch.log(alpha_t_next / sigma_t_next)

        # t_next < t, lambda_t_next > lambda_t
        h_t = lambda_t_next - lambda_t

        # For a given noise schedule (cosine is what we use), compute the intermediate t_lambda
        lambda_t_middle = (lambda_t + lambda_t_next) / 2
        t_lambda = _t_from_lambda(sde=pos_sde, lambda_t=lambda_t_middle)

        # t_lambda has all the same components
        t_lambda = torch.full((batch.num_graphs,), t_lambda[0][0].item(), device=device)
        alpha_t_lambda, sigma_t_lambda = pos_sde.mean_coeff_and_std(
            x=batch.pos, t=t_lambda, batch_idx=batch_idx
        )

        # Note in the paper the algorithm uses noise instead of score, but we use score.
        # So the formulation is slightly different in the prefactor.
        u = (
            alpha_t_lambda / alpha_t * batch.pos
            + sigma_t_lambda * sigma_t * (torch.exp(h_t / 2) - 1) * score["pos"]
        )

        # Update positions to the intermediate timestep t_lambda
        batch_u = batch.replace(pos=u)

        # Get node orientation at t_lambda

        # Denoise from t to t_lambda
        so3_predictor = EulerMaruyamaPredictor(
            corruption=so3_sde, noise_weight=0.0, marginal_concentration_factor=1.0
        )
        drift, _ = so3_predictor.reverse_drift_and_diffusion(
            x=batch.node_orientations,
            t=t,
            score=score["node_orientations"],
            batch_idx=batch_idx,
        )
        sample = so3_predictor.update_given_drift_and_diffusion(
            x=batch.node_orientations,
            dt=(t_lambda - t)[0],
            drift=drift,
            diffusion=0.0,  # type: ignore
        )[
            1
        ]  # dt is negative, diffusion is 0
        batch_u = batch_u.replace(node_orientations=sample)

        # Correction step
        # Evaluate score at updated pos and node orientations
        score_u = _get_score(batch=batch_u, sdes=sdes, score_model=score_model, t=t_lambda)

        pos_next = (
            alpha_t_next / alpha_t * batch.pos
            + sigma_t_next * sigma_t_lambda * (torch.exp(h_t) - 1) * score_u["pos"]
        )

        batch_next = batch.replace(pos=pos_next)

        # Try a 2nd order correction
        node_score = (
            score_u["node_orientations"]
            + 0.5
            * (score_u["node_orientations"] - score["node_orientations"])
            / (t_lambda - t)[0]
            * dts[i]
        )
        drift, _ = so3_predictor.reverse_drift_and_diffusion(
            x=batch_u.node_orientations,
            t=t_lambda,
            score=node_score,
            batch_idx=batch_idx,
        )
        sample = so3_predictor.update_given_drift_and_diffusion(
            x=batch.node_orientations,
            dt=dts[i],
            drift=drift,
            diffusion=0.0,  # type: ignore
        )[
            1
        ]  # dt is negative, diffusion is 0
        batch = batch_next.replace(node_orientations=sample)

    return batch


def sde_dpm_solver_finetune(
    *,
    batch: ChemGraph,
    sdes: SDEs,
    score_model: nn.Module,
    finetune_model: nn.Module,
    num_steps: int,
    max_t: float,
    min_t: float,
    device: DeviceLikeType | None = None,
) -> DenoisedSDEPath: ...
