import torch


def riemannian_ito_integral(fs: torch.Tensor, dWs: torch.Tensor) -> torch.Tensor:
    """Computes the Riemannian Ito integral for a given tensor field and Wiener increments.
    Args:
        fs: Tensor of shape (T, B, D) representing the tensor field.
        dWs: Tensor of shape (T, B, D) representing the Wiener increments.
    Returns:
        Tensor of shape (B,) representing the Riemannian Ito integral.
    """

    return torch.einsum("tb...i,tb...i->b...", fs, dWs)


def riemannian_quadratic_covariation(
    fs: torch.Tensor, gs: torch.Tensor, dts: torch.Tensor
) -> torch.Tensor:
    """Computes the Riemannian quadratic covariation for two tensor fields.
    Args:
        fs: Tensor of shape (T, B, D) representing the first tensor field.
        gs: Tensor of shape (T, B, D) representing the second tensor field.
        dts: Tensor of shape (T,) representing the time increments.
    Returns:
        Tensor of shape (B,) representing the Riemannian quadratic covariation.
    """

    return torch.einsum("tb...i,tb...i,t->b...", fs, gs, dts)


def rloo_baseline(fs: torch.Tensor) -> torch.Tensor:
    """Compute the baseline for given samples using the leave-one-out method.
    Args:
        fs: Tensor of shape (B,) representing the samples.
    Returns:
        Tensor of shape (B,) representing the baseline.
    """

    B = fs.shape[0]
    baseline = (fs.sum(dim=0, keepdim=True) - fs) / (B - 1)  # (B,)

    return baseline


def compute_ws(*, us: torch.Tensor, dWs: torch.Tensor, dts: torch.Tensor) -> torch.Tensor:
    """Computes the importance weights for the reverse diffusion process.
    Args:
        us: Tensor of shape (T, B, D) representing the controlled term.
        dWs: Tensor of shape (T, B, D) representing the Wiener increments.
        dts: Tensor of shape (T,) representing the time increments.
    Returns:
        Tensor of shape (B,) representing the importance weights.
    """

    diff = us - us.detach()  # (T, B, D)

    # Integrate from t=1 to t=0, i.e., reverse time
    int_diff_dW = riemannian_ito_integral(diff, -dWs)  # (B,)
    int_diff_diff_dt = riemannian_quadratic_covariation(diff, diff, -dts)  # (B,)
    ws = torch.exp(int_diff_dW - int_diff_diff_dt / 2)  # (B,)

    return ws


def compute_int_dws(*, us: torch.Tensor, dWs: torch.Tensor) -> torch.Tensor:
    """Computes the importance weights for the reverse diffusion process.
    Args:
        us: Tensor of shape (T, B, D) representing the controlled term.
        dWs: Tensor of shape (T, B, D) representing the Wiener increments.
    Returns:
        Tensor of shape (B,) representing the integrated gradient of the importance weights,
        such that \nabla int_dw = \nabla w
    """

    # Integrate from t=1 to t=0, i.e., reverse time
    int_dws = riemannian_ito_integral(us, -dWs)  # (B,)

    return int_dws


def compute_ev_loss(
    *,
    ws: torch.Tensor,
    hs: torch.Tensor,
    h_stars: torch.Tensor,
    from_int_dws: bool = True,
    use_stab: bool = True,
    tol: float = 1e-7,
) -> torch.Tensor:
    """Compute the expected value loss from the importance weights.
    Args:
        ws: Tensor of shape (B,) representing the importance weights.
        hs: Tensor of shape (B, K) representing the sampled observable values.
        h_stars: Tensor of shape (K,) or (B, K) representing the ground truth expectation values.
        from_int_dws: Whether to compute the loss from integrated gradients of the importance weights.
        use_stab: Whether to use stability correction in the loss computation.
        tol: Small value to avoid division by zero.
    Returns:
        Tensor representing the expected value loss.
    """

    B = ws.shape[0]
    ws_ = ws.unsqueeze(1)  # (B, 1)
    dhs = hs - h_stars  # (B, K)

    if use_stab and B > 1:  # Stability correction only makes sense for degree of freedom B > 1
        pbar = torch.mean(hs, dim=0)
        stab = torch.sum(pbar, dim=0) / (pbar + tol)  # (K,)
        stab = stab / torch.mean(stab)  # (K,)
    else:
        stab = torch.tensor(1.0, device=ws.device)

    # Compute the importance sample estimator
    if from_int_dws:
        # 1) s_1[k] = \sum_i int_dw_i * h_{i,k}
        s_1 = torch.sum(ws_ * dhs, dim=0)  # (K,)

        # 2) s_2[k] = \sum_i h_{i,k}
        s_2 = torch.sum(dhs, dim=0)  # (K,)

        # 3) s_3[k] = \sum_i int_dw_i * h_{i,k}^2
        s_3 = torch.sum(ws_ * dhs**2, dim=0)  # (K,)

        # 4) 2(s_1 * s_2 - s_3) = \sum_{i,j}(int_dw_i + int_dw_j) h_i h_j
        # (\sum_{i\neq j} (int_dw_i + int_dw_j) h_i h_j) / (B(B-1))
        loss_ev = 2 * (s_1 * s_2 - s_3) * stab / (B * (B - 1))  # (K,)
    else:
        w_dhs = ws_ * dhs  # (B, K)

        # (\sum_{i\neq j} w_i w_j h_i h_j) / (B(B-1))
        loss_ev = (
            (torch.sum(w_dhs, dim=0) ** 2 - torch.sum(w_dhs**2, dim=0)) * stab / (B * (B - 1))
        )  # (K,)

    return torch.sum(loss_ev)


def compute_int_u_u_dt(*, us: torch.Tensor, dts: torch.Tensor) -> torch.Tensor:
    """Compute the quadratic variation of the controlled term.
    Args:
        us: Tensor of shape (T, B, D) representing the controlled term.
        dts: Tensor of shape (T,) representing the time increments.
    Returns:
        Tensor of shape (B,) representing the integrated term.
    """

    return riemannian_quadratic_covariation(us, us, -dts)  # (B,)


def compute_kl_loss(
    *,
    ws: torch.Tensor,
    int_u_u_dt: torch.Tensor,
    int_u_u_dt_sg: torch.Tensor,
    from_int_dws: bool = True,
    use_rloo: bool = True,
) -> torch.Tensor:
    """Compute the KL divergence loss for the fine-tuning process.

    .. math ::
        \nabla_\theta\mathbb{E}_{\mathbb{P}_{\operatorname{sg}(\theta)}}\left[\frac{1}{2} w_\theta \int_0^1 \norm{u_\theta}_\mathcal{M}^2\mathrm{d}t \right] = \nabla_\theta\mathbb{E}_{\mathbb{P}_{\operatorname{sg}(\theta)}}\left[\frac{1}{2} \int_0^1 \left\langle{u_\theta, \mathrm{d}\mathbf{W}_t^\mathcal{M}}\right\rangle_\mathcal{M}^2\left( \int_0^1 \norm{u_\theta}_\mathcal{M}^2\mathrm{d}t\right)_{\operatorname{sg}} + \frac{1}{2} \int_0^1 \norm{u_\theta}_\mathcal{M}^2\mathrm{d}t\right]

    Args:
        ws: Tensor of shape (B,) representing the importance weights.
        int_u_u_dt: Tensor of shape (B,) representing the quadratic variation of the controlled term.
        int_u_u_dt_sg: Tensor of shape (B,) representing the quadratic variation of the controlled term
            with stop gradient applied. `int_u_u_dt_sg` is NOT the same as `int_u_u_dt.detach()`;
            it contains the full integral of the controlled term, while `int_u_u_dt` is the integral
            only over the current time interval.
        from_int_dws: Whether to compute the loss from integrated gradients of the importance weights.
        use_rloo: Whether to use the REINFORCE leave-one-out baseline for the loss computation.
    Returns:
        Tensor representing the KL divergence loss.
    """

    if use_rloo:
        baseline = rloo_baseline(int_u_u_dt.detach())
        baseline_sg = rloo_baseline(int_u_u_dt_sg)
    else:
        baseline = torch.zeros_like(int_u_u_dt.detach())
        baseline_sg = torch.zeros_like(int_u_u_dt_sg)

    if from_int_dws:
        w_int_u_u_dt = int_u_u_dt - baseline + (int_u_u_dt_sg - baseline_sg) * ws  # (B,)
    else:
        # This is not applicable for training if integrals are cut into time intervals.
        # The loss here is usually used for validation (i.e., `ws` is set to 1.0).
        w_int_u_u_dt = (int_u_u_dt - baseline) * ws  # (B,)

    loss_kl = torch.mean(w_int_u_u_dt) / 2

    return loss_kl
