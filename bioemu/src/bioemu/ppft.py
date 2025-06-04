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


def compute_ev_loss_from_ws(
    *, ws: torch.Tensor, hs: torch.Tensor, h_stars: torch.Tensor, tol: float = 1e-7
) -> torch.Tensor:
    """Compute the expected value loss from the importance weights.
    Args:
        ws: Tensor of shape (B,) representing the importance weights.
        hs: Tensor of shape (B, K) representing the sampled observable values.
        h_stars: Tensor of shape (K,) or (B, K) representing the ground truth expectation values.
    Returns:
        Tensor representing the expected value loss.
    """

    B = ws.shape[0]
    weighted_hs = ws.unsqueeze(1) * (hs - h_stars)  # (B, K)
    # weighted_hs = ws.unsqueeze(1) * hs - h_stars  # (B, K)

    pbar = torch.mean(hs, dim=0)
    stab = torch.sum(pbar, dim=0) / (pbar + tol)  # (K,)
    # stab = torch.ones_like(pbar)
    stab = stab / torch.mean(stab)  # (K,)

    # Compute the importance sample estimator
    # (\sum_{i\neq j} w_i w_j h_i h_j) / (B(B-1))
    loss_ev = (
        (torch.sum(weighted_hs, dim=0) ** 2 - torch.sum(weighted_hs**2, dim=0))
        * stab
        / (B * (B - 1))
    )  # (K,)

    return torch.sum(loss_ev)


def compute_ev_loss_from_int_dws(
    *, int_dws: torch.Tensor, hs: torch.Tensor, h_stars: torch.Tensor, tol: float = 1e-7
) -> torch.Tensor:
    """Compute the expected value loss from the integrated gradient of the importance weights.
    Args:
        int_dws: Tensor of shape (B,) representing the integrated gradient of the importance weights.
        hs: Tensor of shape (B, K) representing the sampled observable values.
        h_stars: Tensor of shape (K,) or (B, K) representing the ground truth expectation values.
    Returns:
        Tensor representing the expected value loss.
    """

    B = int_dws.shape[0]

    int_dws_ = int_dws.unsqueeze(1)  # (B, 1)
    dhs = hs - h_stars  # (B, K)

    print(hs, h_stars)

    pbar = torch.mean(hs, dim=0)
    stab = torch.sum(pbar, dim=0) / (pbar + tol)  # (K,)
    # stab = torch.ones_like(pbar)
    stab = stab / torch.mean(stab)  # (K,)

    # Compute the importance sample estimator
    # (\sum_{i\neq j} (int_dw_i + int_dw_j) h_i h_j) / (B(B-1))

    # 1) s_1[k] = \sum_i int_dw_i * h_{i,k}
    s_1 = torch.sum(int_dws_ * dhs, dim=0)  # (K,)

    # 2) s_2[k] = \sum_i h_{i,k}
    s_2 = torch.sum(dhs, dim=0)  # (K,)

    # 3) s_3[k] = \sum_i int_dw_i * h_{i,k}^2
    s_3 = torch.sum(int_dws_ * dhs**2, dim=0)  # (K,)

    # 4) combine:  \sum_{i,j}(int_dw_i + int_dw_j) h_i h_j = 2(s_1 * s_2 - s_3)
    loss_ev = 2 * (s_1 * s_2 - s_3) * stab / (B * (B - 1))  # (K,)

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


def compute_kl_loss_from_ws(*, int_u_u_dt: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """Compute the KL divergence loss for the fine-tuning process.
    Args:
        int_u_u_dt: Tensor of shape (B,) representing the quadratic variation of the controlled term.
        ws: Tensor of shape (B,) representing the importance weights.
    Returns:
        Tensor representing the KL divergence loss.
    """

    # Integrate from t=1 to t=0 i.e., reverse time
    w_int_u_u_dt = ws * (int_u_u_dt - rloo_baseline(int_u_u_dt).detach())  # (B,)

    # Compute the KL divergence loss
    loss_kl = torch.mean(w_int_u_u_dt) / 2

    return loss_kl


def compute_kl_loss_from_int_dws(
    *, int_u_u_dt: torch.Tensor, int_dws: torch.Tensor
) -> torch.Tensor:
    """Compute the KL divergence loss for the fine-tuning process.
    Args:
        int_u_u_dt: Tensor of shape (B,) representing the quadratic variation of the controlled term.
        int_dws: Tensor of shape (B,) representing the integrated gradient of the importance weights.
    Returns:
        Tensor representing the KL divergence loss.
    """

    # Integrate from t=1 to t=0 i.e., reverse time
    w_int_u_u_dt = int_u_u_dt + (int_u_u_dt - rloo_baseline(int_u_u_dt)).detach() * int_dws  # (B,)

    # Compute the KL divergence loss
    loss_kl = torch.mean(w_int_u_u_dt) / 2

    return loss_kl
