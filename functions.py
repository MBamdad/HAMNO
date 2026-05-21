import torch
import torch.nn.functional as F
import numpy as np
import config

######
#####
# ============================================================
# AC3D + homogeneous Neumann BCs (DCT / cosine basis)
# ============================================================

import math
import torch
import config


# ------------------------------------------------------------
# Cached orthonormal DCT-II matrix
# ------------------------------------------------------------
_DCT_MAT_CACHE = {}


def _dct2_ortho_matrix(N, device, dtype):
    key = (N, str(device), str(dtype))
    if key in _DCT_MAT_CACHE:
        return _DCT_MAT_CACHE[key]

    n = torch.arange(N, device=device, dtype=torch.float64)
    k = torch.arange(N, device=device, dtype=torch.float64).unsqueeze(1)

    M = torch.cos((math.pi / N) * (n + 0.5) * k)
    M *= math.sqrt(2.0 / N)
    M[0, :] /= math.sqrt(2.0)

    M = M.to(dtype)
    _DCT_MAT_CACHE[key] = M
    return M


def _apply_linear_last(x, A):
    x_shape = x.shape
    x2 = x.reshape(-1, x_shape[-1])
    y2 = x2 @ A.T
    return y2.reshape(*x_shape[:-1], A.shape[0])


def _dct_axis(x, axis):
    M = _dct2_ortho_matrix(x.shape[axis], x.device, x.dtype)
    x_perm = x.movedim(axis, -1)
    y_perm = _apply_linear_last(x_perm, M)
    return y_perm.movedim(-1, axis)


def _idct_axis(xhat, axis):
    M = _dct2_ortho_matrix(xhat.shape[axis], xhat.device, xhat.dtype)
    xhat_perm = xhat.movedim(axis, -1)
    x_perm = _apply_linear_last(xhat_perm, M.T)
    return x_perm.movedim(-1, axis)


def dct3_neumann(u):
    out = _dct_axis(u, 1)
    out = _dct_axis(out, 2)
    out = _dct_axis(out, 3)
    return out


def idct3_neumann(uhat):
    out = _idct_axis(uhat, 1)
    out = _idct_axis(out, 2)
    out = _idct_axis(out, 3)
    return out


# ------------------------------------------------------------
# Neumann cosine-spectrum eigenvalues
# ------------------------------------------------------------
_K2_NEUMANN_CACHE = {}


def _neumann_k2_grid(nx, ny, nz, dx, device, dtype):
    key = (nx, ny, nz, float(dx), str(device), str(dtype))
    if key in _K2_NEUMANN_CACHE:
        return _K2_NEUMANN_CACHE[key]

    Lx, Ly, Lz = nx * dx, ny * dx, nz * dx

    px = torch.arange(nx, device=device, dtype=dtype)
    py = torch.arange(ny, device=device, dtype=dtype)
    pz = torch.arange(nz, device=device, dtype=dtype)

    k2x = (math.pi * px / Lx) ** 2
    k2y = (math.pi * py / Ly) ** 2
    k2z = (math.pi * pz / Lz) ** 2

    K2x, K2y, K2z = torch.meshgrid(k2x, k2y, k2z, indexing="ij")
    k2 = K2x + K2y + K2z

    _K2_NEUMANN_CACHE[key] = k2
    return k2


# ------------------------------------------------------------
# Core AC operators
# ------------------------------------------------------------
def laplacian_neumann_cosine_3d(u, dx):
    _, nx, ny, nz = u.shape
    k2 = _neumann_k2_grid(nx, ny, nz, dx, u.device, u.dtype)
    u_hat = dct3_neumann(u)
    lap_hat = -k2 * u_hat
    return idct3_neumann(lap_hat)


def pde_rhs_ac_neumann(u, dx, eps2):
    lap_u = laplacian_neumann_cosine_3d(u, dx)
    return lap_u - (1.0 / eps2) * (u**3 - u)


# ------------------------------------------------------------
# Teacher step
# ------------------------------------------------------------
def semi_implicit_step_AC_neumann(u_in, dt, dx, eps2):
    T_out = int(getattr(config, "T_OUT", 1))

    u_cur = u_in.squeeze(-1).float()
    _, nx, ny, nz = u_cur.shape
    k2 = _neumann_k2_grid(nx, ny, nz, dx, u_cur.device, u_cur.dtype)
    denom = 1.0 + dt * k2

    steps = []
    for _ in range(T_out):
        nl = u_cur**3 - u_cur
        u_hat = dct3_neumann(u_cur)
        nl_hat = dct3_neumann(nl)
        u_next_hat = (u_hat - (dt / eps2) * nl_hat) / denom
        u_cur = idct3_neumann(u_next_hat)
        steps.append(u_cur)

    return torch.stack(steps, dim=-1).to(u_in.dtype)


# ------------------------------------------------------------
# Temporal collocation
# ------------------------------------------------------------
def temporal_collocation_rule(
    rule="lobatto",
    device=None,
    dtype=torch.float32,
    gegenbauer_lam=1.0,
    jacobi_alpha=0.0,
    jacobi_beta=0.0,
):
    rule = rule.lower()

    if rule == "chebyshev":
        off = 1.0 / (2.0 * math.sqrt(2.0))
        taus, weights = [0.5 - off, 0.5 + off], [0.5, 0.5]

    elif rule == "legendre":
        off = 1.0 / (2.0 * math.sqrt(3.0))
        taus, weights = [0.5 - off, 0.5 + off], [0.5, 0.5]

    elif rule == "gegenbauer":
        lam = float(gegenbauer_lam)
        if lam <= -0.5:
            raise ValueError("gegenbauer_lam must satisfy lambda > -1/2")
        off = 1.0 / (2.0 * math.sqrt(2.0 * (lam + 1.0)))
        taus, weights = [0.5 - off, 0.5 + off], [0.5, 0.5]

    elif rule == "jacobi":
        a, b = float(jacobi_alpha), float(jacobi_beta)
        if a <= -1.0 or b <= -1.0:
            raise ValueError("jacobi_alpha and jacobi_beta must both be > -1")

        den_inner = (a*a + 2*a*b + 7*a + b*b + 7*b + 12.0)
        A = (a + b + 4.0) * den_inner
        B = (-a + b) * den_inner
        C = 2.0 * math.sqrt((a + 2.0) * (b + 2.0) * (a + b + 3.0)) * (a + b + 4.0)

        x1 = (B - C) / A
        x2 = (B + C) / A
        taus, weights = [(x1 + 1.0) * 0.5, (x2 + 1.0) * 0.5], [0.5, 0.5]

    elif rule == "lobatto":
        off = 1.0 / (2.0 * math.sqrt(5.0))
        taus, weights = [0.5 - off, 0.5 + off], [0.5, 0.5]

    elif rule == "lobatto3":
        taus, weights = [0.0, 0.5, 1.0], [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0]

    elif rule == "uniform":
        taus, weights = [0.25, 0.75], [0.5, 0.5]

    else:
        raise ValueError(f"Unknown collocation rule '{rule}'")

    return (
        torch.tensor(taus, device=device, dtype=dtype),
        torch.tensor(weights, device=device, dtype=dtype),
    )


def physics_collocation_tau_L2_AC_neumann(u_in, u_pred, tau, normalize=True):
    dt, dx, eps2 = config.DT, config.DX, config.EPS2

    u0 = u_in.squeeze(-1).float()
    up = u_pred.float()

    if up.shape[-1] == 1:
        up1 = up.squeeze(-1)
        ut = (up1 - u0) / dt
        u_tau = (1.0 - tau) * u0 + tau * up1
        rhs_tau = pde_rhs_ac_neumann(u_tau, dx, eps2)

        if normalize:
            s_t = ut.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
            s_r = rhs_tau.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
            R = ut / s_t - rhs_tau / s_r
        else:
            R = ut - rhs_tau

        return (R**2).mean().to(u_pred.dtype)

    B, Sx, Sy, Sz, T = up.shape
    u_all = torch.cat([u0.unsqueeze(-1), up], dim=-1)
    u_prev, u_next = u_all[..., :-1], u_all[..., 1:]

    ut = (u_next - u_prev) / dt
    u_tau = (1.0 - tau) * u_prev + tau * u_next

    u_tau_flat = u_tau.permute(0, 4, 1, 2, 3).reshape(B * T, Sx, Sy, Sz)
    rhs_tau_flat = pde_rhs_ac_neumann(u_tau_flat, dx, eps2)
    rhs_tau = rhs_tau_flat.view(B, T, Sx, Sy, Sz).permute(0, 2, 3, 4, 1)

    if normalize:
        s_t = ut.pow(2).mean((1, 2, 3, 4), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean((1, 2, 3, 4), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau

    return (R**2).mean().to(u_pred.dtype)


def physics_collocation_multi_rule_AC_neumann(
    u_in,
    u_pred,
    rules=("chebyshev", "jacobi"),
    weights=None,
    normalize=True,
    gegenbauer_lam=1.0,
    jacobi_alpha=0.0,
    jacobi_beta=0.0,
):
    if weights is None:
        weights = [1.0 / len(rules)] * len(rules)

    if len(weights) != len(rules):
        raise ValueError("weights must have the same length as rules")

    total_w = float(sum(weights))
    if total_w <= 0:
        raise ValueError("sum(weights) must be positive")

    loss = torch.zeros((), device=u_pred.device, dtype=u_pred.dtype)

    for rule_name, rule_weight in zip(rules, weights):
        taus, tau_weights = temporal_collocation_rule(
            rule=rule_name,
            device=u_pred.device,
            dtype=u_pred.dtype if u_pred.is_floating_point() else torch.float32,
            gegenbauer_lam=gegenbauer_lam,
            jacobi_alpha=jacobi_alpha,
            jacobi_beta=jacobi_beta,
        )

        loss_rule = torch.zeros((), device=u_pred.device, dtype=u_pred.dtype)
        for tau, tau_w in zip(taus, tau_weights):
            l_tau = physics_collocation_tau_L2_AC_neumann(
                u_in,
                u_pred,
                tau=float(tau),
                normalize=normalize,
            )
            loss_rule = loss_rule + tau_w * l_tau

        loss = loss + float(rule_weight) * loss_rule

    return loss / total_w


# ------------------------------------------------------------
# Energy-based losses
# ------------------------------------------------------------
def energy_density_AC_neumann(u, dx, eps2):
    lap_u = laplacian_neumann_cosine_3d(u, dx)
    grad_term = -0.5 * eps2 * (u * lap_u)
    pot_term = 0.25 * (u**2 - 1.0)**2
    return grad_term + pot_term


def total_free_energy_AC_neumann(u, dx, eps2):
    if u.dim() == 5:
        u = u[..., 0]
    u = u.float()
    return energy_density_AC_neumann(u, dx, eps2).mean(dim=(1, 2, 3))


def energy_dissipation_match_AC_neumann(
    u_in,
    u_pred,
    u_teacher,
    dx,
    eps2,
    relative=True,
    robust=True,
    margin=1e-3,
    increase_weight=0.5,
    eps=1e-8,
):
    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()
    ut = u_teacher.squeeze(-1).float()

    E0 = total_free_energy_AC_neumann(u0, dx, eps2)
    Ep = total_free_energy_AC_neumann(up, dx, eps2)
    Et = total_free_energy_AC_neumann(ut, dx, eps2)

    dE_pred = E0 - Ep
    dE_teach = E0 - Et

    if relative:
        scale = dE_teach.abs().detach() + 0.05 * E0.abs().detach() + 1e-6
        mismatch = (dE_pred - dE_teach) / scale
        inc = torch.relu(Ep - E0) / scale
    else:
        mismatch = dE_pred - dE_teach
        inc = torch.relu(Ep - E0)

    dev = torch.relu(mismatch.abs() - margin)

    if robust:
        loss_match = torch.sqrt(dev * dev + eps)
        loss_inc = torch.sqrt(inc * inc + eps)
    else:
        loss_match = dev * dev
        loss_inc = inc * inc

    return (loss_match + increase_weight * loss_inc).mean().to(u_pred.dtype)


# ------------------------------------------------------------
# Energy-metric scheme loss
# ------------------------------------------------------------
def scheme_loss_AC_neumann_energy_metric(
    u_pred,
    u_teacher,
    dx,
    eps2,
    alpha=1.0,
    normalize=True,
):
    up = u_pred.unsqueeze(-1) if u_pred.dim() == 4 else u_pred
    ut = u_teacher.unsqueeze(-1) if u_teacher.dim() == 4 else u_teacher

    if up.shape[-1] != ut.shape[-1]:
        raise ValueError("u_pred and u_teacher must have matching last dimension")

    B, Sx, Sy, Sz, T = up.shape
    err = (up - ut).permute(0, 4, 1, 2, 3).reshape(B * T, Sx, Sy, Sz).float()

    e_hat = dct3_neumann(err)
    k2 = _neumann_k2_grid(Sx, Sy, Sz, dx, err.device, err.dtype)
    weight = 1.0 + alpha * eps2 * k2

    val = (weight * (e_hat ** 2)).mean()

    if normalize:
        val = val / (weight.mean().detach() + 1e-8)

    return val.to(u_pred.dtype)

######3
###### CH

# ============================================================
# CH3D + homogeneous Neumann BCs (DCT / cosine basis)
# MATLAB-consistent with the CH generator
# ============================================================

def _remove_spatial_mean_4d(u):
    """
    Remove per-sample spatial mean from a tensor of shape (B,S,S,S).
    """
    return u - u.mean(dim=(1, 2, 3), keepdim=True)


def _remove_spatial_mean_lasttime(u):
    """
    Remove per-sample spatial mean from a tensor of shape:
      - (B,S,S,S)
      - (B,S,S,S,1)
      - (B,S,S,S,T)
    """
    if u.dim() == 4:
        return _remove_spatial_mean_4d(u)
    if u.dim() == 5:
        return u - u.mean(dim=(1, 2, 3), keepdim=True)
    raise ValueError("Expected shape (B,S,S,S) or (B,S,S,S,T)")


def biharmonic_neumann_cosine_3d(u, dx):
    """
    Spectral biharmonic under homogeneous Neumann BC:
        Δ²u <-> k^4 û
    """
    return laplacian_neumann_cosine_3d(
        laplacian_neumann_cosine_3d(u, dx), dx
    )


def chemical_potential_CH_neumann(u, dx, eps2):
    """
    CH chemical potential:
        mu = -eps2 * Δu + (u^3 - u)
    """
    lap_u = laplacian_neumann_cosine_3d(u, dx)
    return -eps2 * lap_u + (u**3 - u)


def pde_rhs_ch_neumann(u, dx, eps2):
    """
    MATLAB-consistent CH RHS with Neumann BC:

        u_t = 2Δu - eps2 Δ²u + Δ(u^3 - 3u)

    which is algebraically equivalent to

        u_t = Δ(-eps2 Δu + u^3 - u).
    """
    lap_u = laplacian_neumann_cosine_3d(u, dx)
    bih_u = laplacian_neumann_cosine_3d(lap_u, dx)

    chem_split = u**3 - 3.0 * u
    lap_chem_split = laplacian_neumann_cosine_3d(chem_split, dx)

    return 2.0 * lap_u - eps2 * bih_u + lap_chem_split


def semi_implicit_step_CH_neumann(u_in, dt, dx, eps2):
    """
    CH3D semi-implicit Neumann teacher step matching MATLAB exactly.

        nl_hat   = DCT(u^3 - 3u)
        u_hat    = DCT(u)
        rhs_hat  = u_hat - dt * k2 * nl_hat
        next_hat = rhs_hat / (1 + dt*(2k2 + eps2*k2^2))
    """
    T_out = int(getattr(config, "T_OUT", 1))

    u_cur = u_in.squeeze(-1).float()
    _, nx, ny, nz = u_cur.shape

    k2 = _neumann_k2_grid(nx, ny, nz, dx, u_cur.device, u_cur.dtype)
    denom = 1.0 + dt * (2.0 * k2 + eps2 * (k2 ** 2))

    steps = []
    for _ in range(T_out):
        nl = u_cur**3 - 3.0 * u_cur

        u_hat = dct3_neumann(u_cur)
        nl_hat = dct3_neumann(nl)

        rhs_hat = u_hat - dt * k2 * nl_hat
        u_next_hat = rhs_hat / (denom + 1e-12)
        u_cur = idct3_neumann(u_next_hat)

        steps.append(u_cur)

    return torch.stack(steps, dim=-1).to(u_in.dtype)



def physics_collocation_tau_L2_CH_neumann(u_in, u_pred, tau, normalize=True):
    dt, dx, eps2 = config.DT, config.DX, config.EPS2

    u0 = u_in.squeeze(-1).float()
    up = u_pred.float()

    if up.shape[-1] == 1:
        up1 = up.squeeze(-1)

        ut = (up1 - u0) / dt
        u_tau = (1.0 - tau) * u0 + tau * up1
        rhs_tau = pde_rhs_ch_neumann(u_tau, dx, eps2)

        R = ut - rhs_tau
        R = _remove_spatial_mean_4d(R)

        if normalize:
            s_ut = ut.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach()
            s_rhs = rhs_tau.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach()
            s = 0.5 * (s_ut + s_rhs) + 1e-8
            R = R / s

        return (R**2).mean().to(u_pred.dtype)

    B, Sx, Sy, Sz, T = up.shape

    u_all = torch.cat([u0.unsqueeze(-1), up], dim=-1)
    u_prev = u_all[..., :-1]
    u_next = u_all[..., 1:]

    ut = (u_next - u_prev) / dt
    u_tau = (1.0 - tau) * u_prev + tau * u_next

    u_tau_flat = u_tau.permute(0, 4, 1, 2, 3).reshape(B * T, Sx, Sy, Sz)
    rhs_tau_flat = pde_rhs_ch_neumann(u_tau_flat, dx, eps2)
    rhs_tau = rhs_tau_flat.view(B, T, Sx, Sy, Sz).permute(0, 2, 3, 4, 1)

    R = ut - rhs_tau
    R = R - R.mean(dim=(1, 2, 3), keepdim=True)

    if normalize:
        s_ut = ut.pow(2).mean((1, 2, 3, 4), keepdim=True).sqrt().detach()
        s_rhs = rhs_tau.pow(2).mean((1, 2, 3, 4), keepdim=True).sqrt().detach()
        s = 0.5 * (s_ut + s_rhs) + 1e-8
        R = R / s

    return (R**2).mean().to(u_pred.dtype)
def physics_collocation_multi_rule_CH_neumann(
    u_in,
    u_pred,
    rules=("chebyshev", "jacobi"),
    weights=None,
    normalize=True,
    gegenbauer_lam=1.0,
    jacobi_alpha=0.0,
    jacobi_beta=0.0,
):
    """
    Multi-rule temporal collocation for CH + Neumann BC.
    Same structure as AC.
    """
    if weights is None:
        weights = [1.0 / len(rules)] * len(rules)

    if len(weights) != len(rules):
        raise ValueError("weights must have the same length as rules")

    total_w = float(sum(weights))
    if total_w <= 0:
        raise ValueError("sum(weights) must be positive")

    loss = torch.zeros((), device=u_pred.device, dtype=u_pred.dtype)

    for rule_name, rule_weight in zip(rules, weights):
        taus, tau_weights = temporal_collocation_rule(
            rule=rule_name,
            device=u_pred.device,
            dtype=u_pred.dtype if u_pred.is_floating_point() else torch.float32,
            gegenbauer_lam=gegenbauer_lam,
            jacobi_alpha=jacobi_alpha,
            jacobi_beta=jacobi_beta,
        )

        loss_rule = torch.zeros((), device=u_pred.device, dtype=u_pred.dtype)

        for tau, tau_w in zip(taus, tau_weights):
            l_tau = physics_collocation_tau_L2_CH_neumann(
                u_in,
                u_pred,
                tau=float(tau),
                normalize=normalize,
            )
            loss_rule = loss_rule + tau_w * l_tau

        loss = loss + float(rule_weight) * loss_rule

    return loss / total_w


def scheme_loss_CH_neumann_energy_metric(
    u_pred,
    u_teacher,
    dx,
    eps2,
    alpha=4.0,
    normalize=True,
):
    """
    Mean-free CH teacher mismatch in cosine energy metric.

    For CH, the k=0 mode is handled separately by mass consistency.
    So here we compare only the mean-free component.
    """
    up = u_pred.unsqueeze(-1) if u_pred.dim() == 4 else u_pred
    ut = u_teacher.unsqueeze(-1) if u_teacher.dim() == 4 else u_teacher

    if up.shape[-1] != ut.shape[-1]:
        raise ValueError("u_pred and u_teacher must have matching last dimension")

    e = _remove_spatial_mean_lasttime(up - ut)

    B, Sx, Sy, Sz, T = e.shape
    e = e.permute(0, 4, 1, 2, 3).reshape(B * T, Sx, Sy, Sz).float()
    e_hat = dct3_neumann(e)

    k2 = _neumann_k2_grid(Sx, Sy, Sz, dx, e.device, e.dtype)
    weight = 1.0 + alpha * eps2 * k2

    val = (weight * (e_hat ** 2)).mean()

    if normalize:
        val = val / (weight.mean().detach() + 1e-8)

    return val.to(u_pred.dtype)


def total_free_energy_CH_neumann(u, dx, eps2):
    """
    CH free energy:
        F[u] = ∫ [ eps2/2 |grad u|^2 + 1/4 (u^2-1)^2 ] dx
    """
    if u.dim() == 5:
        u = u[..., 0]
    u = u.float()

    e = energy_density_AC_neumann(u, dx, eps2)
    return e.mean(dim=(1, 2, 3))


def energy_dissipation_match_CH_neumann(
    u_in,
    u_pred,
    u_teacher,
    dx,
    eps2,
    relative=True,
    robust=True,
    margin=1e-3,
    increase_weight=0.5,
    mass_weight=0.25,
    eps=1e-8,
):
    """
    Teacher-referenced CH energy dissipation + safe mass consistency.

    Important:
    - energy part matches teacher dissipation
    - mass term compares to the input mean, not normalized by tiny mean values
    """
    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()
    ut = u_teacher.squeeze(-1).float()

    F0 = total_free_energy_CH_neumann(u0, dx, eps2)
    Fp = total_free_energy_CH_neumann(up, dx, eps2)
    Ft = total_free_energy_CH_neumann(ut, dx, eps2)

    dF_pred = F0 - Fp
    dF_teach = F0 - Ft

    if relative:
        scale = dF_teach.abs().detach() + 0.05 * F0.abs().detach() + 1e-6
        mismatch = (dF_pred - dF_teach) / scale
        inc = torch.relu(Fp - F0) / scale
    else:
        mismatch = dF_pred - dF_teach
        inc = torch.relu(Fp - F0)

    dev = torch.relu(mismatch.abs() - margin)

    if robust:
        loss_match = torch.sqrt(dev * dev + eps)
        loss_inc = torch.sqrt(inc * inc + eps)
    else:
        loss_match = dev * dev
        loss_inc = inc * inc

    # safe mass consistency
    m0 = u0.mean(dim=(1, 2, 3))
    mp = up.mean(dim=(1, 2, 3))
    mt = ut.mean(dim=(1, 2, 3))

    # compare both to the conserved input mass
    mass_err = (mp - m0) ** 2 + 0.25 * (mt - m0) ** 2
    loss_mass = torch.sqrt(mass_err + eps) if robust else mass_err

    return (
        loss_match
        + increase_weight * loss_inc
        + mass_weight * loss_mass
    ).mean().to(u_pred.dtype)


def scheme_loss_CH_neumann_Hminus1(
    u_pred,
    u_teacher,
    dx,
    kappa=1e-6,
    normalize=True,
):
    """
    Mean-free CH teacher mismatch in cosine H^{-1}-type metric.
    This is more natural than H1 for Cahn-Hilliard dynamics.
    """
    up = u_pred.unsqueeze(-1) if u_pred.dim() == 4 else u_pred
    ut = u_teacher.unsqueeze(-1) if u_teacher.dim() == 4 else u_teacher

    if up.shape[-1] != ut.shape[-1]:
        raise ValueError("u_pred and u_teacher must have matching last dimension")

    e = _remove_spatial_mean_lasttime(up - ut)

    B, Sx, Sy, Sz, T = e.shape
    e = e.permute(0, 4, 1, 2, 3).reshape(B * T, Sx, Sy, Sz).float()
    e_hat = dct3_neumann(e)

    k2 = _neumann_k2_grid(Sx, Sy, Sz, dx, e.device, e.dtype)

    # zero mode already removed, but keep a safe regularization
    weight = 1.0 / (k2 + kappa)

    val = (weight * (e_hat ** 2)).mean()

    if normalize:
        val = val / (weight.mean().detach() + 1e-8)

    return val.to(u_pred.dtype)


def physics_collocation_tau_L2_CH_neumann_scaled(u_in, u_pred, tau, normalize=True):
    dt, dx, eps2 = config.DT, config.DX, config.EPS2

    u0 = u_in.squeeze(-1).float()
    up = u_pred.float()

    if up.shape[-1] == 1:
        up1 = up.squeeze(-1)
        ut = (up1 - u0) / dt
        u_tau = (1.0 - tau) * u0 + tau * up1
        rhs_tau = pde_rhs_ch_neumann(u_tau, dx, eps2)

        R = _remove_spatial_mean_4d(ut - rhs_tau)

        if normalize:
            s_ut = ut.pow(2).mean((1,2,3), keepdim=True).sqrt().detach()
            s_rhs = rhs_tau.pow(2).mean((1,2,3), keepdim=True).sqrt().detach()
            s = 0.5 * (s_ut + s_rhs) + 1e-8
            R = R / s

        return (R**2).mean().to(u_pred.dtype)

    B, Sx, Sy, Sz, T = up.shape
    u_all = torch.cat([u0.unsqueeze(-1), up], dim=-1)
    u_prev, u_next = u_all[..., :-1], u_all[..., 1:]

    ut = (u_next - u_prev) / dt
    u_tau = (1.0 - tau) * u_prev + tau * u_next

    u_tau_flat = u_tau.permute(0,4,1,2,3).reshape(B*T, Sx, Sy, Sz)
    rhs_tau_flat = pde_rhs_ch_neumann(u_tau_flat, dx, eps2)
    rhs_tau = rhs_tau_flat.view(B, T, Sx, Sy, Sz).permute(0,2,3,4,1)

    R = ut - rhs_tau
    R = R - R.mean(dim=(1,2,3), keepdim=True)

    if normalize:
        s_ut = ut.pow(2).mean((1,2,3,4), keepdim=True).sqrt().detach()
        s_rhs = rhs_tau.pow(2).mean((1,2,3,4), keepdim=True).sqrt().detach()
        s = 0.5 * (s_ut + s_rhs) + 1e-8
        R = R / s

    return (R**2).mean().to(u_pred.dtype)

###############
###############
def ac_residual(u_prev, u_next, dx, dt, eps2):

    ut = (u_next - u_prev) / dt
    rhs = pde_rhs_ac_neumann(u_next.squeeze(-1), dx, eps2)

    return ut - rhs.unsqueeze(-1)

import torch



def ac_rollout_residual_neumann(u_hist_last, u_pred, dt, dx, eps2):
    """
    u_hist_last: (B,S,S,S,1)
    u_pred     : (B,S,S,S,T)
    returns    : (B,S,S,S,T)
    """
    T = u_pred.shape[-1]
    res_list = []
    u_prev = u_hist_last

    for t in range(T):
        u_next = u_pred[..., t:t+1]
        ut = (u_next - u_prev) / dt
        rhs = pde_rhs_ac_neumann(u_next.squeeze(-1), dx, eps2).unsqueeze(-1)
        res_list.append(ut - rhs)
        u_prev = u_next

    return torch.cat(res_list, dim=-1)


def ac_incremental_energy_step(u_prev, u_next, dt, dx, eps2):
    """
    Incremental energy for one Allen–Cahn step:

        J(u_next ; u_prev)
        = (1/(2 dt)) ||u_next - u_prev||_L2^2 + E(u_next)

    u_prev, u_next: (B,S,S,S,1)
    returns: (B,)
    """
    if u_prev.dim() != 5 or u_next.dim() != 5:
        raise ValueError("u_prev and u_next must have shape (B,S,S,S,1)")

    move = 0.5 / dt * ((u_next - u_prev) ** 2).mean(dim=(1, 2, 3, 4))
    energy = total_free_energy_AC_neumann(u_next, dx, eps2)  # (B,)
    return move + energy


def ac_incremental_energy_loss_neumann(
    u_hist_last,
    u_pred,
    dt,
    dx,
    eps2,
    relative=True,
    robust=True,
    teacher_weight=0.25,
    eps=1e-8,
):
    """
    Teacher-referenced incremental energy loss for AC rollout.

    Idea:
      - predicted step should have small incremental energy
      - and should not be worse than the semi-implicit physical scaffold

    u_hist_last: (B,S,S,S,1)
    u_pred     : (B,S,S,S,T)

    returns: scalar
    """
    if u_hist_last.dim() != 5 or u_hist_last.shape[-1] != 1:
        raise ValueError("u_hist_last must have shape (B,S,S,S,1)")
    if u_pred.dim() != 5:
        raise ValueError("u_pred must have shape (B,S,S,S,T)")

    T = u_pred.shape[-1]
    u_prev = u_hist_last
    losses = []

    for t in range(T):
        u_next = u_pred[..., t:t+1]  # (B,S,S,S,1)

        # predicted incremental energy
        J_pred = ac_incremental_energy_step(u_prev, u_next, dt, dx, eps2)  # (B,)

        # semi-implicit teacher step from the same previous state
        u_teacher = semi_implicit_step_AC_neumann_single(u_prev, dt, dx, eps2)
        J_teach = ac_incremental_energy_step(u_prev, u_teacher, dt, dx, eps2)  # (B,)

        if relative:
            scale = J_teach.abs().detach() + 1e-6
            worse_than_teacher = (J_pred - J_teach) / scale
        else:
            worse_than_teacher = J_pred - J_teach

        # only penalize if predicted step has larger incremental energy
        worse_than_teacher = torch.relu(worse_than_teacher)

        if robust:
            step_loss = torch.sqrt(worse_than_teacher * worse_than_teacher + eps)
        else:
            step_loss = worse_than_teacher * worse_than_teacher

        # tiny absolute anchor so J does not drift even if teacher is imperfect
        if relative:
            abs_anchor = J_pred / (J_teach.abs().detach() + 1e-6)
        else:
            abs_anchor = J_pred

        losses.append(step_loss + teacher_weight * abs_anchor)
        u_prev = u_next

    return torch.stack(losses, dim=0).mean().to(u_pred.dtype)

def ac_incremental_energy_weak_integral_loss_neumann(
    u_hist_last,
    u_pred,
    dt,
    dx,
    eps2,
    tau=0.5,
    n_modes=8,
    normalize=True,
    include_incremental_energy=True,
    energy_weight=0.25,
):
    """
    Hybrid variational + weak/integral-form loss for AC with homogeneous Neumann BC.

    Uses:
      1) incremental energy functional
      2) weak-form / integral residual in cosine test space

    Parameters
    ----------
    u_hist_last : (B,S,S,S,1)
    u_pred      : (B,S,S,S,T)
    dt, dx, eps2: physical parameters
    tau         : collocation point in [0,1]
    n_modes     : number of low cosine modes per axis for weak-form testing
    normalize   : normalize weak residual modewise
    include_incremental_energy : whether to include J(u^{n+1};u^n)
    energy_weight : weight on absolute incremental energy anchor

    Returns
    -------
    scalar tensor
    """
    if u_hist_last.dim() != 5 or u_hist_last.shape[-1] != 1:
        raise ValueError("u_hist_last must have shape (B,S,S,S,1)")
    if u_pred.dim() != 5:
        raise ValueError("u_pred must have shape (B,S,S,S,T)")

    B, Sx, Sy, Sz, T = u_pred.shape
    u_prev = u_hist_last
    losses = []

    # low-mode Neumann spectrum
    k2 = _neumann_k2_grid(Sx, Sy, Sz, dx, u_pred.device, u_pred.dtype)
    mx = min(n_modes, Sx)
    my = min(n_modes, Sy)
    mz = min(n_modes, Sz)

    k2_low = k2[:mx, :my, :mz].reshape(1, -1)  # (1, M)

    for t in range(T):
        u_next = u_pred[..., t:t+1]                          # (B,S,S,S,1)

        # -----------------------------
        # weak / integral form at u_tau
        # -----------------------------
        ut = (u_next - u_prev) / dt                          # (B,S,S,S,1)
        u_tau = (1.0 - tau) * u_prev + tau * u_next         # (B,S,S,S,1)

        ut_hat = dct3_neumann(ut.squeeze(-1))               # (B,S,S,S)
        u_tau_hat = dct3_neumann(u_tau.squeeze(-1))         # (B,S,S,S)

        nonlin = (u_tau.squeeze(-1) ** 3 - u_tau.squeeze(-1)) / eps2
        nonlin_hat = dct3_neumann(nonlin)                   # (B,S,S,S)

        ut_low = ut_hat[:, :mx, :my, :mz].reshape(B, -1)
        u_low = u_tau_hat[:, :mx, :my, :mz].reshape(B, -1)
        n_low = nonlin_hat[:, :mx, :my, :mz].reshape(B, -1)

        # weak residual moments:
        # <u_t,phi_k> + k^2 <u,phi_k> + <(u^3-u)/eps2,phi_k>
        Rw = ut_low + k2_low * u_low + n_low

        if normalize:
            s_ut = ut_low.pow(2).mean(dim=1, keepdim=True).sqrt().detach() + 1e-8
            s_u  = (k2_low * u_low).pow(2).mean(dim=1, keepdim=True).sqrt().detach() + 1e-8
            s_n  = n_low.pow(2).mean(dim=1, keepdim=True).sqrt().detach() + 1e-8
            scale = 0.5 * (s_ut + s_u + s_n)
            Rw = Rw / scale

        loss_weak = (Rw ** 2).mean()

        # -----------------------------
        # incremental energy functional
        # -----------------------------
        if include_incremental_energy:
            J_pred = ac_incremental_energy_step(u_prev, u_next, dt, dx, eps2)  # (B,)
            J_abs = J_pred.mean()
            step_loss = loss_weak + energy_weight * J_abs
        else:
            step_loss = loss_weak

        losses.append(step_loss)
        u_prev = u_next

    return torch.stack(losses).mean().to(u_pred.dtype)

def ac_incremental_energy_step(u_prev, u_next, dt, dx, eps2):
    """
    J(u_next ; u_prev) = 1/(2dt) ||u_next-u_prev||^2 + E(u_next)
    """
    move = 0.5 / dt * ((u_next - u_prev) ** 2).mean(dim=(1, 2, 3, 4))
    energy = total_free_energy_AC_neumann(u_next, dx, eps2)
    return move + energy

##########
########

# ============================================================
# AC3D collocation: keep ONLY original random->sobol path
# ============================================================

import numpy as np
import torch
import config


def _normalize_tau_weights(taus, weights, device=None, dtype=torch.float32):
    taus = torch.as_tensor(taus, device=device, dtype=dtype).flatten()
    weights = torch.as_tensor(weights, device=device, dtype=dtype).flatten()

    taus = taus.clamp(0.0, 1.0)
    weights = torch.clamp(weights, min=0.0)

    s = weights.sum()
    if s <= 0:
        weights = torch.ones_like(taus) / max(1, taus.numel())
    else:
        weights = weights / s

    return taus, weights


def _safe_uniform_weights(n):
    return np.ones(n, dtype=np.float64) / float(n)


def collocation_random_ac(n_pts=3, seed=42):
    """
    Original best-performing random collocation path:
    random family -> sobol subfamily only.
    """
    n_pts = max(1, int(n_pts))

    eng = torch.quasirandom.SobolEngine(
        dimension=1,
        scramble=True,
        seed=int(seed),
    )
    taus = eng.draw(n_pts).squeeze(-1).cpu().numpy()
    taus = np.sort(taus)

    weights = _safe_uniform_weights(n_pts)
    return taus, weights


def physics_collocation_points_AC_neumann(
    u_in,
    u_pred,
    taus,
    weights=None,
    normalize=True,
):
    if weights is None:
        weights = torch.ones_like(
            torch.as_tensor(
                taus,
                device=u_pred.device,
                dtype=u_pred.dtype if u_pred.is_floating_point() else torch.float32,
            )
        )

    taus, weights = _normalize_tau_weights(
        taus,
        weights,
        device=u_pred.device,
        dtype=u_pred.dtype if u_pred.is_floating_point() else torch.float32,
    )

    loss = torch.zeros((), device=u_pred.device, dtype=u_pred.dtype)

    for tau, w in zip(taus, weights):
        l_tau = physics_collocation_tau_L2_AC_neumann(
            u_in,
            u_pred,
            tau=float(tau),
            normalize=normalize,
        )
        loss = loss + w * l_tau

    return loss


def physics_collocation_random_AC_neumann(
    u_in,
    u_pred,
    n_pts=3,
    normalize=True,
    seed=None,
    return_points=False,
):
    """
    Clean wrapper, but behavior matches the original sobol path.
    """
    if seed is None:
        seed = int(getattr(config, "SEED", 42))

    taus, weights = collocation_random_ac(
        n_pts=n_pts,
        seed=seed,
    )

    taus, weights = _normalize_tau_weights(
        taus,
        weights,
        device=u_pred.device,
        dtype=u_pred.dtype if u_pred.is_floating_point() else torch.float32,
    )

    loss = physics_collocation_points_AC_neumann(
        u_in=u_in,
        u_pred=u_pred,
        taus=taus,
        weights=weights,
        normalize=normalize,
    )

    if return_points:
        return loss, taus.detach(), weights.detach()

    return loss


def physics_collocation_points_CH_neumann(
    u_in,
    u_pred,
    taus,
    weights=None,
    normalize=True,
):
    """
    Evaluate CH collocation loss at specified tau points and average with weights.
    """
    if weights is None:
        weights = torch.ones_like(
            torch.as_tensor(
                taus,
                device=u_pred.device,
                dtype=u_pred.dtype if u_pred.is_floating_point() else torch.float32,
            )
        )

    taus, weights = _normalize_tau_weights(
        taus,
        weights,
        device=u_pred.device,
        dtype=u_pred.dtype if u_pred.is_floating_point() else torch.float32,
    )

    loss = torch.zeros((), device=u_pred.device, dtype=u_pred.dtype)

    for tau, w in zip(taus, weights):
        l_tau = physics_collocation_tau_L2_CH_neumann(
            u_in,
            u_pred,
            tau=float(tau),
            normalize=normalize,
        )
        loss = loss + w * l_tau

    return loss


def physics_collocation_random_CH_neumann(
    u_in,
    u_pred,
    n_pts=3,
    normalize=True,
    seed=None,
    return_points=False,
):
    """
    CH version of the AC Sobol/random collocation wrapper.
    """
    if seed is None:
        seed = int(getattr(config, "SEED", 42))

    # reuse the same Sobol tau generator used for AC
    taus, weights = collocation_random_ac(
        n_pts=n_pts,
        seed=seed,
    )

    taus, weights = _normalize_tau_weights(
        taus,
        weights,
        device=u_pred.device,
        dtype=u_pred.dtype if u_pred.is_floating_point() else torch.float32,
    )

    loss = physics_collocation_points_CH_neumann(
        u_in=u_in,
        u_pred=u_pred,
        taus=taus,
        weights=weights,
        normalize=normalize,
    )

    if return_points:
        return loss, taus.detach(), weights.detach()

    return loss



def ac_energy_weighted_rollout_residual_neumann(
    u_hist_last,
    u_pred,
    dt,
    dx,
    eps2,
    detach_weight=True,
    normalize_weight=True,
    weight_floor=0.5,
    weight_cap=3.0,
):
    """
    Energy-density-weighted AC rollout residual.

    Idea
    ----
    Use local AC free-energy density to weight the PDE residual more strongly
    near interfaces / high-energy regions.

    This uses energy in a physically meaningful way:
      - not as a competing scalar loss
      - but as a spatial importance map for enforcing the PDE

    Inputs
    ------
    u_hist_last : (B,S,S,S,1)
    u_pred      : (B,S,S,S,T)

    Returns
    -------
    scalar loss
    """
    if u_hist_last.dim() != 5 or u_hist_last.shape[-1] != 1:
        raise ValueError("u_hist_last must have shape (B,S,S,S,1)")
    if u_pred.dim() != 5:
        raise ValueError("u_pred must have shape (B,S,S,S,T)")

    T = u_pred.shape[-1]
    u_prev = u_hist_last
    losses = []

    for t in range(T):
        u_next = u_pred[..., t:t+1]  # (B,S,S,S,1)

        ut = (u_next - u_prev) / dt
        rhs = pde_rhs_ac_neumann(u_next.squeeze(-1), dx, eps2).unsqueeze(-1)
        R = ut - rhs   # (B,S,S,S,1)

        # local AC energy density at predicted state
        e = energy_density_AC_neumann(u_next.squeeze(-1), dx, eps2).unsqueeze(-1)  # (B,S,S,S,1)

        # make weights positive and normalized
        w = e.abs()
        if detach_weight:
            w = w.detach()

        if normalize_weight:
            w_mean = w.mean(dim=(1, 2, 3, 4), keepdim=True).detach() + 1e-8
            w = w / w_mean

        # clamp so weighting helps but does not explode
        w = w.clamp(min=weight_floor, max=weight_cap)

        losses.append((w * (R ** 2)).mean())

        u_prev = u_next

    return torch.stack(losses).mean().to(u_pred.dtype)



def inverse_neg_laplacian_neumann_meanfree_3d(f, dx):
    """
    Solve
        (-Delta) v = f
    with homogeneous Neumann BC and zero-mean convention.

    f: (B,S,S,S), not necessarily mean-free; mean is removed internally.
    returns v: (B,S,S,S)
    """
    f = f - f.mean(dim=(1, 2, 3), keepdim=True)

    B, Sx, Sy, Sz = f.shape
    k2 = _neumann_k2_grid(Sx, Sy, Sz, dx, f.device, f.dtype)

    f_hat = dct3_neumann(f)
    v_hat = torch.zeros_like(f_hat)

    mask = k2 > 0
    v_hat[:, mask] = f_hat[:, mask] / k2[mask]

    # zero mode fixed to zero
    v_hat[:, 0, 0, 0] = 0.0

    v = idct3_neumann(v_hat)
    return v


def ch_rollout_residual_neumann_precond(
    u_hist_last,
    u_pred,
    dt,
    dx,
    eps2,
    tau=0.5,
    normalize=True,
):
    """
    Pure CH rollout residual using the natural H^{-1}-type preconditioning:

        u_t = Delta(mu)
        mu  = -eps2 Delta(u) + (u^3 - u)

    Apply (-Delta)^{-1} to get:
        (-Delta)^{-1} u_t + mu = 0   (mean-free form)

    We evaluate mu at u_tau = (1-tau)u_prev + tau u_next.

    u_hist_last: (B,S,S,S,1)
    u_pred     : (B,S,S,S,T)

    returns: residual tensor (B,S,S,S,T)
    """
    if u_hist_last.dim() != 5 or u_hist_last.shape[-1] != 1:
        raise ValueError("u_hist_last must have shape (B,S,S,S,1)")
    if u_pred.dim() != 5:
        raise ValueError("u_pred must have shape (B,S,S,S,T)")

    T = u_pred.shape[-1]
    res_list = []
    u_prev = u_hist_last

    for t in range(T):
        u_next = u_pred[..., t:t+1]                         # (B,S,S,S,1)

        ut = ((u_next - u_prev) / dt).squeeze(-1).float()   # (B,S,S,S)

        # midpoint / collocation state
        u_tau = ((1.0 - tau) * u_prev + tau * u_next).squeeze(-1).float()

        # chemical potential at midpoint
        mu_tau = chemical_potential_CH_neumann(u_tau, dx, eps2)
        mu_tau = mu_tau - mu_tau.mean(dim=(1, 2, 3), keepdim=True)

        # eta = (-Delta)^(-1) u_t
        eta = inverse_neg_laplacian_neumann_meanfree_3d(ut, dx)

        # preconditioned CH residual:
        # eta + mu = 0
        R = eta + mu_tau

        if normalize:
            s_eta = eta.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
            s_mu  = mu_tau.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
            R = R / (0.5 * (s_eta + s_mu))

        res_list.append(R.unsqueeze(-1))
        u_prev = u_next

    return torch.cat(res_list, dim=-1).to(u_pred.dtype)


### Analytical integral for Weak form

# ============================================================
# AC3D weak-form loss with Q1 hexahedral rectangular elements
# VINO-style analytical integration
# No Gauss quadrature, no centroid approximation
# ============================================================

import torch
import torch.nn.functional as F

_HEX_Q1_ANALYTIC_CACHE = {}


def _cube_to_hex8(u):
    """
    Convert grid nodal values to Q1 hexahedral element nodal values.

    Input:
        u: (B,Sx,Sy,Sz)

    Output:
        Ue: (B,Sx-1,Sy-1,Sz-1,8)

    Node ordering:
        0: v000
        1: v100
        2: v010
        3: v110
        4: v001
        5: v101
        6: v011
        7: v111
    """
    if u.dim() != 4:
        raise ValueError("u must have shape (B,Sx,Sy,Sz)")

    v000 = u[:, :-1, :-1, :-1]
    v100 = u[:,  1:, :-1, :-1]
    v010 = u[:, :-1,  1:, :-1]
    v110 = u[:,  1:,  1:, :-1]
    v001 = u[:, :-1, :-1,  1:]
    v101 = u[:,  1:, :-1,  1:]
    v011 = u[:, :-1,  1:,  1:]
    v111 = u[:,  1:,  1:,  1:]

    return torch.stack(
        [v000, v100, v010, v110, v001, v101, v011, v111],
        dim=-1,
    )


def _hex_q1_analytic_data(dx, device, dtype):
    """
    Analytical Q1 hexahedral element matrices.

    Returns
    -------
    M : (8,8)
        Exact mass matrix:
            M_ij = ∫_K N_i N_j dV

    K : (8,8)
        Exact stiffness matrix:
            K_ij = ∫_K grad(N_j) · grad(N_i) dV

    T4 : (8,8,8,8)
        Exact nonlinear tensor:
            T4[i,p,q,r] = ∫_K N_i N_p N_q N_r dV
    """
    key = (float(dx), str(device), str(dtype))
    if key in _HEX_Q1_ANALYTIC_CACHE:
        return _HEX_Q1_ANALYTIC_CACHE[key]

    h = torch.tensor(float(dx), device=device, dtype=dtype)

    # ------------------------------------------------------------
    # 1D linear element matrices on [0,h]
    # ------------------------------------------------------------
    M1 = (h / 6.0) * torch.tensor(
        [[2.0, 1.0],
         [1.0, 2.0]],
        device=device,
        dtype=dtype,
    )

    K1 = (1.0 / h) * torch.tensor(
        [[ 1.0, -1.0],
         [-1.0,  1.0]],
        device=device,
        dtype=dtype,
    )

    # ------------------------------------------------------------
    # Tensor-product 3D mass and stiffness matrices
    # Node ordering: x-fast, then y, then z
    # ------------------------------------------------------------
    M = torch.kron(torch.kron(M1, M1), M1)

    Kx = torch.kron(torch.kron(K1, M1), M1)
    Ky = torch.kron(torch.kron(M1, K1), M1)
    Kz = torch.kron(torch.kron(M1, M1), K1)

    K = Kx + Ky + Kz

    # ------------------------------------------------------------
    # Exact 1D fourth-order tensor:
    #
    # G[a,p,q,r] = ∫_0^h l_a l_p l_q l_r dx
    #
    # l_0 = 1-t, l_1 = t, t in [0,1]
    #
    # ∫_0^1 t^m (1-t)^n dt = m! n! / (m+n+1)!
    # Here m+n=4.
    # ------------------------------------------------------------
    idx = torch.arange(2, device=device)

    a = idx.view(2, 1, 1, 1)
    p = idx.view(1, 2, 1, 1)
    q = idx.view(1, 1, 2, 1)
    r = idx.view(1, 1, 1, 2)

    m = (a + p + q + r).to(dtype)      # number of l_1 factors
    n = 4.0 - m                        # number of l_0 factors

    G1 = h * torch.exp(
        torch.lgamma(m + 1.0)
        + torch.lgamma(n + 1.0)
        - torch.lgamma(torch.tensor(6.0, device=device, dtype=dtype))
    )  # denominator = 5!

    # ------------------------------------------------------------
    # Build 3D fourth-order tensor analytically.
    # Each 3D shape function is a tensor product:
    #
    # N_i(x,y,z) = l_ix(x) l_iy(y) l_iz(z)
    # ------------------------------------------------------------
    node_bits = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        device=device,
        dtype=torch.long,
    )

    ix = node_bits[:, 0]
    iy = node_bits[:, 1]
    iz = node_bits[:, 2]

    T4 = (
        G1[ix[:, None, None, None],
           ix[None, :, None, None],
           ix[None, None, :, None],
           ix[None, None, None, :]]
        *
        G1[iy[:, None, None, None],
           iy[None, :, None, None],
           iy[None, None, :, None],
           iy[None, None, None, :]]
        *
        G1[iz[:, None, None, None],
           iz[None, :, None, None],
           iz[None, None, :, None],
           iz[None, None, None, :]]
    )

    _HEX_Q1_ANALYTIC_CACHE[key] = (M, K, T4)
    return M, K, T4


def ac_weak_fe_loss_hex_q1_neumann_analytic(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    normalize=True,
    robust=False,
    eps=1e-8,
):
    """
    Analytical Q1 hexahedral weak-form residual loss for Allen-Cahn.

    PDE:
        u_t = Δu - (1/eps2)(u^3 - u)

    Weak form:
        ∫ u_t v dV
        + ∫ grad(u) · grad(v) dV
        + (1/eps2) ∫ (u^3 - u) v dV
        = 0

    This uses rectangular/hexahedral Q1 shape functions, like the
    VINO rectangular-element idea, extended to 3D.

    No numerical quadrature is used.
    """

    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")

    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()
    u1 = u_next.squeeze(-1).float()

    U0 = _cube_to_hex8(u0)  # (B,Cx,Cy,Cz,8)
    U1 = _cube_to_hex8(u1)  # (B,Cx,Cy,Cz,8)

    M, K, T4 = _hex_q1_analytic_data(dx, u1.device, u1.dtype)

    # ------------------------------------------------------------
    # 1. Time term:
    #
    # ∫ u_t N_i dV
    # =
    # Σ_j Ut_j ∫ N_j N_i dV
    # ------------------------------------------------------------
    Ut = (U1 - U0) / dt

    time_vec = torch.einsum(
        "bxyzj,ij->bxyzi",
        Ut,
        M,
    )

    # ------------------------------------------------------------
    # 2. Diffusion term:
    #
    # ∫ grad(u) · grad(N_i) dV
    # =
    # Σ_j U_j ∫ grad(N_j) · grad(N_i) dV
    # ------------------------------------------------------------
    diff_vec = torch.einsum(
        "bxyzj,ij->bxyzi",
        U1,
        K,
    )

    # ------------------------------------------------------------
    # 3. Nonlinear reaction term:
    #
    # (1/eps2) ∫ (u^3 - u) N_i dV
    #
    # u^3 part:
    # Σ_pqr U_p U_q U_r ∫ N_p N_q N_r N_i dV
    # ------------------------------------------------------------
    cubic_vec = torch.einsum(
        "bxyzp,bxyzq,bxyzr,ipqr->bxyzi",
        U1,
        U1,
        U1,
        T4,
    )

    linear_vec = torch.einsum(
        "bxyzj,ij->bxyzi",
        U1,
        M,
    )

    react_vec = (cubic_vec - linear_vec) / eps2

    # ------------------------------------------------------------
    # Final weak residual vector per element and local node
    # ------------------------------------------------------------
    r_vec = time_vec + diff_vec + react_vec

    if normalize:
        s_t = torch.sqrt(
            (time_vec ** 2).mean(dim=(1, 2, 3, 4), keepdim=True).detach()
            + eps
        )
        s_d = torch.sqrt(
            (diff_vec ** 2).mean(dim=(1, 2, 3, 4), keepdim=True).detach()
            + eps
        )
        s_r = torch.sqrt(
            (react_vec ** 2).mean(dim=(1, 2, 3, 4), keepdim=True).detach()
            + eps
        )

        scale = (s_t + s_d + s_r) / 3.0
        r_vec = r_vec / (scale + eps)

    if robust:
        loss = torch.sqrt(r_vec * r_vec + eps).mean()
    else:
        loss = (r_vec * r_vec).mean()

    return loss.to(u_next.dtype)


#####
# ============================================================
# AC3D fully analytical P1 tetrahedral weak-form residual loss
# No centroid quadrature, no Gauss quadrature.
# ============================================================

import torch
import torch.nn.functional as F

_TETRA_P1_FULL_ANALYTIC_CACHE = {}


def _cube_to_tetrahedra_p1_full(u):
    """
    Split each cube into 6 tetrahedra.

    Input:
        u: (B,Sx,Sy,Sz)

    Output:
        U: (B,Sx-1,Sy-1,Sz-1,6,4)

    Each tetrahedron has 4 P1 nodes.
    """
    if u.dim() != 4:
        raise ValueError("u must have shape (B,Sx,Sy,Sz)")

    v000 = u[:, :-1, :-1, :-1]
    v100 = u[:,  1:, :-1, :-1]
    v010 = u[:, :-1,  1:, :-1]
    v110 = u[:,  1:,  1:, :-1]
    v001 = u[:, :-1, :-1,  1:]
    v101 = u[:,  1:, :-1,  1:]
    v011 = u[:, :-1,  1:,  1:]
    v111 = u[:,  1:,  1:,  1:]

    tet1 = torch.stack([v000, v100, v110, v111], dim=-1)
    tet2 = torch.stack([v000, v100, v101, v111], dim=-1)
    tet3 = torch.stack([v000, v001, v101, v111], dim=-1)
    tet4 = torch.stack([v000, v001, v011, v111], dim=-1)
    tet5 = torch.stack([v000, v010, v011, v111], dim=-1)
    tet6 = torch.stack([v000, v010, v110, v111], dim=-1)

    return torch.stack([tet1, tet2, tet3, tet4, tet5, tet6], dim=-2)


def _tetra_p1_full_analytic_data(dx, device, dtype):
    """
    Fully analytical P1 tetrahedral element data.

    Returns
    -------
    grad_phi : (6,4,3)
        Gradients of P1 shape functions.

    volume : (6,)
        Tetrahedron volumes.

    M : (6,4,4)
        Exact mass matrix:
            M_ij = ∫_K phi_i phi_j dV

    Q4 : (6,4,4,4,4)
        Exact fourth-order tensor:
            Q4[i,p,q,r] = ∫_K phi_i phi_p phi_q phi_r dV
    """
    key = (float(dx), str(device), str(dtype))
    if key in _TETRA_P1_FULL_ANALYTIC_CACHE:
        return _TETRA_P1_FULL_ANALYTIC_CACHE[key]

    # ------------------------------------------------------------
    # Cube vertices
    # ------------------------------------------------------------
    X = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # v000
            [1.0, 0.0, 0.0],  # v100
            [0.0, 1.0, 0.0],  # v010
            [1.0, 1.0, 0.0],  # v110
            [0.0, 0.0, 1.0],  # v001
            [1.0, 0.0, 1.0],  # v101
            [0.0, 1.0, 1.0],  # v011
            [1.0, 1.0, 1.0],  # v111
        ],
        device=device,
        dtype=dtype,
    ) * dx

    tet_ids = torch.tensor(
        [
            [0, 1, 3, 7],
            [0, 1, 5, 7],
            [0, 4, 5, 7],
            [0, 4, 6, 7],
            [0, 2, 6, 7],
            [0, 2, 3, 7],
        ],
        device=device,
        dtype=torch.long,
    )

    tet_xyz = X[tet_ids]  # (6,4,3)

    # ------------------------------------------------------------
    # Reference tetrahedral P1 gradients
    # Reference shape functions:
    # phi_0 = 1-r-s-t
    # phi_1 = r
    # phi_2 = s
    # phi_3 = t
    # ------------------------------------------------------------
    grad_ref = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [ 1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0,  0.0,  1.0],
        ],
        device=device,
        dtype=dtype,
    )  # (4,3)

    x0 = tet_xyz[:, 0, :]
    x1 = tet_xyz[:, 1, :]
    x2 = tet_xyz[:, 2, :]
    x3 = tet_xyz[:, 3, :]

    J = torch.stack([x1 - x0, x2 - x0, x3 - x0], dim=-1)  # (6,3,3)
    invJT = torch.linalg.inv(J).transpose(-1, -2)

    grad_phi = torch.einsum("fij,aj->fai", invJT, grad_ref)  # (6,4,3)
    volume = torch.abs(torch.linalg.det(J)) / 6.0             # (6,)

    # ------------------------------------------------------------
    # Exact P1 mass matrix:
    #
    # ∫_K phi_i phi_j dV =
    #   |K|/10  if i=j
    #   |K|/20  if i≠j
    # ------------------------------------------------------------
    eye4 = torch.eye(4, device=device, dtype=dtype)
    ones4 = torch.ones((4, 4), device=device, dtype=dtype)

    M_base = ones4 / 20.0 + eye4 / 20.0
    M = volume[:, None, None] * M_base[None, :, :]

    # ------------------------------------------------------------
    # Exact fourth-order tensor:
    #
    # Q4[i,p,q,r] = ∫_K phi_i phi_p phi_q phi_r dV
    #
    # General formula on tetrahedron:
    #
    # ∫_K λ_1^a1 λ_2^a2 λ_3^a3 λ_4^a4 dV
    # =
    # |K| * 3! * Π(a_m!) / (3 + Σ a_m)!
    #
    # Here total degree = 4, so denominator = 7!
    # ------------------------------------------------------------
    idx = torch.arange(4, device=device)

    i = idx.view(4, 1, 1, 1)
    p = idx.view(1, 4, 1, 1)
    q = idx.view(1, 1, 4, 1)
    r = idx.view(1, 1, 1, 4)

    ids = torch.stack(
        [
            i.expand(4, 4, 4, 4),
            p.expand(4, 4, 4, 4),
            q.expand(4, 4, 4, 4),
            r.expand(4, 4, 4, 4),
        ],
        dim=-1,
    )  # (4,4,4,4,4)

    counts = F.one_hot(ids, num_classes=4).sum(dim=-2).to(dtype)

    coeff = (
        6.0
        * torch.exp(torch.lgamma(counts + 1.0).sum(dim=-1))
        / 5040.0
    )  # (4,4,4,4)

    Q4 = volume[:, None, None, None, None] * coeff[None, :, :, :, :]

    _TETRA_P1_FULL_ANALYTIC_CACHE[key] = (grad_phi, volume, M, Q4)
    return grad_phi, volume, M, Q4


def _tetra_p1_cubic_vec_safe(U, Q4):
    """
    Safe fully analytical cubic term for P1 tetrahedra.

    Computes:
        cubic_vec_i = sum_{p,q,r} U_p U_q U_r Q4[i,p,q,r]

    U  : (B,Cx,Cy,Cz,6,4)
    Q4 : (6,4,4,4,4)

    Output:
        cubic_vec : (B,Cx,Cy,Cz,6,4)

    This avoids a large einsum/GEMM during backward.
    The loops are only over local basis indices 0..3.
    """
    out = U.new_zeros(U.shape[:-1] + (4,))  # (B,Cx,Cy,Cz,6,4)

    for p in range(4):
        Up = U[..., p]
        for q in range(4):
            Upq = Up * U[..., q]
            for r in range(4):
                term = Upq * U[..., r]  # (B,Cx,Cy,Cz,6)

                coeff = Q4[:, :, p, q, r]  # (6,4)
                coeff = coeff.view(1, 1, 1, 1, 6, 4)

                out = out + term.unsqueeze(-1) * coeff

    return out

def ac_weak_fe_loss_tetra_neumann_fully_analytic(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    normalize=True,
    robust=False,
    eps=1e-8,
):
    """
    Fully analytical P1 tetrahedral weak-form residual loss
    for the Allen--Cahn equation with homogeneous Neumann BC.

    PDE:
        u_t = Δu - (1/eps2)(u^3 - u)

    Weak form:
        ∫_K u_t phi_i dV
        + ∫_K grad(u) · grad(phi_i) dV
        + (1/eps2) ∫_K (u^3 - u) phi_i dV
        = 0

    This function computes all element integrals analytically:
        - time term by exact mass matrix,
        - diffusion term by exact stiffness action,
        - nonlinear cubic term by exact fourth-order tensor.

    No Gauss quadrature.
    No centroid approximation.
    """

    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")

    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()
    u1 = u_next.squeeze(-1).float()

    # Element nodal values: (B,Cx,Cy,Cz,6,4)
    U0 = _cube_to_tetrahedra_p1_full(u0)
    U1 = _cube_to_tetrahedra_p1_full(u1)

    grad_phi, volume, M, Q4 = _tetra_p1_full_analytic_data(
        dx, u1.device, u1.dtype
    )

    # ------------------------------------------------------------
    # 1. Exact time term:
    #
    # ∫_K u_t phi_i dV
    # =
    # Σ_j Ut_j ∫_K phi_j phi_i dV
    # ------------------------------------------------------------
    Ut = (U1 - U0) / dt

    time_vec = torch.einsum(
        "nxyzkj,kij->nxyzki",
        Ut,
        M,
    )

    # ------------------------------------------------------------
    # 2. Exact diffusion term:
    #
    # ∫_K grad(u) · grad(phi_i) dV
    #
    # grad(u) = Σ_j U_j grad(phi_j)
    # ------------------------------------------------------------
    grad_u = torch.einsum(
        "nxyzkj,kjm->nxyzkm",
        U1,
        grad_phi,
    )

    diff_vec = torch.einsum(
        "nxyzkm,kim->nxyzki",
        grad_u,
        grad_phi,
    ) * volume.view(1, 1, 1, 1, 6, 1)

    # ------------------------------------------------------------
    # 3. Exact nonlinear reaction term:
    #
    # (1/eps2) ∫_K (u^3-u) phi_i dV
    #
    # ∫_K u^3 phi_i dV
    # =
    # Σ_pqr U_p U_q U_r ∫_K phi_p phi_q phi_r phi_i dV
    # ------------------------------------------------------------
    cubic_vec = _tetra_p1_cubic_vec_safe(U1, Q4)

    linear_vec = torch.einsum(
        "nxyzkj,kij->nxyzki",
        U1,
        M,
    )

    react_vec = (cubic_vec - linear_vec) / eps2

    # ------------------------------------------------------------
    # Final local weak residual vector
    # ------------------------------------------------------------
    r_vec = time_vec + diff_vec + react_vec

    if normalize:
        s_t = torch.sqrt(
            (time_vec ** 2).mean(dim=(1, 2, 3, 4, 5), keepdim=True).detach()
            + eps
        )
        s_d = torch.sqrt(
            (diff_vec ** 2).mean(dim=(1, 2, 3, 4, 5), keepdim=True).detach()
            + eps
        )
        s_r = torch.sqrt(
            (react_vec ** 2).mean(dim=(1, 2, 3, 4, 5), keepdim=True).detach()
            + eps
        )

        scale = (s_t + s_d + s_r) / 3.0
        r_vec = r_vec / (scale + eps)

    if robust:
        loss = torch.sqrt(r_vec * r_vec + eps).mean()
    else:
        loss = (r_vec * r_vec).mean()

    return loss.to(u_next.dtype)

####


def _tetrahedron_quadrature_points(dx, device):
    """
    Returns barycentric coordinates and weights for 1-point quadrature
    in reference tetrahedron.
    """
    # centroid of tetrahedron
    xi = torch.tensor([0.25, 0.25, 0.25, 0.25], device=device)
    weight = dx**3 / 6.0  # volume of tetrahedron
    return xi, weight

def _cube_to_tetrahedra(u):
    """
    Vectorized split of every voxel/cell into 6 tetrahedra.

    Parameters
    ----------
    u : torch.Tensor
        Shape (B, Sx, Sy, Sz)

    Returns
    -------
    torch.Tensor
        Shape (B, N_tets, 4), where N_tets = 6 * (Sx-1) * (Sy-1) * (Sz-1)

    Tetrahedra use the same 6-tet decomposition as the original loop version:
        [v000, v100, v010, v001]
        [v100, v110, v010, v111]
        [v100, v010, v001, v111]
        [v010, v001, v011, v111]
        [v100, v001, v101, v111]
        [v001, v011, v101, v111]
    """
    if u.dim() != 4:
        raise ValueError("u must have shape (B,Sx,Sy,Sz)")

    # Corner values for every cube, all at once
    v000 = u[:, :-1, :-1, :-1]
    v100 = u[:,  1:, :-1, :-1]
    v010 = u[:, :-1,  1:, :-1]
    v110 = u[:,  1:,  1:, :-1]
    v001 = u[:, :-1, :-1,  1:]
    v101 = u[:,  1:, :-1,  1:]
    v011 = u[:, :-1,  1:,  1:]
    v111 = u[:,  1:,  1:,  1:]

    # Build 6 tetrahedra per cube, preserving the original ordering
    tet1 = torch.stack([v000, v100, v010, v001], dim=-1)
    tet2 = torch.stack([v100, v110, v010, v111], dim=-1)
    tet3 = torch.stack([v100, v010, v001, v111], dim=-1)
    tet4 = torch.stack([v010, v001, v011, v111], dim=-1)
    tet5 = torch.stack([v100, v001, v101, v111], dim=-1)
    tet6 = torch.stack([v001, v011, v101, v111], dim=-1)

    # Stack tetra family dimension
    # Shape: (B, Sx-1, Sy-1, Sz-1, 6, 4)
    tets = torch.stack([tet1, tet2, tet3, tet4, tet5, tet6], dim=-2)

    # Flatten all cells and tetrahedra into one tet axis
    B = u.shape[0]
    return tets.reshape(B, -1, 4)


def weak_form_AC_gauss(u_prev, u_next, dx, dt, eps2, normalize=True, robust=False, eps=1e-8):
    """
    Weak-form AC loss using vectorized tetrahedral Gauss quadrature.

    Parameters
    ----------
    u_prev, u_next : torch.Tensor
        Shape (B,S,S,S,1)
    dx, dt, eps2 : float

    Returns
    -------
    torch.Tensor
        Scalar loss
    """
    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()
    u1 = u_next.squeeze(-1).float()

    ut = (u1 - u0) / dt
    rhs = pde_rhs_ac_neumann(u1, dx, eps2)
    R = ut - rhs

    # Vectorized tetra assembly
    R_tets = _cube_to_tetrahedra(R)      # (B, Ntets, 4)

    # 1-point tetra centroid quadrature:
    # linear field value at centroid = average of vertex values
    R_centroid = R_tets.mean(dim=-1)     # (B, Ntets)

    # volume of each tetra from a cube of size dx
    tet_volume = (dx ** 3) / 6.0

    # Mean over batch and elements, then multiply by tet volume
    #loss = tet_volume * (R_centroid ** 2).mean()
    # EXACTLY the same integration as local residual term
    #loss = tetra_gauss_l2_scalar(R, dx).mean()

    # EXACTLY the same normalization as local residual term
    if normalize:
        s_ut = torch.sqrt(tetra_gauss_l2_scalar(ut, dx).detach() + eps)
        s_rhs = torch.sqrt(tetra_gauss_l2_scalar(rhs, dx).detach() + eps)
        scale = 0.5 * (s_ut + s_rhs) + eps
        R = R / scale.view(-1, 1, 1, 1)

    # EXACTLY the same integration logic as local residual term
    if robust:
        loss = tetra_gauss_integral_scalar(torch.sqrt(R * R + eps), dx).mean()
    else:
        loss = tetra_gauss_l2_scalar(R, dx).mean()

    return loss.to(u_next.dtype)



def weak_form_CH_gauss(u_prev, u_next, dx, dt, eps2, normalize=True, robust=False, mass_weight=0.1):
    """
    Weak-form CH loss using vectorized tetrahedral Gauss quadrature.
    """
    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()
    u1 = u_next.squeeze(-1).float()

    ut = (u1 - u0) / dt
    rhs = pde_rhs_ch_neumann(u1, dx, eps2)

    R = ut - rhs
    R = R - R.mean(dim=(1, 2, 3), keepdim=True)

    R_tets = _cube_to_tetrahedra(R)      # (B, Ntets, 4)
    R_centroid = R_tets.mean(dim=-1)     # (B, Ntets)

    tet_volume = (dx ** 3) / 6.0
    loss = tet_volume * (R_centroid ** 2).mean()



    if normalize:
        s_ut = torch.sqrt(tetra_gauss_l2_scalar(ut, dx).detach() + 1e-8)        # (B,)
        s_rhs = torch.sqrt(tetra_gauss_l2_scalar(rhs, dx).detach() + 1e-8) # (B,)
        scale = 0.5 * (s_ut + s_rhs) + 1e-8
        R = R / scale.view(-1, 1, 1, 1)

    if robust:
        loss_res = tetra_gauss_integral_scalar(torch.sqrt(R * R + 1e-8), dx).mean()
    else:
        loss_res = tetra_gauss_l2_scalar(R, dx).mean()


    return (loss_res).to(u_next.dtype)

def mixed_form_CH_spectral_gauss_single_step(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    tau=0.5,
    normalize=True,
    robust=False,
    mass_weight=0.25,
    eps=1e-8,
):
    """
    Mixed-form CH least-squares residual, single step, spectral Neumann version.

    CH mixed form:
        u_t = Delta(mu)
        mu  = -eps2 * Delta(u) + (u^3 - u)

    We compute:
        u_tau = (1-tau) u_prev + tau u_next
        mu_tau = -eps2 * Delta(u_tau) + (u_tau^3 - u_tau)
        R_mix = (u_next - u_prev)/dt - Delta(mu_tau)

    The ONLY intended difference from the local version
    mixed_form_CH_physical_gauss_single_step is:
        - Laplacians are computed with the Neumann cosine spectral operator
          instead of finite differences.

    Everything else is kept identical:
        - same mixed formulation
        - same tau-interpolation
        - same normalization
        - same tetrahedral integration
        - same mass penalty

    Inputs
    ------
    u_prev, u_next : (B,S,S,S,1)

    Returns
    -------
    scalar tensor
    """
    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()   # (B,S,S,S)
    u1 = u_next.squeeze(-1).float()   # (B,S,S,S)

    # same time derivative as local version
    ut = (u1 - u0) / dt

    # same tau-state as local version
    u_tau = (1.0 - tau) * u0 + tau * u1

    # ONLY difference: spectral Neumann Laplacians instead of FD
    lap_u_tau = laplacian_neumann_cosine_3d(u_tau, dx)
    mu_tau = -eps2 * lap_u_tau + (u_tau ** 3 - u_tau)
    lap_mu_tau = laplacian_neumann_cosine_3d(mu_tau, dx)

    # same mixed residual
    R = ut - lap_mu_tau

    # same mean-free correction
    R = R - R.mean(dim=(1, 2, 3), keepdim=True)

    # same normalization
    if normalize:
        s_ut = torch.sqrt(tetra_gauss_l2_scalar(ut, dx).detach() + eps)         # (B,)
        s_lm = torch.sqrt(tetra_gauss_l2_scalar(lap_mu_tau, dx).detach() + eps) # (B,)
        scale = 0.5 * (s_ut + s_lm) + eps
        R = R / scale.view(-1, 1, 1, 1)

    # same residual integration
    if robust:
        loss_res = tetra_gauss_integral_scalar(torch.sqrt(R * R + eps), dx).mean()
    else:
        loss_res = tetra_gauss_l2_scalar(R, dx).mean()



    return (loss_res ).to(u_next.dtype)


import torch
import torch.nn.functional as F


def laplacian_neumann_fd_3d(u, dx):
    """
    3D finite-difference Laplacian with homogeneous Neumann BC
    implemented by replicate padding.

    u: (B,Sx,Sy,Sz)
    returns: (B,Sx,Sy,Sz)
    """
    if u.dim() != 4:
        raise ValueError("u must have shape (B,Sx,Sy,Sz)")

    up = F.pad(u.unsqueeze(1), (1, 1, 1, 1, 1, 1), mode="replicate").squeeze(1)

    c = up[:, 1:-1, 1:-1, 1:-1]

    lap = (
        up[:, :-2, 1:-1, 1:-1] + up[:, 2:, 1:-1, 1:-1]
        + up[:, 1:-1, :-2, 1:-1] + up[:, 1:-1, 2:, 1:-1]
        + up[:, 1:-1, 1:-1, :-2] + up[:, 1:-1, 1:-1, 2:]
        - 6.0 * c
    ) / (dx ** 2)

    return lap


def _tet6_centroid_values(field):
    """
    Vectorized tetrahedral centroid values from a scalar nodal field on a regular grid.

    field: (B,Sx,Sy,Sz)

    returns: (B,Sx-1,Sy-1,Sz-1,6)
    where the last dimension contains the centroid values of the 6 tetrahedra
    inside each cube.
    """
    if field.dim() != 4:
        raise ValueError("field must have shape (B,Sx,Sy,Sz)")

    v000 = field[:, :-1, :-1, :-1]
    v100 = field[:,  1:, :-1, :-1]
    v010 = field[:, :-1,  1:, :-1]
    v110 = field[:,  1:,  1:, :-1]
    v001 = field[:, :-1, :-1,  1:]
    v101 = field[:,  1:, :-1,  1:]
    v011 = field[:, :-1,  1:,  1:]
    v111 = field[:,  1:,  1:,  1:]

    c1 = 0.25 * (v000 + v100 + v010 + v001)
    c2 = 0.25 * (v100 + v110 + v010 + v111)
    c3 = 0.25 * (v100 + v010 + v001 + v111)
    c4 = 0.25 * (v010 + v001 + v011 + v111)
    c5 = 0.25 * (v100 + v001 + v101 + v111)
    c6 = 0.25 * (v001 + v011 + v101 + v111)

    return torch.stack([c1, c2, c3, c4, c5, c6], dim=-1)


def tetra_gauss_integral_scalar(field, dx):
    """
    Integrate a scalar nodal field over the domain using:
      - 6 tetrahedra per cube
      - 1-point Gauss quadrature at each tetra centroid

    field: (B,Sx,Sy,Sz)

    returns: (B,)
    """
    centroids = _tet6_centroid_values(field)          # (B,Sx-1,Sy-1,Sz-1,6)
    tet_volume = (dx ** 3) / 6.0
    return tet_volume * centroids.sum(dim=(1, 2, 3, 4))


def tetra_gauss_l2_scalar(field, dx):
    """
    Integrate field^2 over the domain using tetrahedral centroid Gauss quadrature.

    field: (B,Sx,Sy,Sz)

    returns: (B,)
    """
    centroids = _tet6_centroid_values(field)          # (B,Sx-1,Sy-1,Sz-1,6)
    tet_volume = (dx ** 3) / 6.0
    return tet_volume * (centroids ** 2).sum(dim=(1, 2, 3, 4))


def low_k_mse_neumann(u_pred, u_ref, frac=0.45):
    """
    Low-mode spectral MSE for homogeneous Neumann BC.

    Uses 3D cosine-transform coefficients instead of periodic FFT modes.

    Parameters
    ----------
    u_pred, u_ref : torch.Tensor
        Shape (B,S,S,S,1)
    frac : float
        Fraction of low cosine modes kept, based on radial index.

    Returns
    -------
    scalar tensor
    """
    if u_pred.dim() != 5 or u_pred.shape[-1] != 1:
        raise ValueError("u_pred must have shape (B,S,S,S,1)")
    if u_ref.dim() != 5 or u_ref.shape[-1] != 1:
        raise ValueError("u_ref must have shape (B,S,S,S,1)")

    up = u_pred.squeeze(-1).float()   # (B,S,S,S)
    ur = u_ref.squeeze(-1).float()    # (B,S,S,S)

    B, nx, ny, nz = up.shape
    device = up.device
    dtype = up.dtype

    # Neumann-consistent cosine coefficients
    Up = dct3_neumann(up)
    Ur = dct3_neumann(ur)

    # Mode-index grid (not fftfreq, since this is cosine/Neumann basis)
    px = torch.arange(nx, device=device, dtype=dtype)
    py = torch.arange(ny, device=device, dtype=dtype)
    pz = torch.arange(nz, device=device, dtype=dtype)

    PX, PY, PZ = torch.meshgrid(px, py, pz, indexing='ij')
    r = torch.sqrt(PX * PX + PY * PY + PZ * PZ)
    rmax = r.max().clamp_min(1.0)

    mask = (r <= frac * rmax).to(dtype)   # (nx,ny,nz)

    Dh = Up - Ur                          # (B,nx,ny,nz)
    spec_mse = (Dh ** 2) * mask.unsqueeze(0)

    denom = mask.sum().clamp_min(1.0)
    return spec_mse.sum() / denom

def ac_incremental_energy_loss_autoregressive(
    model,
    x_init,
    dt,
    dx,
    eps2,
    rollout_steps=5,
    relative=True,
    robust=True,
    end_weight=2.0,
    eps=1e-8,
):
    """
    Multi-step energy loss using autoregressive rollout (independent of T_OUT).

    Parameters
    ----------
    model : neural operator
    x_init : (B,S,S,S,T_in)
        input window
    rollout_steps : int
        number of autoregressive steps (e.g. 5)

    Returns
    -------
    scalar tensor
    """

    B = x_init.shape[0]
    device = x_init.device
    dtype = x_init.dtype

    # initial last frame (u^n)
    u_prev = x_init[..., -1:].contiguous()

    # rolling input window
    x_cur = x_init

    losses = []

    # step weights
    if rollout_steps == 1:
        w = torch.ones(1, device=device, dtype=dtype)
    else:
        w = torch.linspace(1.0, float(end_weight), rollout_steps,
                           device=device, dtype=dtype)

    for k in range(rollout_steps):

        # predict one step
        u_next = model(x_cur)  # (B,S,S,S,1)

        # incremental energy
        du = u_next - u_prev
        move = 0.5 / dt * l2_sq_tetra_linear(du, dx)
        energy = energy_AC_tetra_exact(u_next, dx, eps2)
        Jk = move + energy

        # normalization
        if relative:
            E_prev = energy_AC_tetra_exact(u_prev, dx, eps2).abs().detach() + 1e-6
            val = Jk / E_prev
        else:
            val = Jk

        if robust:
            lk = torch.sqrt(val * val + eps)
        else:
            lk = val * val

        losses.append(lk)

        # update for next step
        u_prev = u_next
        x_cur = torch.cat([x_cur[..., 1:], u_next], dim=-1)

    L = torch.stack(losses, dim=-1) * w.view(1, rollout_steps)

    return (L.sum(dim=-1) / (w.sum() + eps)).mean().to(dtype)



###

import math
import torch


def pde_rhs_ac_neumann_fd(u, dx, eps2):
    """
    Allen-Cahn RHS in physical space with FD Neumann Laplacian.

    u : (B,S,S,S)
    returns : (B,S,S,S)
    """
    lap_u = laplacian_neumann_fd_3d(u, dx)
    return lap_u - (1.0 / eps2) * (u**3 - u)


def ac_strong_form_residual_tau_neumann(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    tau,
    normalize=True,
    eps=1e-8,
):
    """
    Strong-form AC residual at one temporal collocation point tau.

    Parameters
    ----------
    u_prev, u_next : torch.Tensor
        Shape (B,S,S,S,1)
    tau : float in [0,1]

    Returns
    -------
    R_tau : torch.Tensor
        Shape (B,S,S,S)
        Residual field at collocation point tau
    """
    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()   # (B,S,S,S)
    u1 = u_next.squeeze(-1).float()   # (B,S,S,S)

    # one-step time derivative over [t_n, t_{n+1}]
    u_dot = (u1 - u0) / dt

    # temporal collocation state
    u_tau = (1.0 - tau) * u0 + tau * u1

    # Allen-Cahn operator evaluated at collocation state
    Du_tau = pde_rhs_ac_neumann_fd(u_tau, dx, eps2)

    if normalize:
        n_udot = torch.sqrt(tetra_gauss_l2_scalar(u_dot, dx).detach() + eps)   # (B,)
        n_Du   = torch.sqrt(tetra_gauss_l2_scalar(Du_tau, dx).detach() + eps)  # (B,)

        u_dot_n = u_dot / n_udot.view(-1, 1, 1, 1)
        Du_tau_n = Du_tau / n_Du.view(-1, 1, 1, 1)

        R_tau = u_dot_n - Du_tau_n
    else:
        R_tau = u_dot - Du_tau

    return R_tau.to(u_next.dtype)


def ac_strong_form_gauss_lobatto_loss_neumann(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    taus=None,
    normalize=True,
    robust=False,
    eps=1e-8,
):
    """
    Two-point Gauss-Lobatto strong-form collocation loss for Allen-Cahn.

    Uses
        tau_{1,2} = 1/2 ± 1/(2 sqrt(5))
    by default.

    Loss:
        1/2 ( ||Rhat_tau1||_{L2}^2 + ||Rhat_tau2||_{L2}^2 )

    where each L2 norm is computed over the spatial domain with tetrahedral
    centroid quadrature.

    Parameters
    ----------
    u_prev, u_next : (B,S,S,S,1)
    normalize : bool
        If True, use normalized residual formulation.
    robust : bool
        If True, integrate sqrt(R^2 + eps) instead of R^2.

    Returns
    -------
    scalar tensor
    """
    if taus is None:
        tau_off = 1.0 / (2.0 * math.sqrt(5.0))
        taus = (0.5 - tau_off, 0.5 + tau_off)

    losses = []
    for tau in taus:
        R_tau = ac_strong_form_residual_tau_neumann(
            u_prev,
            u_next,
            dt,
            dx,
            eps2,
            tau=tau,
            normalize=normalize,
            eps=eps,
        )  # (B,S,S,S)

        if robust:
            l_tau = tetra_gauss_integral_scalar(torch.sqrt(R_tau * R_tau + eps), dx).mean()
        else:
            l_tau = tetra_gauss_l2_scalar(R_tau, dx).mean()

        losses.append(l_tau)

    return (0.5 * sum(losses)).to(u_next.dtype)



import torch


def ac_variational_residual_modes_neumann(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    modes=(6, 6, 6),
    normalize=True,
    eps=1e-8,
):
    """
    Weak/variational Allen-Cahn residual projected onto Neumann cosine modes.

    Parameters
    ----------
    u_prev, u_next : torch.Tensor
        Shape (B,S,S,S,1)
    dt, dx, eps2 : float
    modes : tuple(int,int,int)
        Number of low cosine modes kept in each direction, e.g. (6,6,6)
    normalize : bool
        If True, normalize time part and PDE part for stability.

    Returns
    -------
    r_hat : torch.Tensor
        Shape (B,mx,my,mz)
        Modal weak residual coefficients.
    """
    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()   # (B,S,S,S)
    u1 = u_next.squeeze(-1).float()   # (B,S,S,S)

    B, nx, ny, nz = u1.shape
    mx = min(int(modes[0]), nx)
    my = min(int(modes[1]), ny)
    mz = min(int(modes[2]), nz)

    # 1) time term
    ut = (u1 - u0) / dt
    ut_hat = dct3_neumann(ut)[:, :mx, :my, :mz]

    # 2) diffusion term in weak form:
    #    <grad u, grad phi_k> = k^2 <u, phi_k>
    u1_hat = dct3_neumann(u1)[:, :mx, :my, :mz]
    k2 = _neumann_k2_grid(nx, ny, nz, dx, u1.device, u1.dtype)[:mx, :my, :mz]
    diff_hat = u1_hat * k2.unsqueeze(0)

    # 3) nonlinear term
    nonlin = (u1**3 - u1) / eps2
    nonlin_hat = dct3_neumann(nonlin)[:, :mx, :my, :mz]

    # weak residual in modal form:
    # <u_t,phi> + <grad u, grad phi> + <(u^3-u)/eps2, phi>
    pde_hat = diff_hat + nonlin_hat

    if normalize:
        n_t = torch.sqrt((ut_hat**2).mean(dim=(1, 2, 3), keepdim=True) + eps)
        n_p = torch.sqrt((pde_hat**2).mean(dim=(1, 2, 3), keepdim=True) + eps)
        r_hat = ut_hat / n_t + pde_hat / n_p
    else:
        r_hat = ut_hat + pde_hat

    return r_hat.to(u_next.dtype)


def ac_variational_galerkin_loss_neumann(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    modes=(6, 6, 6),
    normalize=True,
    robust=False,
    eps=1e-8,
):
    """
    Variational / weak-form Galerkin loss for Allen-Cahn with Neumann cosine tests.

    Parameters
    ----------
    u_prev, u_next : (B,S,S,S,1)
    modes : tuple(int,int,int)
        Number of retained low modes in x,y,z
    normalize : bool
        Normalize time and PDE parts before forming residual
    robust : bool
        If True, use sqrt(r^2 + eps) instead of r^2

    Returns
    -------
    scalar tensor
    """
    r_hat = ac_variational_residual_modes_neumann(
        u_prev,
        u_next,
        dt,
        dx,
        eps2,
        modes=modes,
        normalize=normalize,
        eps=eps,
    )  # (B,mx,my,mz)

    if robust:
        loss = torch.sqrt(r_hat * r_hat + eps).mean()
    else:
        loss = (r_hat * r_hat).mean()

    return loss.to(u_next.dtype)



import torch


def ac_weak_fe_residual_tetra_neumann(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    normalize=True,
    eps=1e-8,
):
    """
    Tetrahedral FE weak-form residual for Allen-Cahn with homogeneous Neumann BC.

    Weak form on each tetrahedron K, for each local P1 basis/test function phi_i:
        r_i^K
        =
        ∫_K u_t phi_i dV
        +
        ∫_K grad(u) · grad(phi_i) dV
        +
        (1/eps2) ∫_K (u^3-u) phi_i dV

    Notes
    -----
    - Neumann BC is natural in the weak form, so no boundary penalty is needed.
    - u is represented with piecewise-linear P1 tetrahedral basis functions.
    - grad(u) is constant on each tetrahedron.
    - The gradient term is exact on each tetrahedron.
    - The time and nonlinear terms are evaluated by centroid quadrature.

    Parameters
    ----------
    u_prev, u_next : torch.Tensor
        Shape (B,S,S,S,1)
    dt, dx, eps2 : float
    normalize : bool
        If True, normalize the weak residual vectors by the scale of their components.

    Returns
    -------
    r_vec : torch.Tensor
        Shape (B,Cx,Cy,Cz,6,4)
        Weak residual vector entries for all tetrahedra and local basis functions.
    """
    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()   # (B,S,S,S)
    u1 = u_next.squeeze(-1).float()   # (B,S,S,S)

    # Nodal values on all tetrahedra:
    # U0, U1: (B, Cx, Cy, Cz, 6, 4)
    U0 = _cube_to_tetrahedra_all(u0)
    U1 = _cube_to_tetrahedra_all(u1)

    # Geometry / P1 basis data
    grad_phi, volume = _tetra_grad_shape_families(dx, u1.device, u1.dtype)
    # grad_phi: (6,4,3), volume: (6,)
    vol = volume.view(1, 1, 1, 1, 6, 1)   # broadcast to (...,6,1)

    # ------------------------------------------------------------
    # 1) Time term: ∫_K u_t phi_i dV
    # ------------------------------------------------------------
    # u_t nodal values, then centroid value on each tetra
    Ut = (U1 - U0) / dt                          # (B,Cx,Cy,Cz,6,4)
    ut_cent = Ut.mean(dim=-1, keepdim=True)     # (B,Cx,Cy,Cz,6,1)

    # For P1 tetrahedra, phi_i(centroid)=1/4 exactly
    time_vec = ut_cent * (0.25 * vol)           # (B,Cx,Cy,Cz,6,1)
    time_vec = time_vec.expand_as(U1)           # (B,Cx,Cy,Cz,6,4)

    # ------------------------------------------------------------
    # 2) Diffusion weak term: ∫_K grad(u)·grad(phi_i) dV
    # ------------------------------------------------------------
    # grad(u) = sum_a U_a grad(phi_a), constant on each tetra
    grad_u = torch.einsum("bxyzkf,kfi->bxyzki", U1, grad_phi)   # (B,Cx,Cy,Cz,6,3)

    # For each local basis/test function i:
    # ∫_K grad(u)·grad(phi_i) dV = (grad_u · grad_phi_i) * volume
    diff_vec = torch.einsum("bxyzki,kfi->bxyzkf", grad_u, grad_phi) * vol  # (B,Cx,Cy,Cz,6,4)

    # ------------------------------------------------------------
    # 3) Nonlinear term: (1/eps2) ∫_K (u^3-u) phi_i dV
    # ------------------------------------------------------------
    u_cent = U1.mean(dim=-1, keepdim=True)      # (B,Cx,Cy,Cz,6,1)
    nonlin_cent = (u_cent**3 - u_cent) / eps2   # (B,Cx,Cy,Cz,6,1)

    react_vec = nonlin_cent * (0.25 * vol)      # (B,Cx,Cy,Cz,6,1)
    react_vec = react_vec.expand_as(U1)         # (B,Cx,Cy,Cz,6,4)

    # ------------------------------------------------------------
    # Weak residual vector per tetrahedron / local basis
    # ------------------------------------------------------------
    r_vec = time_vec + diff_vec + react_vec     # (B,Cx,Cy,Cz,6,4)

    if normalize:
        # Normalize by the average scale of the three weak-form components
        s_t = torch.sqrt((time_vec**2).mean(dim=(1, 2, 3, 4, 5), keepdim=True).detach() + eps)
        s_d = torch.sqrt((diff_vec**2).mean(dim=(1, 2, 3, 4, 5), keepdim=True).detach() + eps)
        s_r = torch.sqrt((react_vec**2).mean(dim=(1, 2, 3, 4, 5), keepdim=True).detach() + eps)

        scale = (s_t + s_d + s_r) / 3.0
        r_vec = r_vec / (scale + eps)

    return r_vec.to(u_next.dtype)


def ac_weak_fe_loss_tetra_neumann(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    normalize=True,
    robust=False,
    eps=1e-8,
):
    """
    Scalar weak-form FE loss for Allen-Cahn with tetrahedral P1 test functions.

    Loss = mean over all tetrahedra and local basis functions of residual-vector size.

    Parameters
    ----------
    u_prev, u_next : (B,S,S,S,1)

    Returns
    -------
    scalar tensor
    """
    r_vec = ac_weak_fe_residual_tetra_neumann(
        u_prev,
        u_next,
        dt,
        dx,
        eps2,
        normalize=normalize,
        eps=eps,
    )  # (B,Cx,Cy,Cz,6,4)

    if robust:
        loss = torch.sqrt(r_vec * r_vec + eps).mean()
    else:
        loss = (r_vec * r_vec).mean()

    return loss.to(u_next.dtype)

def mixed_form_CH_physical_gauss_single_step(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    tau=0.5,
    normalize=True,
    robust=False,
    mass_weight=0.25,
    eps=1e-8,
):
    """
    Mixed-form CH least-squares residual in physical space, single step, no spectral ops.

    CH mixed form:
        u_t = Delta(mu)
        mu  = -eps2 * Delta(u) + (u^3 - u)

    We compute:
        u_tau = (1-tau) u_prev + tau u_next
        mu_tau = -eps2 * Delta(u_tau) + (u_tau^3 - u_tau)
        R_mix = (u_next - u_prev)/dt - Delta(mu_tau)

    Then integrate R_mix^2 over tetrahedra using 1-point Gauss quadrature.

    Inputs
    ------
    u_prev, u_next : (B,S,S,S,1)

    Returns
    -------
    scalar tensor
    """
    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()   # (B,S,S,S)
    u1 = u_next.squeeze(-1).float()   # (B,S,S,S)

    ut = (u1 - u0) / dt
    u_tau = (1.0 - tau) * u0 + tau * u1

    # mixed split in physical space
    lap_u_tau = laplacian_neumann_fd_3d(u_tau, dx)
    mu_tau = -eps2 * lap_u_tau + (u_tau ** 3 - u_tau)
    lap_mu_tau = laplacian_neumann_fd_3d(mu_tau, dx)

    R = ut - lap_mu_tau

    # CH residual must be mean-free
    R = R - R.mean(dim=(1, 2, 3), keepdim=True)

    if normalize:
        s_ut = torch.sqrt(tetra_gauss_l2_scalar(ut, dx).detach() + 1e-8)        # (B,)
        s_lm = torch.sqrt(tetra_gauss_l2_scalar(lap_mu_tau, dx).detach() + 1e-8) # (B,)
        scale = 0.5 * (s_ut + s_lm) + 1e-8
        R = R / scale.view(-1, 1, 1, 1)

    if robust:
        loss_res = tetra_gauss_integral_scalar(torch.sqrt(R * R + eps), dx).mean()
    else:
        loss_res = tetra_gauss_l2_scalar(R, dx).mean()



    return (loss_res ).to(u_next.dtype)


def mixed_form_CH_physical_gauss_rollout(
    u_hist_last,
    u_pred,
    dt,
    dx,
    eps2,
    tau=0.5,
    normalize=True,
    robust=False,
    mass_weight=0.25,
    end_weight=1.0,
    eps=1e-8,
):
    """
    Mixed-form CH physical-space tetrahedral Gauss loss over a rollout block.

    u_hist_last : (B,S,S,S,1)
    u_pred      : (B,S,S,S,T)

    returns: scalar tensor
    """
    if u_hist_last.dim() != 5 or u_hist_last.shape[-1] != 1:
        raise ValueError("u_hist_last must have shape (B,S,S,S,1)")
    if u_pred.dim() != 5:
        raise ValueError("u_pred must have shape (B,S,S,S,T)")

    T = u_pred.shape[-1]
    u_prev = u_hist_last
    losses = []

    w_t = torch.linspace(
        1.0,
        float(end_weight),
        T,
        device=u_pred.device,
        dtype=u_pred.dtype,
    )

    for t in range(T):
        u_next = u_pred[..., t:t+1]

        l_step = mixed_form_CH_physical_gauss_single_step(
            u_prev,
            u_next,
            dt,
            dx,
            eps2,
            tau=tau,
            normalize=normalize,
            robust=robust,
            mass_weight=mass_weight,
            eps=eps,
        )

        losses.append(l_step)
        u_prev = u_next

    losses = torch.stack(losses)  # (T,)
    return ((w_t * losses).sum() / w_t.sum()).to(u_pred.dtype)



import torch
import torch.nn.functional as F


def grad_neumann_fd_3d(u, dx):
    """
    Central-difference gradient with homogeneous Neumann BC via replicate padding.

    u: (B,Sx,Sy,Sz)
    returns: ux, uy, uz each of shape (B,Sx,Sy,Sz)
    """
    if u.dim() != 4:
        raise ValueError("u must have shape (B,Sx,Sy,Sz)")

    up = F.pad(u.unsqueeze(1), (1, 1, 1, 1, 1, 1), mode="replicate").squeeze(1)

    ux = (up[:, 2:, 1:-1, 1:-1] - up[:, :-2, 1:-1, 1:-1]) / (2.0 * dx)
    uy = (up[:, 1:-1, 2:, 1:-1] - up[:, 1:-1, :-2, 1:-1]) / (2.0 * dx)
    uz = (up[:, 1:-1, 1:-1, 2:] - up[:, 1:-1, 1:-1, :-2]) / (2.0 * dx)

    return ux, uy, uz


def tetra_gauss_integral_scalar(field, dx):
    """
    Integrate a scalar nodal field over the domain using:
      - 6 tetrahedra per cube
      - 1-point Gauss quadrature at each tetra centroid

    field: (B,Sx,Sy,Sz)

    returns: (B,)
    """
    centroids = _tet6_centroid_values(field)          # (B,Sx-1,Sy-1,Sz-1,6)
    tet_volume = (dx ** 3) / 6.0
    return tet_volume * centroids.sum(dim=(1, 2, 3, 4))


def free_energy_density_CH_physical(u, dx, eps2):
    """
    CH free-energy density in physical space:

        f(u) = eps2/2 * |grad u|^2 + 1/4 * (u^2 - 1)^2

    u: (B,Sx,Sy,Sz)
    returns: (B,Sx,Sy,Sz)
    """
    ux, uy, uz = grad_neumann_fd_3d(u, dx)
    grad_sq = ux * ux + uy * uy + uz * uz
    potential = 0.25 * (u * u - 1.0) ** 2
    return 0.5 * eps2 * grad_sq + potential


def total_free_energy_CH_physical_gauss(u, dx, eps2):
    """
    Total CH free energy using physical-space FEM-style tetrahedral Gauss integration.

    u: (B,S,S,S,1)
    returns: (B,)
    """
    if u.dim() != 5 or u.shape[-1] != 1:
        raise ValueError("u must have shape (B,S,S,S,1)")

    uf = u.squeeze(-1).float()
    f = free_energy_density_CH_physical(uf, dx, eps2)
    return tetra_gauss_integral_scalar(f, dx).to(u.dtype)


def mixed_energy_CH_physical_gauss_single_step(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    tau=0.5,
    relative=True,
    robust=True,
    increase_weight=0.25,
    mass_weight=0.10,
    eps=1e-8,
):
    """
    Mixed-form CH energy loss in physical space with FEM-style tetrahedral Gauss integration.

    Mixed CH:
        u_t = Delta(mu)
        mu  = -eps2 * Delta(u) + (u^3 - u)

    Continuous energy law:
        dF/dt = - ∫ |grad(mu)|^2 dV

    Discrete single-step penalty:
        balance = F(u_next) - F(u_prev) + dt * ∫ |grad(mu_tau)|^2 dV

    plus:
        - hinge penalty for energy increase
        - exact mass conservation penalty

    Inputs
    ------
    u_prev, u_next : (B,S,S,S,1)

    Returns
    -------
    scalar tensor
    """
    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()
    u1 = u_next.squeeze(-1).float()

    # midpoint / collocation state
    u_tau = (1.0 - tau) * u0 + tau * u1

    # mixed chemical potential in physical space
    lap_u_tau = laplacian_neumann_fd_3d(u_tau, dx)
    mu_tau = -eps2 * lap_u_tau + (u_tau**3 - u_tau)

    # dissipation density = |grad mu|^2
    mux, muy, muz = grad_neumann_fd_3d(mu_tau, dx)
    diss_density = mux * mux + muy * muy + muz * muz

    # integrated dissipation
    diss = tetra_gauss_integral_scalar(diss_density, dx)   # (B,)

    # free energies
    F0 = total_free_energy_CH_physical_gauss(u_prev, dx, eps2)  # (B,)
    F1 = total_free_energy_CH_physical_gauss(u_next, dx, eps2)  # (B,)

    # discrete energy law:
    # F1 - F0 + dt * Diss ≈ 0
    balance = F1 - F0 + dt * diss

    if relative:
        scale = F0.abs().detach() + dt * diss.detach() + 1e-6
        balance = balance / scale
        inc = torch.relu(F1 - F0) / scale
    else:
        inc = torch.relu(F1 - F0)

    if robust:
        loss_balance = torch.sqrt(balance * balance + eps)
        loss_inc = torch.sqrt(inc * inc + eps)
    else:
        loss_balance = balance * balance
        loss_inc = inc * inc

    # exact mass conservation
    m0 = u0.mean(dim=(1, 2, 3))
    m1 = u1.mean(dim=(1, 2, 3))
    mass_err = (m1 - m0) ** 2

    if robust:
        loss_mass = torch.sqrt(mass_err + eps)
    else:
        loss_mass = mass_err

    return (
        loss_balance
        + increase_weight * loss_inc
        + mass_weight * loss_mass
    ).mean().to(u_next.dtype)




'''''
def mixed_form_AC_physical_gauss_single_step(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    tau=0.5,
    normalize=True,
    robust=False,
    mass_weight=0.0,
    eps=1e-8,
):
    """
    Mixed-form AC least-squares residual in physical space (FEM-style Gauss).

    AC:
        u_t = -mu
        mu  = -eps2 * Delta(u) + (u^3 - u)

    Residual:
        R = u_t + mu_tau
    """

    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()
    u1 = u_next.squeeze(-1).float()

    # time derivative
    ut = (u1 - u0) / dt

    # midpoint state
    u_tau = (1.0 - tau) * u0 + tau * u1

    # chemical potential (physical space FD)
    lap_u_tau = laplacian_neumann_fd_3d(u_tau, dx)
    mu_tau = -eps2 * lap_u_tau + (u_tau**3 - u_tau)

    # mixed residual
    R = ut + mu_tau

    # -------- normalization (FIXED) --------
    if normalize:
        s_ut = ut.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
        s_mu = mu_tau.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
        scale = 0.5 * (s_ut + s_mu)
        R = R / scale

    # -------- FEM Gauss integration --------
    if robust:
        loss_res = tetra_gauss_integral_scalar(torch.sqrt(R * R + eps), dx).mean()
    else:
        loss_res = tetra_gauss_l2_scalar(R, dx).mean()

    # -------- optional mean drift penalty (NOT needed for AC) --------
    if mass_weight > 0.0:
        m0 = u0.mean(dim=(1, 2, 3))
        m1 = u1.mean(dim=(1, 2, 3))
        mean_err = (m1 - m0) ** 2

        if robust:
            loss_mean = torch.sqrt(mean_err + eps).mean()
        else:
            loss_mean = mean_err.mean()

        return (loss_res + mass_weight * loss_mean).to(u_next.dtype)

    return loss_res.to(u_next.dtype)
'''

def mixed_form_AC_physical_gauss_single_step(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    tau=0.5,
    normalize=False,
    robust=True,
    mass_weight=0.0,
    eps=1e-8,
    scale_by_state=True,
):
    """
    Physical-space least-squares residual for AC, single step, no order reduction.

    Allen–Cahn:
        u_t = eps2 * Delta(u) - (u^3 - u)

    Residual at collocation state:
        R_ac = (u_next - u_prev)/dt - eps2 * Delta(u_tau) + (u_tau^3 - u_tau)

    Compared with the earlier version, this one is better for training because:
      - it does NOT normalize by ut/rhs at every sample by default
      - it uses a robust Charbonnier penalty
      - optional scaling uses the state magnitude instead of residual magnitude

    Inputs
    ------
    u_prev, u_next : (B,S,S,S,1)

    Returns
    -------
    scalar tensor
    """
    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()   # (B,S,S,S)
    u1 = u_next.squeeze(-1).float()   # (B,S,S,S)

    # time derivative
    ut = (u1 - u0) / dt

    # collocation / midpoint state
    u_tau = (1.0 - tau) * u0 + tau * u1

    # direct AC residual in physical space
    lap_u_tau = laplacian_neumann_fd_3d(u_tau, dx)
    R = ut - eps2 * lap_u_tau + (u_tau**3 - u_tau)

    # optional gentle normalization using state size only
    # much better behaved for AC than normalizing by ut/rhs themselves
    if normalize:
        if scale_by_state:
            s_state = torch.sqrt(tetra_gauss_l2_scalar(u_tau, dx).detach() + 1e-8)   # (B,)
            scale = s_state + 1e-6
        else:
            scale = torch.ones(u_tau.shape[0], device=u_tau.device, dtype=u_tau.dtype)

        R = R / scale.view(-1, 1, 1, 1)

    # robust residual penalty
    if robust:
        # Charbonnier-type FEM integral
        loss_res = tetra_gauss_integral_scalar(torch.sqrt(R * R + eps), dx).mean()
    else:
        loss_res = tetra_gauss_l2_scalar(R, dx).mean()

    # optional mean-drift penalty only if explicitly wanted
    if mass_weight > 0.0:
        m0 = u0.mean(dim=(1, 2, 3))
        m1 = u1.mean(dim=(1, 2, 3))
        mean_err = (m1 - m0) ** 2

        if robust:
            loss_mean = torch.sqrt(mean_err + eps).mean()
        else:
            loss_mean = mean_err.mean()

        return (loss_res + mass_weight * loss_mean).to(u_next.dtype)

    return loss_res.to(u_next.dtype)


def energy_AC_gauss(u, dx, eps2):
    """
    Allen–Cahn energy computed using tetrahedral Gauss quadrature.

    E[u] = ∫ ( eps2/2 * |∇u|^2 + 1/4 * (u^2 - 1)^2 ) dV

    Uses:
    - linear tetrahedra (P1)
    - exact gradient inside tetra
    - centroid quadrature for potential
    """

    if u.dim() != 5 or u.shape[-1] != 1:
        raise ValueError("u must have shape (B,S,S,S,1)")

    u = u.squeeze(-1).float()  # (B,S,S,S)

    # ----------------------------------
    # Build tetrahedra nodal values
    # ----------------------------------
    u_tets = _cube_to_tetrahedra(u)   # (B, Ntets, 4)

    # ----------------------------------
    # Predefined gradients of shape functions
    # Reference tetra (unit cube split)
    # ----------------------------------
    # grad(phi_i) for linear tetra
    # These are constant for each tetra
    grad_phi = torch.tensor([
        [-1, -1, -1],
        [ 1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0,  1],
    ], dtype=u.dtype, device=u.device)  # (4,3)

    # Scale by dx
    grad_phi = grad_phi / dx

    # ----------------------------------
    # Compute ∇u inside each tetra
    # ∇u = sum u_i * ∇phi_i
    # ----------------------------------
    # u_tets: (B, Ntets, 4)
    # grad_phi: (4,3)

    grad_u = torch.einsum('bti,ij->btj', u_tets, grad_phi)  # (B, Ntets, 3)

    grad_sq = (grad_u ** 2).sum(dim=-1)  # (B, Ntets)

    # ----------------------------------
    # Potential at centroid
    # ----------------------------------
    u_centroid = u_tets.mean(dim=-1)  # (B, Ntets)

    potential = 0.25 * (u_centroid**2 - 1.0)**2

    # ----------------------------------
    # Combine energy density
    # ----------------------------------
    energy_density = 0.5 * eps2 * grad_sq + potential

    # ----------------------------------
    # Integrate over tetrahedra
    # ----------------------------------
    tet_volume = (dx ** 3) / 6.0

    energy = tet_volume * energy_density.mean()

    return energy.to(u.dtype)


def ac_physical_gauss_single_step(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    normalize=True,
    robust=False,
    eps=1e-8,
):
    """
    Strong-form Allen–Cahn residual in physical space (finite-difference),
    integrated using tetrahedral Gauss quadrature.

    PDE:
        u_t = Δu - (1/eps2)(u^3 - u)

    Residual:
        R = (u_next - u_prev)/dt - [Δu_next - (1/eps2)(u_next^3 - u_next)]

    Inputs
    ------
    u_prev, u_next : (B,S,S,S,1)

    Returns
    -------
    scalar tensor
    """

    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()   # (B,S,S,S)
    u1 = u_next.squeeze(-1).float()   # (B,S,S,S)

    # time derivative
    ut = (u1 - u0) / dt

    # diffusion (FD Laplacian with Neumann BC)
    lap_u1 = laplacian_neumann_fd_3d(u1, dx)

    # nonlinear reaction
    reaction = (u1 ** 3 - u1) / eps2

    # residual
    R = ut - (lap_u1 - reaction)

    # optional normalization (same spirit as CH)
    if normalize:
        s_ut = torch.sqrt(tetra_gauss_l2_scalar(ut, dx).detach() + eps)
        s_rhs = torch.sqrt(tetra_gauss_l2_scalar(lap_u1 - reaction, dx).detach() + eps)
        scale = 0.5 * (s_ut + s_rhs) + eps
        R = R / scale.view(-1, 1, 1, 1)

    # integrate residual
    if robust:
        loss = tetra_gauss_integral_scalar(torch.sqrt(R * R + eps), dx).mean()
    else:
        loss = tetra_gauss_l2_scalar(R, dx).mean()

    return loss.to(u_next.dtype)



# ============================================================
# Exact AC free energy on a tetrahedral background mesh
# - physical-space FE-style integration
# - linear tetrahedra
# - exact constant gradient term
# - exact quartic potential integral for linear shape functions
# - fully vectorized (no expensive Python loops over elements)
# ============================================================

_TETRA_GRAD_CACHE = {}


def _cube_to_tetrahedra_all(u):
    """
    Valid 6-tetra decomposition of each cube using the body diagonal
    from v000 to v111.

    Parameters
    ----------
    u : torch.Tensor
        Shape (B, Sx, Sy, Sz)

    Returns
    -------
    torch.Tensor
        Shape (B, Sx-1, Sy-1, Sz-1, 6, 4)
    """
    if u.dim() != 4:
        raise ValueError("u must have shape (B,Sx,Sy,Sz)")

    v000 = u[:, :-1, :-1, :-1]
    v100 = u[:,  1:, :-1, :-1]
    v010 = u[:, :-1,  1:, :-1]
    v110 = u[:,  1:,  1:, :-1]
    v001 = u[:, :-1, :-1,  1:]
    v101 = u[:,  1:, :-1,  1:]
    v011 = u[:, :-1,  1:,  1:]
    v111 = u[:,  1:,  1:,  1:]

    tet1 = torch.stack([v000, v100, v110, v111], dim=-1)
    tet2 = torch.stack([v000, v100, v101, v111], dim=-1)
    tet3 = torch.stack([v000, v001, v101, v111], dim=-1)
    tet4 = torch.stack([v000, v001, v011, v111], dim=-1)
    tet5 = torch.stack([v000, v010, v011, v111], dim=-1)
    tet6 = torch.stack([v000, v010, v110, v111], dim=-1)

    return torch.stack([tet1, tet2, tet3, tet4, tet5, tet6], dim=-2)
    # shape: (B, Sx-1, Sy-1, Sz-1, 6, 4)


def _tetra_grad_shape_families(dx, device, dtype):
    """
    Gradients of linear tetrahedral shape functions in physical space
    for the valid 6-tetra cube decomposition used in _cube_to_tetrahedra_all.

    Returns
    -------
    grad_phi : (6, 4, 3)
    volume   : (6,)
    """
    key = (float(dx), str(device), str(dtype))
    if key in _TETRA_GRAD_CACHE:
        return _TETRA_GRAD_CACHE[key]

    X = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # v000 -> 0
            [1.0, 0.0, 0.0],  # v100 -> 1
            [0.0, 1.0, 0.0],  # v010 -> 2
            [1.0, 1.0, 0.0],  # v110 -> 3
            [0.0, 0.0, 1.0],  # v001 -> 4
            [1.0, 0.0, 1.0],  # v101 -> 5
            [0.0, 1.0, 1.0],  # v011 -> 6
            [1.0, 1.0, 1.0],  # v111 -> 7
        ],
        device=device,
        dtype=dtype
    ) * dx

    tet_ids = torch.tensor(
        [
            [0, 1, 3, 7],  # [v000, v100, v110, v111]
            [0, 1, 5, 7],  # [v000, v100, v101, v111]
            [0, 4, 5, 7],  # [v000, v001, v101, v111]
            [0, 4, 6, 7],  # [v000, v001, v011, v111]
            [0, 2, 6, 7],  # [v000, v010, v011, v111]
            [0, 2, 3, 7],  # [v000, v010, v110, v111]
        ],
        device=device
    )

    tet_xyz = X[tet_ids]  # (6,4,3)

    grad_ref = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [ 1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0,  0.0,  1.0],
        ],
        device=device,
        dtype=dtype
    )  # (4,3)

    x0 = tet_xyz[:, 0, :]
    x1 = tet_xyz[:, 1, :]
    x2 = tet_xyz[:, 2, :]
    x3 = tet_xyz[:, 3, :]

    J = torch.stack([x1 - x0, x2 - x0, x3 - x0], dim=-1)   # (6,3,3)
    invJT = torch.linalg.inv(J).transpose(-1, -2)          # (6,3,3)

    grad_phi = torch.einsum("fij,aj->fai", invJT, grad_ref)  # (6,4,3)
    volume = torch.abs(torch.linalg.det(J)) / 6.0            # (6,)

    _TETRA_GRAD_CACHE[key] = (grad_phi, volume)
    return grad_phi, volume


def energy_AC_tetra_exact(u, dx, eps2):
    """
    Exact Allen-Cahn free energy on a tetrahedral background mesh.

    E[u] = ∫ [ eps2/2 * |grad u|^2 + 1/4 * (u^2 - 1)^2 ] dV

    Assumptions
    -----------
    - u is represented by linear shape functions on each tetrahedron
    - gradient term is exact because grad(u) is constant on each tetrahedron
    - quartic potential is integrated exactly using simplex polynomial moments

    Parameters
    ----------
    u : torch.Tensor
        Shape (B,S,S,S,1)
    dx : float
    eps2 : float

    Returns
    -------
    torch.Tensor
        Shape (B,)
        Per-sample free energy.
    """
    if u.dim() != 5 or u.shape[-1] != 1:
        raise ValueError("u must have shape (B,S,S,S,1)")

    uf = u.squeeze(-1).float()  # (B,S,S,S)

    # nodal values on all tetrahedra:
    # (B, Cx, Cy, Cz, 6, 4)
    U = _cube_to_tetrahedra_all(uf)

    grad_phi, volume = _tetra_grad_shape_families(dx, uf.device, uf.dtype)
    # grad_phi: (6,4,3)
    # volume  : (6,)

    # ------------------------------------------------------------
    # Exact gradient energy
    # ------------------------------------------------------------
    # grad_u on each tetra is constant:
    # grad_u = sum_a U_a * grad_phi_a
    grad_u = torch.einsum("bxyzkf,kfi->bxyzki", U, grad_phi)  # (B,Cx,Cy,Cz,6,3)
    grad_sq = (grad_u ** 2).sum(dim=-1)                        # (B,Cx,Cy,Cz,6)

    grad_energy = 0.5 * eps2 * grad_sq * volume.view(1, 1, 1, 1, 6)

    # ------------------------------------------------------------
    # Exact potential integral for linear tetrahedron
    # ------------------------------------------------------------
    u0 = U[..., 0]
    u1 = U[..., 1]
    u2 = U[..., 2]
    u3 = U[..., 3]

    # ---- integral of u^2 ----
    s2 = u0*u0 + u1*u1 + u2*u2 + u3*u3
    p2 = (
        u0*u1 + u0*u2 + u0*u3 +
        u1*u2 + u1*u3 + u2*u3
    )
    int_u2 = volume.view(1,1,1,1,6) * (s2 + p2) / 10.0

    # ---- integral of u^4 ----
    s4 = u0**4 + u1**4 + u2**4 + u3**4

    s31 = (
        u0**3*(u1+u2+u3) +
        u1**3*(u0+u2+u3) +
        u2**3*(u0+u1+u3) +
        u3**3*(u0+u1+u2)
    )

    s22 = (
        u0**2*u1**2 + u0**2*u2**2 + u0**2*u3**2 +
        u1**2*u2**2 + u1**2*u3**2 + u2**2*u3**2
    )

    s211 = (
        u0**2*(u1*u2 + u1*u3 + u2*u3) +
        u1**2*(u0*u2 + u0*u3 + u2*u3) +
        u2**2*(u0*u1 + u0*u3 + u1*u3) +
        u3**2*(u0*u1 + u0*u2 + u1*u2)
    )

    s1111 = u0*u1*u2*u3

    int_u4 = volume.view(1,1,1,1,6) * (s4 + s31 + s22 + s211 + s1111) / 35.0

    # ∫ 1 dV
    int_one = volume.view(1,1,1,1,6)

    potential_energy = 0.25 * (int_u4 - 2.0 * int_u2 + int_one)

    # ------------------------------------------------------------
    # Total energy per sample
    # ------------------------------------------------------------
    total = grad_energy + potential_energy  # (B,Cx,Cy,Cz,6)
    E = total.sum(dim=(1, 2, 3, 4))         # (B,)

    return E.to(u.dtype)


def ac_energy_decay_loss_tetra_exact(
    u_prev,
    u_next,
    dx,
    eps2,
    relative=True,
    robust=True,
    margin=0.0,
    eps=1e-8,
):
    """
    Pure non-teacher AC energy decay loss.

    Penalizes only violations of the Allen-Cahn energy law:
        E(u_next) <= E(u_prev)

    Parameters
    ----------
    u_prev, u_next : (B,S,S,S,1)
    relative : bool
        If True, normalize by |E_prev| to make the penalty scale-stable.
    robust : bool
        If True, use Charbonnier-type penalty instead of plain square.
    margin : float
        Small allowed tolerance before penalizing.

    Returns
    -------
    scalar tensor
    """
    E_prev = energy_AC_tetra_exact(u_prev, dx, eps2)  # (B,)
    E_next = energy_AC_tetra_exact(u_next, dx, eps2)  # (B,)

    viol = torch.relu(E_next - E_prev - margin)

    if relative:
        scale = E_prev.abs().detach() + 1e-6
        viol = viol / scale

    if robust:
        loss = torch.sqrt(viol * viol + eps)
    else:
        loss = viol * viol

    return loss.mean().to(u_next.dtype)



def energy_CH_tetra_exact(u, dx, eps2):
    """
    Exact Cahn–Hilliard free energy on a tetrahedral background mesh.

    For the standard CH model used here, the free-energy functional is
        E[u] = ∫ [ eps2/2 * |grad u|^2 + 1/4 * (u^2 - 1)^2 ] dV

    This is the same phase-field free energy form used in the AC case.
    We keep a separate wrapper for clarity and manuscript consistency.

    Parameters
    ----------
    u : torch.Tensor
        Shape (B,S,S,S,1)
    dx : float
    eps2 : float

    Returns
    -------
    torch.Tensor
        Shape (B,)
        Per-sample free energy.
    """
    return energy_AC_tetra_exact(u, dx, eps2)


def ch_energy_decay_loss_tetra_exact(
    u_prev,
    u_next,
    dx,
    eps2,
    relative=True,
    robust=True,
    margin=0.0,
    eps=1e-8,
):
    """
    Pure non-teacher CH energy decay loss.

    Penalizes only violations of the Cahn–Hilliard energy law:
        E(u_next) <= E(u_prev)

    Parameters
    ----------
    u_prev, u_next : (B,S,S,S,1)
    relative : bool
        If True, normalize by |E_prev| to make the penalty scale-stable.
    robust : bool
        If True, use Charbonnier-type penalty instead of plain square.
    margin : float
        Small allowed tolerance before penalizing.

    Returns
    -------
    scalar tensor
    """
    E_prev = energy_CH_tetra_exact(u_prev, dx, eps2)  # (B,)
    E_next = energy_CH_tetra_exact(u_next, dx, eps2)  # (B,)

    viol = torch.relu(E_next - E_prev - margin)

    if relative:
        scale = E_prev.abs().detach() + 1e-6
        viol = viol / scale

    if robust:
        loss = torch.sqrt(viol * viol + eps)
    else:
        loss = viol * viol

    return loss.mean().to(u_next.dtype)



def l2_sq_tetra_linear(u, dx):
    """
    Exact integral of u^2 over the tetrahedral background mesh,
    assuming piecewise-linear interpolation on each tetrahedron.

    Parameters
    ----------
    u : torch.Tensor
        Shape (B,S,S,S,1)

    Returns
    -------
    torch.Tensor
        Shape (B,)
        Exact per-sample integral ∫ u^2 dx over the mesh.
    """
    if u.dim() != 5 or u.shape[-1] != 1:
        raise ValueError("u must have shape (B,S,S,S,1)")

    uf = u.squeeze(-1).float()  # (B,S,S,S)

    # nodal values on all tetrahedra
    # shape: (B, Cx, Cy, Cz, 6, 4)
    U = _cube_to_tetrahedra_all(uf)

    _, volume = _tetra_grad_shape_families(dx, uf.device, uf.dtype)
    vol = volume.view(1, 1, 1, 1, 6)

    u0 = U[..., 0]
    u1 = U[..., 1]
    u2 = U[..., 2]
    u3 = U[..., 3]

    s2 = u0*u0 + u1*u1 + u2*u2 + u3*u3
    p2 = (
        u0*u1 + u0*u2 + u0*u3 +
        u1*u2 + u1*u3 + u2*u3
    )

    # exact formula for linear tetra:
    # ∫ u^2 dx = V/10 * [sum ui^2 + sum_{i<j} ui uj]
    int_u2 = vol * (s2 + p2) / 10.0   # (B,Cx,Cy,Cz,6)

    return int_u2.sum(dim=(1, 2, 3, 4)).to(u.dtype)


def ac_incremental_energy_tetra_exact(u_prev, u_next, dt, dx, eps2):
    """
    Exact incremental Allen-Cahn energy functional on tetrahedral mesh:

        J(u_next ; u_prev)
        = 1/(2 dt) * ||u_next - u_prev||_L2^2 + E(u_next)

    Parameters
    ----------
    u_prev, u_next : torch.Tensor
        Shape (B,S,S,S,1)

    Returns
    -------
    torch.Tensor
        Shape (B,)
        Per-sample incremental energy.
    """
    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    du = u_next - u_prev

    move = 0.5 / dt * l2_sq_tetra_linear(du, dx)       # (B,)
    energy = energy_AC_tetra_exact(u_next, dx, eps2)   # (B,)

    return (move + energy).to(u_next.dtype)

def ac_incremental_energy_loss_tetra_exact(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    relative=True,
    robust=True,
    eps=1e-8,
):
    """
    Pure non-teacher incremental-energy loss for Allen-Cahn.

    This penalizes the incremental energy functional
        J(u_next ; u_prev)
    directly, using exact tetrahedral physical-space integration.

    Parameters
    ----------
    u_prev, u_next : torch.Tensor
        Shape (B,S,S,S,1)
    relative : bool
        If True, normalize by the previous-state energy magnitude.
    robust : bool
        If True, use sqrt(J^2 + eps) after normalization for stability.

    Returns
    -------
    scalar tensor
    """
    J = ac_incremental_energy_tetra_exact(u_prev, u_next, dt, dx, eps2)  # (B,)

    if relative:
        E_prev = energy_AC_tetra_exact(u_prev, dx, eps2).abs().detach() + 1e-6
        val = J / E_prev
    else:
        val = J

    if robust:
        loss = torch.sqrt(val * val + eps)
    else:
        loss = val * val

    return loss.mean().to(u_next.dtype)


'''''
def ch_incremental_energy_tetra_exact(u_prev, u_next, dt, dx, eps2):
    """
    CH incremental energy in the same style as the AC version:

        J(u_next ; u_prev)
        = 1/(2 dt) * ||u_next - u_prev||_L2^2 + E_CH(u_next)

    where E_CH is computed in physical space using tetrahedral Gauss integration.

    Parameters
    ----------
    u_prev, u_next : torch.Tensor
        Shape (B,S,S,S,1)

    Returns
    -------
    torch.Tensor
        Shape (B,)
        Per-sample incremental energy.
    """
    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    du = u_next - u_prev

    move = 0.5 / dt * l2_sq_tetra_linear(du, dx)                  # (B,)
    energy = total_free_energy_CH_physical_gauss(u_next, dx, eps2)  # (B,)

    return (move + energy).to(u_next.dtype)

'''


#
import torch
import torch.nn.functional as F


def l2_sq_tetra_linear(u, dx):
    """
    Exact tetrahedral L2 norm squared for a nodal field represented
    by linear shape functions on the tetrahedral background mesh.

    u : (B,S,S,S,1)
    returns : (B,)
    """
    if u.dim() != 5 or u.shape[-1] != 1:
        raise ValueError("u must have shape (B,S,S,S,1)")

    uf = u.squeeze(-1).float()  # (B,S,S,S)

    # nodal values on all tetrahedra
    U = _cube_to_tetrahedra_all(uf)  # (B,Cx,Cy,Cz,6,4)

    _, volume = _tetra_grad_shape_families(dx, uf.device, uf.dtype)  # volume: (6,)

    u0 = U[..., 0]
    u1 = U[..., 1]
    u2 = U[..., 2]
    u3 = U[..., 3]

    # exact integral of u^2 on linear tetrahedron
    s2 = u0*u0 + u1*u1 + u2*u2 + u3*u3
    p2 = (
        u0*u1 + u0*u2 + u0*u3 +
        u1*u2 + u1*u3 + u2*u3
    )

    int_u2 = volume.view(1, 1, 1, 1, 6) * (s2 + p2) / 10.0
    val = int_u2.sum(dim=(1, 2, 3, 4))  # (B,)

    return val.to(u.dtype)


def ac_incremental_energy_tetra_exact(u_prev, u_next, dt, dx, eps2):
    """
    True incremental Allen-Cahn energy functional:

        J(u_next ; u_prev)
        = 1/(2 dt) * ||u_next - u_prev||_L2^2 + E(u_next)

    u_prev, u_next : (B,S,S,S,1)
    returns        : (B,)
    """
    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    du = u_next - u_prev
    move = 0.5 / dt * l2_sq_tetra_linear(du, dx)       # (B,)
    energy = energy_AC_tetra_exact(u_next, dx, eps2)   # (B,)

    return (move + energy).to(u_next.dtype)


def ac_true_incremental_energy_target_tetra_exact(
    u_prev,
    dt,
    dx,
    eps2,
    u_init=None,
    n_inner=8,
    lr=5e-2,
    clamp_value=None,
):
    """
    Approximate the true variational minimizer

        u_star = argmin_u J(u ; u_prev)

    by inner gradient descent on the field itself.

    Parameters
    ----------
    u_prev : (B,S,S,S,1)
    u_init : optional initialization, same shape
    n_inner : number of inner optimization steps
    lr : inner descent step size
    clamp_value : optional clipping for stability

    Returns
    -------
    u_star : (B,S,S,S,1), detached
    """
    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")

    if u_init is None:
        u = u_prev.clone().detach()
    else:
        if u_init.shape != u_prev.shape:
            raise ValueError("u_init must have the same shape as u_prev")
        u = u_init.clone().detach()

    u.requires_grad_(True)

    for _ in range(n_inner):
        J = ac_incremental_energy_tetra_exact(u_prev, u, dt, dx, eps2).mean()
        grad, = torch.autograd.grad(J, u, create_graph=False)

        with torch.no_grad():
            u -= lr * grad
            if clamp_value is not None:
                u.clamp_(-clamp_value, clamp_value)

        u.requires_grad_(True)

    return u.detach()


def ac_true_incremental_energy_matching_loss(
    u_prev,
    u_pred,
    dt,
    dx,
    eps2,
    n_inner=8,
    lr=5e-2,
    init_with_pred=True,
    clamp_value=None,
):
    """
    Loss that compares the NN prediction to the true incremental-energy minimizer.

    u_star = argmin_u J(u ; u_prev)

    Then:
        loss = MSE(u_pred, u_star)

    This is much closer to the true incremental-energy formulation than
    directly penalizing J(u_pred ; u_prev).

    Parameters
    ----------
    u_prev, u_pred : (B,S,S,S,1)

    Returns
    -------
    scalar tensor
    """
    if u_pred.dim() != 5 or u_pred.shape[-1] != 1:
        raise ValueError("u_pred must have shape (B,S,S,S,1)")

    u_init = u_pred.detach() if init_with_pred else u_prev.detach()

    u_star = ac_true_incremental_energy_target_tetra_exact(
        u_prev=u_prev,
        dt=dt,
        dx=dx,
        eps2=eps2,
        u_init=u_init,
        n_inner=n_inner,
        lr=lr,
        clamp_value=clamp_value,
    )

    return F.mse_loss(u_pred, u_star)

### AC non-periodic PENCO
def _rhs_ac3d_neumann(u, dx, eps2):
    # AC RHS with homogeneous Neumann BC:
    # u_t = Δu - 1/eps2 * (u^3 - u)
    lap_u = laplacian_neumann_3d_phys(u, dx)
    return lap_u - (1.0 / eps2) * (u ** 3 - u)


def physics_collocation_tau_L2_AC_neumann(u_in, u_pred, tau=0.5 - 1.0 / (2.0 * math.sqrt(5.0)), normalize=True):
    """
    AC3D Neumann collocation at u_tau:
        R_tau = (u^{n+1}-u^n)/dt - RHS_AC_Neumann(u_tau)

    Uses DCT/cosine Neumann Laplacian instead of periodic FFT Laplacian.
    """
    assert config.PROBLEM == 'AC3D'
    dt, dx = config.DT, config.DX

    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()

    ut = (up - u0) / dt
    u_tau = (1.0 - tau) * u0 + tau * up

    rhs_tau = _rhs_ac3d_neumann(u_tau, dx, config.EPS2)

    if normalize:
        s_t = ut.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau

    return (R ** 2).mean().to(u_pred.dtype)


def semi_implicit_step_AC_neumann_simple(u_in, dt, dx, eps2):
    """
    AC3D semi-implicit Neumann teacher step using DCT:

        (u^{n+1} - u^n)/dt = Δu^{n+1} - 1/eps2 * ((u^n)^3 - u^n)

    In cosine space:
        U1 = (U0 - dt/eps2 * DCT(u0^3 - u0)) / (1 + dt*k2)
    """
    with torch.amp.autocast(device_type='cuda', enabled=False):
        u0 = u_in.squeeze(-1).float()
        B, nx, ny, nz = u0.shape

        k2 = _neumann_k2_3d(nx, ny, nz, dx, device=u0.device, dtype=u0.dtype)

        nl = u0 ** 3 - u0

        U0 = dct3_neumann(u0)
        NL = dct3_neumann(nl)

        num = U0 - (dt / eps2) * NL
        den = 1.0 + dt * k2.unsqueeze(0)

        U1 = num / (den + 1e-12)
        u1 = idct3_neumann(U1).real

    return u1.unsqueeze(-1).to(u_in.dtype)


def energy_density_neumann(u, dx, eps2):
    """
    AC free-energy density with homogeneous Neumann BC:

        E(u) = eps2/2 * |grad u|^2 + 1/4 * (u^2 - 1)^2

    Uses Neumann finite-difference gradient.
    """
    grad2 = _neumann_grad2(u, dx)
    potential = 0.25 * (u ** 2 - 1.0) ** 2
    interface = 0.5 * eps2 * grad2
    return interface + potential


def energy_penalty_AC_neumann(u_in, u_pred, dx, eps2):
    """
    One-sided AC energy hinge for Neumann BC.
    Penalizes only if E(u^{n+1}) > E(u^n).
    """
    u0 = u_in.squeeze(-1)
    up = u_pred.squeeze(-1)

    E0 = energy_density_neumann(u0, dx, eps2).mean(dim=(1, 2, 3))
    Ep = energy_density_neumann(up, dx, eps2).mean(dim=(1, 2, 3))

    inc = torch.relu(Ep - E0)
    return inc.mean()


#### CH non-periodic PENCO

def _rhs_ch3d_neumann(u, dx, eps):
    """
    CH3D RHS with homogeneous Neumann BC:
        u_t = -Δ[2u + eps^2 Δu + (u^3 - 3u)]

    Uses DCT/cosine Neumann Laplacian instead of periodic FFT Laplacian.
    """
    lap_u = laplacian_neumann_3d_phys(u, dx)
    chem = u ** 3 - 3.0 * u
    return -laplacian_neumann_3d_phys(2.0 * u + (eps ** 2) * lap_u + chem, dx)


def physics_collocation_tau_L2_CH_neumann(u_in, u_pred, tau=0.5 - 1.0 / (2.0 * math.sqrt(5.0)), normalize=True):
    """
    CH3D Neumann L2 collocation:
        R_tau = (u^{n+1}-u^n)/dt - RHS_CH_Neumann(u_tau)
    """
    assert config.PROBLEM == 'CH3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM

    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()

    ut = (up - u0) / dt
    u_tau = (1.0 - tau) * u0 + tau * up

    rhs_tau = _rhs_ch3d_neumann(u_tau, dx, eps)

    if normalize:
        s_t = ut.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau

    return (R ** 2).mean().to(u_pred.dtype)


def semi_implicit_step_ch_neumann(u_in, dt, dx, eps):
    """
    CH3D semi-implicit Neumann teacher step.

    Periodic version:
        (1 + dt*(2k² + eps²k⁴)) U^{n+1}
        = U^n - dt*k²*FFT((u^n)^3 - 3u^n)

    Neumann version:
        replace FFT/IFFT with DCT/IDCT and use cosine k².
    """
    with torch.amp.autocast(device_type='cuda', enabled=False):
        u0 = u_in.squeeze(-1).float()
        B, nx, ny, nz = u0.shape

        k2 = _neumann_k2_3d(nx, ny, nz, dx, device=u0.device, dtype=u0.dtype)

        U0 = dct3_neumann(u0)
        chem0_hat = dct3_neumann(u0 ** 3 - 3.0 * u0)

        denom = 1.0 + dt * (2.0 * k2 + (eps ** 2) * (k2 ** 2))
        numer = U0 - dt * k2.unsqueeze(0) * chem0_hat

        U1 = numer / (denom.unsqueeze(0) + 1e-12)
        u1 = idct3_neumann(U1).real

    return u1.unsqueeze(-1).to(u_in.dtype)


def energy_density_CH_neumann(u, dx, eps):
    """
    CH free-energy density consistent with:
        mu = eps^2 Δu + u^3 - u

    Energy:
        E(u) = 1/4*(u^2 - 1)^2 - eps^2/2*|grad u|^2

    This follows your CH sign convention:
        u_t = -Δ[eps^2 Δu + u^3 - u]
    """
    grad2 = _neumann_grad2(u, dx)
    potential = 0.25 * (u ** 2 - 1.0) ** 2
    gradient = -0.5 * (eps ** 2) * grad2
    return potential + gradient


def energy_penalty_CH_neumann(u_in, u_pred, dx, eps):
    """
    One-sided CH energy hinge for homogeneous Neumann BC.
    Penalizes only if E(u^{n+1}) > E(u^n).
    """
    u0 = u_in.squeeze(-1)
    up = u_pred.squeeze(-1)

    E0 = energy_density_CH_neumann(u0, dx, eps).mean(dim=(1, 2, 3))
    Ep = energy_density_CH_neumann(up, dx, eps).mean(dim=(1, 2, 3))

    inc = torch.relu(Ep - E0)
    return inc.mean()



###### SH non periodic - PENCO

# ============================================================
# SH3D NON-PERIODIC / HOMOGENEOUS NEUMANN HELPERS
# Requires existing:
#   dct3_neumann(u)
#   idct3_neumann(uhat)
# ============================================================

def _neumann_k2_3d(nx, ny, nz, dx, device, dtype=torch.float32):
    """
    Neumann cosine-basis eigenvalues:
        Delta phi_pqr = -k2sum phi_pqr
    with Lx = nx*dx, Ly = ny*dx, Lz = nz*dx.
    """
    Lx = nx * dx
    Ly = ny * dx
    Lz = nz * dx

    px = torch.arange(nx, device=device, dtype=dtype)
    py = torch.arange(ny, device=device, dtype=dtype)
    pz = torch.arange(nz, device=device, dtype=dtype)

    k2x = (math.pi * px / Lx) ** 2
    k2y = (math.pi * py / Ly) ** 2
    k2z = (math.pi * pz / Lz) ** 2

    k2x, k2y, k2z = torch.meshgrid(k2x, k2y, k2z, indexing="ij")
    return k2x + k2y + k2z


def laplacian_neumann_3d_phys(u, dx):
    """
    Neumann Laplacian using DCT:
        Delta u = idct3(-k2 * dct3(u))
    Shape:
        u: (B,S,S,S)
    """
    with torch.amp.autocast(device_type="cuda", enabled=False):
        u32 = u.float()
        B, nx, ny, nz = u32.shape

        k2 = _neumann_k2_3d(
            nx, ny, nz, dx,
            device=u32.device,
            dtype=u32.dtype,
        )

        U = dct3_neumann(u32)
        lap = idct3_neumann(-k2.unsqueeze(0) * U)

    return lap.to(u.dtype)


def biharmonic_neumann_3d_phys(u, dx):
    """
    Neumann biharmonic:
        Delta^2 u = idct3(k2^2 * dct3(u))
    """
    with torch.amp.autocast(device_type="cuda", enabled=False):
        u32 = u.float()
        B, nx, ny, nz = u32.shape

        k2 = _neumann_k2_3d(
            nx, ny, nz, dx,
            device=u32.device,
            dtype=u32.dtype,
        )

        U = dct3_neumann(u32)
        bi = idct3_neumann((k2 ** 2).unsqueeze(0) * U)

    return bi.to(u.dtype)


def physics_collocation_tau_L2_SH_neumann(
    u_in,
    u_pred,
    tau=0.5 - 1.0 / (2.0 * math.sqrt(5.0)),
    normalize=True,
):
    """
    SH3D Neumann collocation residual:
        R_tau = (u^{n+1}-u^n)/dt - RHS_SH(u_tau)

    RHS_SH(u):
        (1-eps)u - 2 Delta u - Delta^2 u - u^3

    Uses DCT Neumann Laplacian/biharmonic instead of FFT periodic operators.
    """
    assert config.PROBLEM == "SH3D"

    dt = config.DT
    dx = config.DX
    eps = config.EPSILON_PARAM

    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()

    ut = (up - u0) / dt
    u_tau = (1.0 - tau) * u0 + tau * up

    lap_u = laplacian_neumann_3d_phys(u_tau, dx)
    bi_u = biharmonic_neumann_3d_phys(u_tau, dx)

    rhs_tau = (1.0 - eps) * u_tau - 2.0 * lap_u - bi_u - u_tau ** 3

    if normalize:
        s_t = ut.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau

    return (R ** 2).mean().to(u_pred.dtype)


def semi_implicit_step_sh_neumann(u_in, dt, dx, eps_param):
    """
    One SH semi-implicit Neumann step matching your MATLAB non-periodic solver:

        u_hat = dct3(u)
        nonlinear_hat = dct3(u^3)

        rhs_hat = dct3(u/dt) - nonlinear_hat + 2*k2sum*u_hat

        v_hat = rhs_hat / (1/dt + (1-eps) + k2sum^2)

        u_next = idct3(v_hat)
    """
    with torch.amp.autocast(device_type="cuda", enabled=False):
        u0 = u_in.squeeze(-1).float()
        B, nx, ny, nz = u0.shape

        k2 = _neumann_k2_3d(
            nx, ny, nz, dx,
            device=u0.device,
            dtype=u0.dtype,
        )

        U0 = dct3_neumann(u0)
        U3 = dct3_neumann(u0 ** 3)

        rhs_hat = dct3_neumann(u0 / dt) - U3 + 2.0 * k2.unsqueeze(0) * U0

        denom = (
            1.0 / dt
            + (1.0 - float(eps_param))
            + (k2 ** 2)
        )

        VHat = rhs_hat / (denom.unsqueeze(0) + 1e-12)
        u1 = idct3_neumann(VHat).real

    return u1.unsqueeze(-1).to(u_in.dtype)


def _neumann_grad2(u, dx):
    """
    Simple homogeneous-Neumann finite-difference |grad u|^2.

    Interior: forward difference.
    Boundary: zero normal derivative.
    """
    gx = torch.zeros_like(u)
    gy = torch.zeros_like(u)
    gz = torch.zeros_like(u)

    gx[:, :-1, :, :] = (u[:, 1:, :, :] - u[:, :-1, :, :]) / dx
    gy[:, :, :-1, :] = (u[:, :, 1:, :] - u[:, :, :-1, :]) / dx
    gz[:, :, :, :-1] = (u[:, :, :, 1:] - u[:, :, :, :-1]) / dx

    return gx * gx + gy * gy + gz * gz


def sh_free_energy_density_neumann(u, dx, eps):
    """
    SH free-energy density with Neumann operators:

        f_SH(u) =
            - (1-eps)/2 * u^2
            - |grad u|^2
            + 1/2 * (Delta u)^2
            + 1/4 * u^4
    """
    grad2 = _neumann_grad2(u, dx)
    lap = laplacian_neumann_3d_phys(u, dx)

    bulk_quad = -0.5 * (1.0 - eps) * (u * u)
    grad_term = -grad2
    bih_term = 0.5 * (lap * lap)
    quartic = 0.25 * (u * u * u * u)

    return bulk_quad + grad_term + bih_term + quartic


def energy_penalty_sh_neumann(u_in, u_pred, dx, eps):
    """
    One-sided SH energy hinge for Neumann BC:
        penalize only if F(u^{n+1}) > F(u^n)
    """
    u0 = u_in.squeeze(-1)
    up = u_pred.squeeze(-1)

    F0 = sh_free_energy_density_neumann(u0, dx, eps).mean(dim=(1, 2, 3))
    Fp = sh_free_energy_density_neumann(up, dx, eps).mean(dim=(1, 2, 3))

    inc = torch.relu(Fp - F0)
    return inc.mean()



####


def _ch_incremental_energy_terms_tetra_exact(u_prev, u_next, dt, dx, eps2):
    """
    Internal helper for CH incremental-energy quantities in physical space.

    Returns
    -------
    balance : (B,)
        Discrete CH energy-law defect:
            F_next - F_prev + dt * ∫ |grad mu_tau|^2 dx
    mass_err : (B,)
        Exact mass drift penalty:
            (mean(u_next) - mean(u_prev))^2
    scale : (B,)
        Natural scaling for relative normalization.
    """
    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()   # (B,S,S,S)
    u1 = u_next.squeeze(-1).float()   # (B,S,S,S)

    # midpoint / collocation state
    u_tau = 0.5 * (u0 + u1)

    # chemical potential at midpoint in physical space
    lap_u_tau = laplacian_neumann_fd_3d(u_tau, dx)
    mu_tau = -eps2 * lap_u_tau + (u_tau**3 - u_tau)

    # dissipation density = |grad mu|^2
    mux, muy, muz = grad_neumann_fd_3d(mu_tau, dx)
    diss_density = mux * mux + muy * muy + muz * muz

    # integrated dissipation
    diss = tetra_gauss_integral_scalar(diss_density, dx)   # (B,)

    # CH free energies
    F_prev = total_free_energy_CH_physical_gauss(u_prev, dx, eps2)  # (B,)
    F_next = total_free_energy_CH_physical_gauss(u_next, dx, eps2)  # (B,)

    # discrete CH energy law defect:
    # F_next - F_prev + dt * diss ≈ 0
    balance = F_next - F_prev + dt * diss

    # exact mass conservation defect
    m_prev = u0.mean(dim=(1, 2, 3))
    m_next = u1.mean(dim=(1, 2, 3))
    mass_err = (m_next - m_prev) ** 2

    # natural scale for relative normalization
    scale = F_prev.abs().detach() + dt * diss.detach() + 1e-6

    return balance.to(u_next.dtype), mass_err.to(u_next.dtype), scale.to(u_next.dtype)


def ch_incremental_energy_tetra_exact(u_prev, u_next, dt, dx, eps2):
    """
    CH incremental-energy quantity in physical space.

    Unlike the AC version, this uses the CH discrete energy-law defect:
        J = F_next - F_prev + dt * ∫ |grad mu_tau|^2 dx

    This is the correct CH analogue in spirit.

    Returns
    -------
    torch.Tensor
        Shape (B,)
        Per-sample energy-law defect.
    """
    balance, _, _ = _ch_incremental_energy_terms_tetra_exact(
        u_prev, u_next, dt, dx, eps2
    )
    return balance


def ch_incremental_energy_loss_tetra_exact(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    relative=True,
    robust=True,
    eps=1e-8,
):
    """
    Pure non-teacher CH incremental-energy loss in physical space.

    This is a CH-specific version of the incremental-energy idea:
      - penalizes violation of the discrete CH energy law
      - includes exact mass conservation penalty
      - stays fully in physical space
      - uses vectorized tetrahedral integration

    Parameters
    ----------
    u_prev, u_next : (B,S,S,S,1)
    relative : bool
        If True, normalize by a natural energy/dissipation scale.
    robust : bool
        If True, use Charbonnier-type penalty.

    Returns
    -------
    scalar tensor
    """
    balance, mass_err, scale = _ch_incremental_energy_terms_tetra_exact(
        u_prev, u_next, dt, dx, eps2
    )

    if relative:
        balance = balance / scale

    if robust:
        loss_balance = torch.sqrt(balance * balance + eps)
        loss_mass = torch.sqrt(mass_err + eps)
    else:
        loss_balance = balance * balance
        loss_mass = mass_err

    # light mass term; can be tuned later if needed
    loss = loss_balance +  loss_mass # modified

    return loss.mean().to(u_next.dtype)

##### SH
# ============================================================
# SH3D: Swift-Hohenberg with homogeneous Neumann BCs
# Split:
#
#     p = Delta u
#
# Original:
#     u_t = epsilon*u - u^3 - (1 + Delta)^2 u
#
# Expanded:
#     u_t = (epsilon - 1)u - u^3 - 2 Delta u - Delta^2 u
#
# With p = Delta u:
#     p   = Delta u
#     u_t = (epsilon - 1)u - u^3 - 2p - Delta p
# ============================================================


def mixed_form_SH_physical_gauss_single_step(
    u_prev,
    u_next,
    dt,
    dx,
    epsilon_param,
    tau=0.5,
    normalize=True,
    robust=False,
    p_weight=1.0,
    helmholtz_reg=1e-4,
    eps=1e-8,
):
    """
    Strong-form split SH residual.

    Split variable:
        p = Delta u

    Evolution equation:
        u_t = (epsilon - 1)u - u^3 - 2p - Delta p

    To avoid the auxiliary residual being algebraically zero, construct two
    representations of p:

        p_aux  = Delta u

    and from the PDE:

        (2 + Delta)p = (epsilon - 1)u - u^3 - u_t

    so:

        p_evol = inverse(2 + Delta)[(epsilon - 1)u - u^3 - u_t]

    Exact SH solution satisfies:

        p_aux = p_evol
    """

    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()
    u1 = u_next.squeeze(-1).float()

    u_tau = (1.0 - tau) * u0 + tau * u1
    ut = (u1 - u0) / dt

    # ------------------------------------------------------------
    # Auxiliary representation 1:
    #     p_aux = Delta u
    # ------------------------------------------------------------
    lap_u = laplacian_neumann_fd_3d(u_tau, dx)
    p_aux = lap_u

    # ------------------------------------------------------------
    # Auxiliary representation 2:
    #     (2 + Delta)p_evol = (epsilon - 1)u - u^3 - u_t
    #
    # Neumann cosine space:
    #     Delta -> -k^2
    #     (2 + Delta) -> (2 - k^2)
    # ------------------------------------------------------------
    B, nx, ny, nz = u_tau.shape

    k2 = _neumann_k2_grid(nx, ny, nz, dx, u_tau.device, u_tau.dtype)
    helm = 2.0 - k2

    g = (epsilon_param - 1.0) * u_tau - u_tau ** 3 - ut

    g_hat = dct3_neumann(g)

    # Regularized inverse of (2 + Delta)
    p_evol_hat = g_hat * helm / (helm ** 2 + helmholtz_reg)
    p_evol = idct3_neumann(p_evol_hat)

    # ------------------------------------------------------------
    # Residual 1:
    #     p_aux should match p_evol
    # ------------------------------------------------------------
    R_p = p_aux - p_evol

    # ------------------------------------------------------------
    # Residual 2:
    #     u_t = (epsilon - 1)u - u^3 - 2p_aux - Delta p_aux
    # ------------------------------------------------------------
    lap_p_aux = laplacian_neumann_fd_3d(p_aux, dx)

    rhs_u = (
        (epsilon_param - 1.0) * u_tau
        - u_tau ** 3
        - 2.0 * p_aux
        - lap_p_aux
    )

    R_u = ut - rhs_u

    if normalize:
        s_ut = torch.sqrt(tetra_gauss_l2_scalar(ut, dx).detach() + eps)
        s_rhs = torch.sqrt(tetra_gauss_l2_scalar(rhs_u, dx).detach() + eps)
        scale_u = 0.5 * (s_ut + s_rhs) + eps
        R_u = R_u / scale_u.view(-1, 1, 1, 1)

        s_p_aux = torch.sqrt(tetra_gauss_l2_scalar(p_aux, dx).detach() + eps)
        s_p_evol = torch.sqrt(tetra_gauss_l2_scalar(p_evol, dx).detach() + eps)
        scale_p = 0.5 * (s_p_aux + s_p_evol) + eps
        R_p = R_p / scale_p.view(-1, 1, 1, 1)

    if robust:
        loss_u = tetra_gauss_integral_scalar(torch.sqrt(R_u * R_u + eps), dx).mean()
        loss_p = tetra_gauss_integral_scalar(torch.sqrt(R_p * R_p + eps), dx).mean()
    else:
        loss_u = tetra_gauss_l2_scalar(R_u, dx).mean()
        loss_p = tetra_gauss_l2_scalar(R_p, dx).mean()

    loss = loss_u + p_weight * loss_p

    return (
        loss.to(u_next.dtype),
        loss_u.to(u_next.dtype),
        loss_p.to(u_next.dtype),
    )


def sh_weak_fe_residual_tetra_neumann(
    u_prev,
    u_next,
    dt,
    dx,
    epsilon_param,
    tau=0.5,
    normalize=True,
    eps=1e-8,
):
    """
    Mixed weak-form FE residual for SH using:

        p = Delta u

        u_t = (epsilon - 1)u - u^3 - 2p - Delta p

    Weak block 1:

        ∫ u_t phi
        - ∫ [(epsilon - 1)u - u^3 - 2p] phi
        - ∫ grad(p) · grad(phi)
        = 0

    because:

        ∫ Delta p phi = - ∫ grad(p) · grad(phi)

    Weak block 2:

        p = Delta u

    Written as:

        p - Delta u = 0

    Weak form:

        ∫ p phi + ∫ grad(u) · grad(phi) = 0

    because:

        -∫ Delta u phi = ∫ grad(u) · grad(phi)
    """

    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()
    u1 = u_next.squeeze(-1).float()

    u_tau_grid = (1.0 - tau) * u0 + tau * u1

    # p = Delta u
    p_tau_grid = laplacian_neumann_fd_3d(u_tau_grid, dx)

    U0 = _cube_to_tetrahedra_all(u0)
    U1 = _cube_to_tetrahedra_all(u1)
    U_tau = _cube_to_tetrahedra_all(u_tau_grid)
    P_tau = _cube_to_tetrahedra_all(p_tau_grid)

    grad_phi, volume = _tetra_grad_shape_families(dx, u1.device, u1.dtype)
    vol = volume.view(1, 1, 1, 1, 6, 1)

    # ============================================================
    # Block 1: evolution equation
    # ============================================================

    Ut = (U1 - U0) / dt
    ut_cent = Ut.mean(dim=-1, keepdim=True)

    time_vec = ut_cent * (0.25 * vol)
    time_vec = time_vec.expand_as(U1)

    u_cent = U_tau.mean(dim=-1, keepdim=True)
    p_cent = P_tau.mean(dim=-1, keepdim=True)

    # source = (epsilon - 1)u - u^3 - 2p
    source_cent = (
        (epsilon_param - 1.0) * u_cent
        - u_cent ** 3
        - 2.0 * p_cent
    )

    source_vec = source_cent * (0.25 * vol)
    source_vec = source_vec.expand_as(U1)

    grad_p = torch.einsum("bxyzkf,kfi->bxyzki", P_tau, grad_phi)
    diff_p_vec = torch.einsum("bxyzki,kfi->bxyzkf", grad_p, grad_phi) * vol

    # residual:
    # u_t - source + Delta p = 0
    # weak Delta p gives - grad(p).grad(phi)
    r_u_vec = time_vec - source_vec - diff_p_vec

    # ============================================================
    # Block 2: p = Delta u
    # ============================================================

    p_vec = p_cent * (0.25 * vol)
    p_vec = p_vec.expand_as(U1)

    grad_u = torch.einsum("bxyzkf,kfi->bxyzki", U_tau, grad_phi)
    diff_u_vec = torch.einsum("bxyzki,kfi->bxyzkf", grad_u, grad_phi) * vol

    # weak form of p - Delta u = 0:
    # ∫ p phi + ∫ grad(u).grad(phi) = 0
    r_p_vec = p_vec + diff_u_vec

    if normalize:
        s_time = torch.sqrt(
            (time_vec ** 2).mean(dim=(1, 2, 3, 4, 5), keepdim=True).detach() + eps
        )
        s_source = torch.sqrt(
            (source_vec ** 2).mean(dim=(1, 2, 3, 4, 5), keepdim=True).detach() + eps
        )
        s_diff_p = torch.sqrt(
            (diff_p_vec ** 2).mean(dim=(1, 2, 3, 4, 5), keepdim=True).detach() + eps
        )

        scale_u = (s_time + s_source + s_diff_p) / 3.0
        r_u_vec = r_u_vec / (scale_u + eps)

        s_p = torch.sqrt(
            (p_vec ** 2).mean(dim=(1, 2, 3, 4, 5), keepdim=True).detach() + eps
        )
        s_diff_u = torch.sqrt(
            (diff_u_vec ** 2).mean(dim=(1, 2, 3, 4, 5), keepdim=True).detach() + eps
        )

        scale_p = 0.5 * (s_p + s_diff_u)
        r_p_vec = r_p_vec / (scale_p + eps)

    return r_u_vec.to(u_next.dtype), r_p_vec.to(u_next.dtype)


def sh_weak_fe_loss_tetra_neumann(
    u_prev,
    u_next,
    dt,
    dx,
    epsilon_param,
    tau=0.5,
    normalize=True,
    robust=False,
    p_weight=1.0,
    return_parts=False,
    eps=1e-8,
):
    r_u_vec, r_p_vec = sh_weak_fe_residual_tetra_neumann(
        u_prev,
        u_next,
        dt,
        dx,
        epsilon_param,
        tau=tau,
        normalize=normalize,
        eps=eps,
    )

    if robust:
        loss_u = torch.sqrt(r_u_vec * r_u_vec + eps).mean()
        loss_p = torch.sqrt(r_p_vec * r_p_vec + eps).mean()
    else:
        loss_u = (r_u_vec * r_u_vec).mean()
        loss_p = (r_p_vec * r_p_vec).mean()

    loss = loss_u + p_weight * loss_p

    if return_parts:
        return (
            loss.to(u_next.dtype),
            loss_u.to(u_next.dtype),
            loss_p.to(u_next.dtype),
        )

    return loss.to(u_next.dtype)

####

def ch_weak_fe_residual_tetra_neumann(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    tau=0.5,
    normalize=True,
    eps=1e-8,
):
    """
    Mixed weak-form FE residual for Cahn-Hilliard with homogeneous Neumann BC.

    Mixed CH system:
        u_t = Delta(mu)
        mu  = -eps2 * Delta(u) + (u^3 - u)

    Weak form on each tetrahedron K, for each local P1 basis/test function phi_i:

        r_u,i^K =
            ∫_K u_t phi_i dV + ∫_K grad(mu) · grad(phi_i) dV

        r_mu,i^K =
            ∫_K mu phi_i dV
            - eps2 ∫_K grad(u) · grad(phi_i) dV
            - ∫_K (u^3-u) phi_i dV

    Notes
    -----
    - Uses piecewise-linear P1 tetrahedral shape functions.
    - grad(u) and grad(mu) are constant on each tetrahedron.
    - Neumann BC is natural in the weak form.
    - Mean-free structure of CH is reflected by adding a mass penalty in the scalar loss.

    Parameters
    ----------
    u_prev, u_next : torch.Tensor
        Shape (B,S,S,S,1)
    tau : float
        Midpoint/collocation state parameter for mixed weak form.
    normalize : bool
        If True, normalize both residual blocks.
    eps : float

    Returns
    -------
    r_u_vec : torch.Tensor
        Shape (B,Cx,Cy,Cz,6,4)
    r_mu_vec : torch.Tensor
        Shape (B,Cx,Cy,Cz,6,4)
    mass_err : torch.Tensor
        Shape (B,)
    """
    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()   # (B,S,S,S)
    u1 = u_next.squeeze(-1).float()   # (B,S,S,S)

    # mixed/collocation state
    u_tau_grid = (1.0 - tau) * u0 + tau * u1  # (B,S,S,S)

    # build mu_tau on the grid using FD Neumann Laplacian
    lap_u_tau = laplacian_neumann_fd_3d(u_tau_grid, dx)
    mu_tau_grid = -eps2 * lap_u_tau + (u_tau_grid**3 - u_tau_grid)

    # tetra nodal values
    U0 = _cube_to_tetrahedra_all(u0)          # (B,Cx,Cy,Cz,6,4)
    U1 = _cube_to_tetrahedra_all(u1)
    U_tau = _cube_to_tetrahedra_all(u_tau_grid)
    MU_tau = _cube_to_tetrahedra_all(mu_tau_grid)

    grad_phi, volume = _tetra_grad_shape_families(dx, u1.device, u1.dtype)
    vol = volume.view(1, 1, 1, 1, 6, 1)

    # ------------------------------------------------------------
    # Block 1: weak form of u_t = Delta(mu)
    #   ∫ u_t phi_i + ∫ grad(mu)·grad(phi_i) = 0
    # ------------------------------------------------------------
    Ut = (U1 - U0) / dt                              # nodal time derivative
    ut_cent = Ut.mean(dim=-1, keepdim=True)         # centroid value
    time_vec = ut_cent * (0.25 * vol)               # phi_i(centroid)=1/4
    time_vec = time_vec.expand_as(U1)

    grad_mu = torch.einsum("bxyzkf,kfi->bxyzki", MU_tau, grad_phi)  # (B,Cx,Cy,Cz,6,3)
    diff_mu_vec = torch.einsum("bxyzki,kfi->bxyzkf", grad_mu, grad_phi) * vol

    r_u_vec = time_vec + diff_mu_vec

    # ------------------------------------------------------------
    # Block 2: weak form of mu = -eps2 Delta(u) + (u^3-u)
    #   ∫ mu phi_i - eps2 ∫ grad(u)·grad(phi_i) - ∫ (u^3-u) phi_i = 0
    # ------------------------------------------------------------
    mu_cent = MU_tau.mean(dim=-1, keepdim=True)
    mu_vec = mu_cent * (0.25 * vol)
    mu_vec = mu_vec.expand_as(U1)

    grad_u = torch.einsum("bxyzkf,kfi->bxyzki", U_tau, grad_phi)
    diff_u_vec = torch.einsum("bxyzki,kfi->bxyzkf", grad_u, grad_phi) * vol

    nonlin_cent = ((U_tau.mean(dim=-1, keepdim=True) ** 3) - U_tau.mean(dim=-1, keepdim=True))
    react_vec = nonlin_cent * (0.25 * vol)
    react_vec = react_vec.expand_as(U1)

    r_mu_vec = mu_vec - eps2 * diff_u_vec - react_vec

    if normalize:
        s_ru = torch.sqrt((r_u_vec**2).mean(dim=(1, 2, 3, 4, 5), keepdim=True).detach() + eps)
        s_rmu = torch.sqrt((r_mu_vec**2).mean(dim=(1, 2, 3, 4, 5), keepdim=True).detach() + eps)

        r_u_vec = r_u_vec / (s_ru + eps)
        r_mu_vec = r_mu_vec / (s_rmu + eps)

    # exact mass conservation defect
    m_prev = u0.mean(dim=(1, 2, 3))
    m_next = u1.mean(dim=(1, 2, 3))
    mass_err = (m_next - m_prev) ** 2

    return r_u_vec.to(u_next.dtype), r_mu_vec.to(u_next.dtype), mass_err.to(u_next.dtype)


def ch_weak_fe_loss_tetra_neumann(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    tau=0.5,
    normalize=True,
    robust=False,
    mass_weight=0.25,
    eps=1e-8,
):
    """
    Scalar mixed weak-form FE loss for Cahn-Hilliard with tetrahedral P1 basis.

    Loss =
        mean(r_u^2) + mean(r_mu^2) + mass_weight * mass_err

    Parameters
    ----------
    u_prev, u_next : (B,S,S,S,1)

    Returns
    -------
    scalar tensor
    """
    r_u_vec, r_mu_vec, mass_err = ch_weak_fe_residual_tetra_neumann(
        u_prev,
        u_next,
        dt,
        dx,
        eps2,
        tau=tau,
        normalize=normalize,
        eps=eps,
    )

    if robust:
        loss_u = torch.sqrt(r_u_vec * r_u_vec + eps).mean()
        loss_mu = torch.sqrt(r_mu_vec * r_mu_vec + eps).mean()
        loss_mass = torch.sqrt(mass_err + eps).mean()
    else:
        loss_u = (r_u_vec * r_u_vec).mean()
        loss_mu = (r_mu_vec * r_mu_vec).mean()
        loss_mass = mass_err.mean()

    loss = loss_u + loss_mu # + mass_weight * loss_mass
    return loss.to(u_next.dtype)

## Quadrature

import math
import torch


def _tetra_p1_quad4_rule(device, dtype):
    """
    4-point symmetric quadrature rule on a tetrahedron.
    Exact for quadratic polynomials.

    Returns
    -------
    phi_q : (4, 4)
        Barycentric coordinates of the 4 quadrature points.
        Since P1 shape functions equal barycentric coordinates,
        phi_q[q, i] = phi_i(x_q).
    w_q : (4,)
        Quadrature weights, summing to 1 on the physical tetra after
        multiplication by tetra volume.
    """
    a = 0.5854101966249685
    b = 0.1381966011250105

    phi_q = torch.tensor(
        [
            [a, b, b, b],
            [b, a, b, b],
            [b, b, a, b],
            [b, b, b, a],
        ],
        device=device,
        dtype=dtype,
    )
    w_q = torch.full((4,), 0.25, device=device, dtype=dtype)
    return phi_q, w_q


def _tetra_p1_load_vector_from_nodal_scalar_quad4(Fnodal, vol, phi_q, w_q):
    """
    Assemble FE load vector
        ∫_K f phi_i dV
    using 4-point tetrahedral quadrature, where f is represented by its
    nodal P1 values on the tetrahedron.

    Parameters
    ----------
    Fnodal : torch.Tensor
        Shape (B,Cx,Cy,Cz,6,4)
        Nodal values of a scalar field on each tetrahedron.
    vol : torch.Tensor
        Shape (1,1,1,1,6,1)
        Tetrahedron volumes.
    phi_q : torch.Tensor
        Shape (Q,4)
    w_q : torch.Tensor
        Shape (Q,)

    Returns
    -------
    load_vec : torch.Tensor
        Shape (B,Cx,Cy,Cz,6,4)
    """
    # Evaluate scalar field at quadrature points:
    # F_q(..., q) = sum_a F_a * phi_a(x_q)
    F_q = torch.einsum("bxyzkf,qf->bxyzkq", Fnodal, phi_q)  # (B,Cx,Cy,Cz,6,Q)

    # ∫ f phi_i dV ≈ vol * Σ_q w_q f(x_q) phi_i(x_q)
    load_vec = vol * torch.einsum("bxyzkq,q,qi->bxyzki", F_q, w_q, phi_q)
    return load_vec


def _tetra_p1_load_vector_from_point_scalar_quad4(F_q, vol, phi_q, w_q):
    """
    Assemble FE load vector
        ∫_K f phi_i dV
    when f is already evaluated at the quadrature points.

    Parameters
    ----------
    F_q : torch.Tensor
        Shape (B,Cx,Cy,Cz,6,Q)
    vol : torch.Tensor
        Shape (1,1,1,1,6,1)
    phi_q : torch.Tensor
        Shape (Q,4)
    w_q : torch.Tensor
        Shape (Q,)

    Returns
    -------
    load_vec : torch.Tensor
        Shape (B,Cx,Cy,Cz,6,4)
    """
    load_vec = vol * torch.einsum("bxyzkq,q,qi->bxyzki", F_q, w_q, phi_q)
    return load_vec


def ch_weak_fe_residual_tetra_neumann_quad4(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    tau=0.5,
    normalize=True,
    eps=1e-8,
):
    """
    Mixed weak-form FE residual for Cahn-Hilliard with homogeneous Neumann BC,
    using 4-point tetrahedral quadrature for the lower-order/load terms.

    Mixed CH system:
        u_t = Delta(mu)
        mu  = -eps2 * Delta(u) + (u^3 - u)

    Weak form on each tetrahedron K, for each local P1 basis/test function phi_i:

        r_u,i^K =
            ∫_K u_t phi_i dV + ∫_K grad(mu) · grad(phi_i) dV

        r_mu,i^K =
            ∫_K mu phi_i dV
            - eps2 ∫_K grad(u) · grad(phi_i) dV
            - ∫_K (u^3-u) phi_i dV

    Compared with the original centroid version, this keeps the same P1 FE method
    but improves the quadrature of the time/mu/nonlinear terms.
    """
    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()   # (B,S,S,S)
    u1 = u_next.squeeze(-1).float()   # (B,S,S,S)

    # midpoint/collocation state
    u_tau_grid = (1.0 - tau) * u0 + tau * u1

    # chemical potential on the grid (FD Neumann Laplacian)
    lap_u_tau = laplacian_neumann_fd_3d(u_tau_grid, dx)
    mu_tau_grid = -eps2 * lap_u_tau + (u_tau_grid**3 - u_tau_grid)

    # tetra nodal values
    U0 = _cube_to_tetrahedra_all(u0)            # (B,Cx,Cy,Cz,6,4)
    U1 = _cube_to_tetrahedra_all(u1)
    U_tau = _cube_to_tetrahedra_all(u_tau_grid)
    MU_tau = _cube_to_tetrahedra_all(mu_tau_grid)

    # P1 tetra geometry
    grad_phi, volume = _tetra_grad_shape_families(dx, u1.device, u1.dtype)
    vol = volume.view(1, 1, 1, 1, 6, 1)

    # 4-point tetra quadrature
    phi_q, w_q = _tetra_p1_quad4_rule(u1.device, u1.dtype)

    # ------------------------------------------------------------
    # Block 1: weak form of u_t = Delta(mu)
    #   ∫ u_t phi_i + ∫ grad(mu)·grad(phi_i) = 0
    # ------------------------------------------------------------
    Ut = (U1 - U0) / dt                                # nodal u_t
    time_vec = _tetra_p1_load_vector_from_nodal_scalar_quad4(Ut, vol, phi_q, w_q)

    # grad(mu) constant on each tetra because MU_tau is represented by P1 nodal values
    grad_mu = torch.einsum("bxyzkf,kfi->bxyzki", MU_tau, grad_phi)   # (B,Cx,Cy,Cz,6,3)
    diff_mu_vec = torch.einsum("bxyzki,kfi->bxyzkf", grad_mu, grad_phi) * vol

    r_u_vec = time_vec + diff_mu_vec

    # ------------------------------------------------------------
    # Block 2: weak form of mu = -eps2 Delta(u) + (u^3-u)
    #   ∫ mu phi_i - eps2 ∫ grad(u)·grad(phi_i) - ∫ (u^3-u) phi_i = 0
    # ------------------------------------------------------------
    mu_vec = _tetra_p1_load_vector_from_nodal_scalar_quad4(MU_tau, vol, phi_q, w_q)

    grad_u = torch.einsum("bxyzkf,kfi->bxyzki", U_tau, grad_phi)
    diff_u_vec = torch.einsum("bxyzki,kfi->bxyzkf", grad_u, grad_phi) * vol

    # nonlinear term at quadrature points
    u_tau_q = torch.einsum("bxyzkf,qf->bxyzkq", U_tau, phi_q)     # (B,Cx,Cy,Cz,6,Q)
    nonlin_q = u_tau_q**3 - u_tau_q
    react_vec = _tetra_p1_load_vector_from_point_scalar_quad4(nonlin_q, vol, phi_q, w_q)

    r_mu_vec = mu_vec - eps2 * diff_u_vec - react_vec

    if normalize:
        s_ru = torch.sqrt((r_u_vec**2).mean(dim=(1, 2, 3, 4, 5), keepdim=True).detach() + eps)
        s_rmu = torch.sqrt((r_mu_vec**2).mean(dim=(1, 2, 3, 4, 5), keepdim=True).detach() + eps)

        r_u_vec = r_u_vec / (s_ru + eps)
        r_mu_vec = r_mu_vec / (s_rmu + eps)

    # exact mass conservation defect (optional in scalar loss)
    m_prev = u0.mean(dim=(1, 2, 3))
    m_next = u1.mean(dim=(1, 2, 3))
    mass_err = (m_next - m_prev) ** 2

    return r_u_vec.to(u_next.dtype), r_mu_vec.to(u_next.dtype), mass_err.to(u_next.dtype)


def ch_weak_fe_loss_tetra_neumann_quad4(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    tau=0.5,
    normalize=True,
    robust=False,
    mass_weight=0.0,
    eps=1e-8,
):
    """
    Scalar mixed weak-form FE loss for Cahn-Hilliard with tetrahedral P1 basis,
    but using 4-point tetrahedral quadrature for the lower-order terms.

    Loss =
        mean(r_u^2) + mean(r_mu^2) + mass_weight * mass_err
    """
    r_u_vec, r_mu_vec, mass_err = ch_weak_fe_residual_tetra_neumann_quad4(
        u_prev,
        u_next,
        dt,
        dx,
        eps2,
        tau=tau,
        normalize=normalize,
        eps=eps,
    )

    if robust:
        loss_u = torch.sqrt(r_u_vec * r_u_vec + eps).mean()
        loss_mu = torch.sqrt(r_mu_vec * r_mu_vec + eps).mean()
        loss_mass = torch.sqrt(mass_err + eps).mean()
    else:
        loss_u = (r_u_vec * r_u_vec).mean()
        loss_mu = (r_mu_vec * r_mu_vec).mean()
        loss_mass = mass_err.mean()

    loss = loss_u + loss_mu + mass_weight * loss_mass
    return loss.to(u_next.dtype)




import torch
import math




#######

def _tetra_p2_from_p1_edge_average(U4):
    """
    Construct P2 tetrahedral nodal values from P1 vertex values.

    P2 node ordering:
        0,1,2,3        : vertices
        4=(0,1)
        5=(0,2)
        6=(0,3)
        7=(1,2)
        8=(1,3)
        9=(2,3)

    Parameters
    ----------
    U4 : torch.Tensor
        Shape (...,4), vertex values on each tetrahedron.

    Returns
    -------
    U10 : torch.Tensor
        Shape (...,10), P2 nodal values.
    """
    u0 = U4[..., 0]
    u1 = U4[..., 1]
    u2 = U4[..., 2]
    u3 = U4[..., 3]

    e01 = 0.5 * (u0 + u1)
    e02 = 0.5 * (u0 + u2)
    e03 = 0.5 * (u0 + u3)
    e12 = 0.5 * (u1 + u2)
    e13 = 0.5 * (u1 + u3)
    e23 = 0.5 * (u2 + u3)

    return torch.stack(
        [u0, u1, u2, u3, e01, e02, e03, e12, e13, e23],
        dim=-1,
    )


def _tetra_p2_shape_data_quad4(device, dtype, grad_phi_p1):
    """
    P2 tetrahedral shape functions and gradients evaluated at the 4-point
    tetrahedral quadrature rule.

    Parameters
    ----------
    grad_phi_p1 : torch.Tensor
        Shape (6,4,3)
        Gradients of P1 barycentric shape functions for each tetra family.

    Returns
    -------
    Nq : torch.Tensor
        Shape (Q,10), P2 shape values at quadrature points.

    wq : torch.Tensor
        Shape (Q,), quadrature weights.

    gradNq : torch.Tensor
        Shape (6,Q,10,3), P2 shape gradients at quadrature points.
    """
    # 4-point quadrature in barycentric coordinates
    a = 0.5854101966249685
    b = 0.1381966011250105

    L = torch.tensor(
        [
            [a, b, b, b],
            [b, a, b, b],
            [b, b, a, b],
            [b, b, b, a],
        ],
        device=device,
        dtype=dtype,
    )  # (Q,4)

    wq = torch.full((4,), 0.25, device=device, dtype=dtype)

    L0 = L[:, 0]
    L1 = L[:, 1]
    L2 = L[:, 2]
    L3 = L[:, 3]

    # P2 shape values
    N0 = L0 * (2.0 * L0 - 1.0)
    N1 = L1 * (2.0 * L1 - 1.0)
    N2 = L2 * (2.0 * L2 - 1.0)
    N3 = L3 * (2.0 * L3 - 1.0)

    N01 = 4.0 * L0 * L1
    N02 = 4.0 * L0 * L2
    N03 = 4.0 * L0 * L3
    N12 = 4.0 * L1 * L2
    N13 = 4.0 * L1 * L3
    N23 = 4.0 * L2 * L3

    Nq = torch.stack(
        [N0, N1, N2, N3, N01, N02, N03, N12, N13, N23],
        dim=-1,
    )  # (Q,10)

    # P1 gradients are gradients of barycentric coordinates:
    # grad lambda_i
    g = grad_phi_p1  # (6,4,3)

    g0 = g[:, 0, :]
    g1 = g[:, 1, :]
    g2 = g[:, 2, :]
    g3 = g[:, 3, :]

    grad_list = []

    for q in range(4):
        l0 = L[q, 0]
        l1 = L[q, 1]
        l2 = L[q, 2]
        l3 = L[q, 3]

        # vertex P2 gradients:
        # grad[lambda_i(2lambda_i-1)] = (4lambda_i-1) grad lambda_i
        dN0 = (4.0 * l0 - 1.0) * g0
        dN1 = (4.0 * l1 - 1.0) * g1
        dN2 = (4.0 * l2 - 1.0) * g2
        dN3 = (4.0 * l3 - 1.0) * g3

        # edge P2 gradients:
        # grad[4 lambda_i lambda_j] = 4(lambda_i grad lambda_j + lambda_j grad lambda_i)
        dN01 = 4.0 * (l0 * g1 + l1 * g0)
        dN02 = 4.0 * (l0 * g2 + l2 * g0)
        dN03 = 4.0 * (l0 * g3 + l3 * g0)
        dN12 = 4.0 * (l1 * g2 + l2 * g1)
        dN13 = 4.0 * (l1 * g3 + l3 * g1)
        dN23 = 4.0 * (l2 * g3 + l3 * g2)

        grad_q = torch.stack(
            [dN0, dN1, dN2, dN3, dN01, dN02, dN03, dN12, dN13, dN23],
            dim=1,
        )  # (6,10,3)

        grad_list.append(grad_q)

    gradNq = torch.stack(grad_list, dim=1)  # (6,Q,10,3)

    return Nq, wq, gradNq


def _tetra_p2_load_vector_quad4(F10, vol, Nq, wq):
    """
    Assemble P2 load vector:
        ∫_K f N_i dV

    Parameters
    ----------
    F10 : torch.Tensor
        Shape (B,Cx,Cy,Cz,6,10)
        P2 nodal values of scalar field f.

    vol : torch.Tensor
        Shape (1,1,1,1,6,1)

    Nq : torch.Tensor
        Shape (Q,10)

    wq : torch.Tensor
        Shape (Q,)

    Returns
    -------
    load_vec : torch.Tensor
        Shape (B,Cx,Cy,Cz,6,10)
    """
    Fq = torch.einsum("bxyzkf,qf->bxyzkq", F10, Nq)
    load_vec = vol * torch.einsum("bxyzkq,q,qi->bxyzki", Fq, wq, Nq)
    return load_vec


def _tetra_p2_load_vector_from_quad_values(Fq, vol, Nq, wq):
    """
    Assemble P2 load vector:
        ∫_K f N_i dV

    when f is already evaluated at quadrature points.

    Parameters
    ----------
    Fq : torch.Tensor
        Shape (B,Cx,Cy,Cz,6,Q)

    Returns
    -------
    load_vec : torch.Tensor
        Shape (B,Cx,Cy,Cz,6,10)
    """
    load_vec = vol * torch.einsum("bxyzkq,q,qi->bxyzki", Fq, wq, Nq)
    return load_vec


def ch_weak_fe_residual_tetra_neumann_p2_quad4(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    tau=0.5,
    normalize=True,
    eps=1e-8,
):
    """
    Mixed weak-form FE residual for Cahn-Hilliard using P2 tetrahedral
    shape functions and 4-point tetrahedral quadrature.

    Important:
    This constructs P2 edge values by averaging vertex values, because the
    model/data only provide vertex grid values. Therefore this is a
    P2-compatible quadratic assembly, not a fully independent P2 FEM with
    separate edge DOFs.

    Mixed CH system:
        u_t = Delta(mu)
        mu  = -eps2 * Delta(u) + (u^3 - u)

    Weak form:
        r_u,i^K =
            ∫_K u_t N_i dV + ∫_K grad(mu) · grad(N_i) dV

        r_mu,i^K =
            ∫_K mu N_i dV
            - eps2 ∫_K grad(u) · grad(N_i) dV
            - ∫_K (u^3-u) N_i dV
    """
    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()
    u1 = u_next.squeeze(-1).float()

    # midpoint / collocation state
    u_tau_grid = (1.0 - tau) * u0 + tau * u1

    # chemical potential on grid, same as your current CH weak-FE formulation
    lap_u_tau = laplacian_neumann_fd_3d(u_tau_grid, dx)
    mu_tau_grid = -eps2 * lap_u_tau + (u_tau_grid**3 - u_tau_grid)

    # P1 tetra nodal values, shape (...,4)
    U0_4 = _cube_to_tetrahedra_all(u0)
    U1_4 = _cube_to_tetrahedra_all(u1)
    Utau_4 = _cube_to_tetrahedra_all(u_tau_grid)
    MU_4 = _cube_to_tetrahedra_all(mu_tau_grid)

    # Convert to P2 nodal values, shape (...,10)
    U0_10 = _tetra_p2_from_p1_edge_average(U0_4)
    U1_10 = _tetra_p2_from_p1_edge_average(U1_4)
    Utau_10 = _tetra_p2_from_p1_edge_average(Utau_4)
    MU_10 = _tetra_p2_from_p1_edge_average(MU_4)

    # geometry
    grad_phi_p1, volume = _tetra_grad_shape_families(dx, u1.device, u1.dtype)
    vol = volume.view(1, 1, 1, 1, 6, 1)

    # P2 shape data at quadrature points
    Nq, wq, gradNq = _tetra_p2_shape_data_quad4(
        u1.device,
        u1.dtype,
        grad_phi_p1,
    )
    # Nq: (Q,10)
    # gradNq: (6,Q,10,3)

    # ------------------------------------------------------------
    # Block 1:
    # ∫ u_t N_i dV + ∫ grad(mu) · grad(N_i) dV
    # ------------------------------------------------------------
    Ut_10 = (U1_10 - U0_10) / dt
    time_vec = _tetra_p2_load_vector_quad4(Ut_10, vol, Nq, wq)

    # grad(mu) at quadrature points:
    # grad_mu_q = sum_a MU_a grad N_a(x_q)
    grad_mu_q = torch.einsum(
        "bxyzkf,kqfi->bxyzkqi",
        MU_10,
        gradNq,
    )  # (B,Cx,Cy,Cz,6,Q,3)

    diff_mu_vec = vol * torch.einsum(
        "bxyzkqd,kqid,q->bxyzki",
        grad_mu_q,
        gradNq,
        wq,
    )  # (B,Cx,Cy,Cz,6,10)

    r_u_vec = time_vec + diff_mu_vec

    # ------------------------------------------------------------
    # Block 2:
    # ∫ mu N_i dV
    # - eps2 ∫ grad(u) · grad(N_i) dV
    # - ∫ (u^3-u) N_i dV
    # ------------------------------------------------------------
    mu_vec = _tetra_p2_load_vector_quad4(MU_10, vol, Nq, wq)

    grad_u_q = torch.einsum(
        "bxyzkf,kqfi->bxyzkqi",
        Utau_10,
        gradNq,
    )  # (B,Cx,Cy,Cz,6,Q,3)

    diff_u_vec = vol * torch.einsum(
        "bxyzkqd,kqid,q->bxyzki",
        grad_u_q,
        gradNq,
        wq,
    )

    u_q = torch.einsum("bxyzkf,qf->bxyzkq", Utau_10, Nq)
    nonlin_q = u_q**3 - u_q
    react_vec = _tetra_p2_load_vector_from_quad_values(nonlin_q, vol, Nq, wq)

    r_mu_vec = mu_vec - eps2 * diff_u_vec - react_vec

    if normalize:
        s_ru = torch.sqrt(
            (r_u_vec**2).mean(dim=(1, 2, 3, 4, 5), keepdim=True).detach() + eps
        )
        s_rmu = torch.sqrt(
            (r_mu_vec**2).mean(dim=(1, 2, 3, 4, 5), keepdim=True).detach() + eps
        )

        r_u_vec = r_u_vec / (s_ru + eps)
        r_mu_vec = r_mu_vec / (s_rmu + eps)

    # optional mass defect
    m_prev = u0.mean(dim=(1, 2, 3))
    m_next = u1.mean(dim=(1, 2, 3))
    mass_err = (m_next - m_prev) ** 2

    return r_u_vec.to(u_next.dtype), r_mu_vec.to(u_next.dtype), mass_err.to(u_next.dtype)


def ch_weak_fe_loss_tetra_neumann_p2_quad4(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    tau=0.5,
    normalize=True,
    robust=False,
    mass_weight=0.0,
    eps=1e-8,
):
    """
    Scalar mixed weak-form FE loss for CH using P2 tetrahedral shape functions
    and 4-point quadrature.

    Loss:
        mean(r_u^2) + mean(r_mu^2) + mass_weight * mass_err
    """
    r_u_vec, r_mu_vec, mass_err = ch_weak_fe_residual_tetra_neumann_p2_quad4(
        u_prev,
        u_next,
        dt,
        dx,
        eps2,
        tau=tau,
        normalize=normalize,
        eps=eps,
    )

    if robust:
        loss_u = torch.sqrt(r_u_vec * r_u_vec + eps).mean()
        loss_mu = torch.sqrt(r_mu_vec * r_mu_vec + eps).mean()
        loss_mass = torch.sqrt(mass_err + eps).mean()
    else:
        loss_u = (r_u_vec * r_u_vec).mean()
        loss_mu = (r_mu_vec * r_mu_vec).mean()
        loss_mass = mass_err.mean()

    loss = loss_u + loss_mu + mass_weight * loss_mass
    return loss.to(u_next.dtype)


######



# ============================================================
# P2 tetrahedral utilities
# ============================================================

def _tetra_p2_quad5_rule(device, dtype):
    """
    5-point tetrahedral quadrature rule.

    This rule is commonly used for low-order nonlinear FE integration.
    Returns barycentric coordinates and weights.

    Returns
    -------
    lam_q : (5,4)
        barycentric coordinates of quadrature points
    w_q : (5,)
        quadrature weights on reference tetra, summing to 1
        after multiplication by physical tetra volume
    """
    # centroid
    c = 0.25

    # symmetric 4-point shell
    a = 0.5
    b = 1.0 / 6.0

    lam_q = torch.tensor(
        [
            [c, c, c, c],
            [a, b, b, b],
            [b, a, b, b],
            [b, b, a, b],
            [b, b, b, a],
        ],
        device=device,
        dtype=dtype,
    )

    # positive-weight 5-point rule
    w_q = torch.tensor(
        [
            -0.8,
             0.45,
             0.45,
             0.45,
             0.45,
        ],
        device=device,
        dtype=dtype,
    )

    return lam_q, w_q


def _tetra_p2_shape_functions(lam):
    """
    Quadratic tetrahedral (P2) shape functions.

    Parameters
    ----------
    lam : (Q,4)
        barycentric coordinates [l1,l2,l3,l4]

    Returns
    -------
    N : (Q,10)
        shape functions in the node ordering:
        [v0,v1,v2,v3,e01,e02,e03,e12,e13,e23]
    """
    l0, l1, l2, l3 = lam.unbind(dim=-1)

    N0 = l0 * (2.0 * l0 - 1.0)
    N1 = l1 * (2.0 * l1 - 1.0)
    N2 = l2 * (2.0 * l2 - 1.0)
    N3 = l3 * (2.0 * l3 - 1.0)

    N4 = 4.0 * l0 * l1   # e01
    N5 = 4.0 * l0 * l2   # e02
    N6 = 4.0 * l0 * l3   # e03
    N7 = 4.0 * l1 * l2   # e12
    N8 = 4.0 * l1 * l3   # e13
    N9 = 4.0 * l2 * l3   # e23

    return torch.stack([N0, N1, N2, N3, N4, N5, N6, N7, N8, N9], dim=-1)


def _tetra_p2_shape_gradients(lam, grad_lam):
    """
    Gradients of quadratic tetrahedral (P2) shape functions.

    Parameters
    ----------
    lam : (Q,4)
        barycentric coordinates
    grad_lam : (6,4,3)
        gradients of barycentric coordinates / P1 shape functions
        for each of the 6 tetra families in a cube

    Returns
    -------
    gradN : (Q,6,10,3)
        gradients of the 10 P2 shape functions
    """
    l0, l1, l2, l3 = lam.unbind(dim=-1)   # each (Q,)

    g0 = grad_lam[:, 0, :]   # (6,3)
    g1 = grad_lam[:, 1, :]
    g2 = grad_lam[:, 2, :]
    g3 = grad_lam[:, 3, :]

    Q = lam.shape[0]

    # vertex functions
    G0 = (4.0 * l0 - 1.0).view(Q, 1, 1) * g0.unsqueeze(0)
    G1 = (4.0 * l1 - 1.0).view(Q, 1, 1) * g1.unsqueeze(0)
    G2 = (4.0 * l2 - 1.0).view(Q, 1, 1) * g2.unsqueeze(0)
    G3 = (4.0 * l3 - 1.0).view(Q, 1, 1) * g3.unsqueeze(0)

    # edge functions
    G4 = 4.0 * (l0.view(Q,1,1) * g1.unsqueeze(0) + l1.view(Q,1,1) * g0.unsqueeze(0))
    G5 = 4.0 * (l0.view(Q,1,1) * g2.unsqueeze(0) + l2.view(Q,1,1) * g0.unsqueeze(0))
    G6 = 4.0 * (l0.view(Q,1,1) * g3.unsqueeze(0) + l3.view(Q,1,1) * g0.unsqueeze(0))
    G7 = 4.0 * (l1.view(Q,1,1) * g2.unsqueeze(0) + l2.view(Q,1,1) * g1.unsqueeze(0))
    G8 = 4.0 * (l1.view(Q,1,1) * g3.unsqueeze(0) + l3.view(Q,1,1) * g1.unsqueeze(0))
    G9 = 4.0 * (l2.view(Q,1,1) * g3.unsqueeze(0) + l3.view(Q,1,1) * g2.unsqueeze(0))

    gradN = torch.stack([G0, G1, G2, G3, G4, G5, G6, G7, G8, G9], dim=2)  # (Q,6,10,3)
    return gradN


def _cube_to_tetrahedra_all_p2(u):
    """
    Build pseudo-P2 tetrahedral nodal values from grid nodal field.

    Input
    -----
    u : (B,Sx,Sy,Sz)

    Output
    ------
    U_p2 : (B,Cx,Cy,Cz,6,10)
        Node order:
        [v0,v1,v2,v3,e01,e02,e03,e12,e13,e23]

    Notes
    -----
    - The 4 vertex values come directly from the grid.
    - The 6 edge values are reconstructed as midpoint averages.
    """
    U = _cube_to_tetrahedra_all(u)  # (B,Cx,Cy,Cz,6,4)

    v0 = U[..., 0]
    v1 = U[..., 1]
    v2 = U[..., 2]
    v3 = U[..., 3]

    e01 = 0.5 * (v0 + v1)
    e02 = 0.5 * (v0 + v2)
    e03 = 0.5 * (v0 + v3)
    e12 = 0.5 * (v1 + v2)
    e13 = 0.5 * (v1 + v3)
    e23 = 0.5 * (v2 + v3)

    U_p2 = torch.stack([v0, v1, v2, v3, e01, e02, e03, e12, e13, e23], dim=-1)
    return U_p2


def _tetra_p2_eval_scalar(U_p2, Nq):
    """
    Evaluate a P2 scalar field at quadrature points.

    Parameters
    ----------
    U_p2 : (B,Cx,Cy,Cz,6,10)
    Nq   : (Q,10)

    Returns
    -------
    val_q : (B,Cx,Cy,Cz,6,Q)
    """
    return torch.einsum("bxyzkf,qf->bxyzkq", U_p2, Nq)


def _tetra_p2_eval_grad(U_p2, gradNq):
    """
    Evaluate gradient of a P2 scalar field at quadrature points.

    Parameters
    ----------
    U_p2 : (B,Cx,Cy,Cz,6,10)
    gradNq : (Q,6,10,3)

    Returns
    -------
    grad_q : (B,Cx,Cy,Cz,6,Q,3)
    """
    return torch.einsum("bxyzkf,qkfi->bxyzkqi", U_p2, gradNq)


def _tetra_p2_load_vector_from_scalar_q(F_q, Nq, w_q, vol):
    """
    Assemble load vector:
        ∫_K f N_i dV

    Parameters
    ----------
    F_q : (B,Cx,Cy,Cz,6,Q)
    Nq  : (Q,10)
    w_q : (Q,)
    vol : (1,1,1,1,6,1)

    Returns
    -------
    vec : (B,Cx,Cy,Cz,6,10)
    """
    return vol * torch.einsum("bxyzkq,q,qf->bxyzkf", F_q, w_q, Nq)


def _tetra_p2_stiffness_action_from_grad(grad_u_q, gradNq, w_q, vol):
    """
    Assemble vector:
        ∫_K grad(u) · grad(N_i) dV

    Parameters
    ----------
    grad_u_q : (B,Cx,Cy,Cz,6,Q,3)
    gradNq   : (Q,6,10,3)
    w_q      : (Q,)
    vol      : (1,1,1,1,6,1)

    Returns
    -------
    vec : (B,Cx,Cy,Cz,6,10)
    """
    # grad_u · gradN_i, summed over spatial dim
    # result shape: (B,Cx,Cy,Cz,6,Q,10)
    dot_q = torch.einsum("bxyzkqi,qkfi->bxyzkqf", grad_u_q, gradNq)
    return vol * torch.einsum("bxyzkqf,q->bxyzkf", dot_q, w_q)


# ============================================================
# CH P2 tetrahedral weak-FE residual / loss
# ============================================================

def ch_weak_fe_residual_tetra_neumann_p2(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    tau=0.5,
    normalize=True,
    eps=1e-8,
):
    """
    Mixed weak-form FE residual for Cahn-Hilliard using quadratic tetrahedral (P2) shape functions.

    Mixed CH system:
        u_t = Delta(mu)
        mu  = -eps2 * Delta(u) + (u^3 - u)

    Weak form on each tetrahedron K, for each local P2 basis/test function N_i:

        r_u,i^K =
            ∫_K u_t N_i dV + ∫_K grad(mu) · grad(N_i) dV

        r_mu,i^K =
            ∫_K mu N_i dV
            - eps2 ∫_K grad(u) · grad(N_i) dV
            - ∫_K (u^3-u) N_i dV

    Notes
    -----
    - P2 edge DOFs are reconstructed by midpoint averaging from grid vertex values.
    - Neumann BC is natural in the weak form.
    - Uses 5-point tetrahedral quadrature.
    """
    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()   # (B,S,S,S)
    u1 = u_next.squeeze(-1).float()   # (B,S,S,S)

    # midpoint / collocation state on the grid
    u_tau_grid = (1.0 - tau) * u0 + tau * u1

    # same grid-level mu_tau construction as your current CH setup
    lap_u_tau = laplacian_neumann_fd_3d(u_tau_grid, dx)
    mu_tau_grid = -eps2 * lap_u_tau + (u_tau_grid**3 - u_tau_grid)

    # P2 tetra nodal values (10 local nodes)
    U0_p2   = _cube_to_tetrahedra_all_p2(u0)
    U1_p2   = _cube_to_tetrahedra_all_p2(u1)
    Uta_p2  = _cube_to_tetrahedra_all_p2(u_tau_grid)
    MUta_p2 = _cube_to_tetrahedra_all_p2(mu_tau_grid)

    # geometry from P1 barycentric gradients
    grad_phi, volume = _tetra_grad_shape_families(dx, u1.device, u1.dtype)
    vol = volume.view(1, 1, 1, 1, 6, 1)

    # P2 quadrature + P2 shape data
    lam_q, w_q = _tetra_p2_quad5_rule(u1.device, u1.dtype)    # (Q,4), (Q,)
    Nq = _tetra_p2_shape_functions(lam_q)                     # (Q,10)
    gradNq = _tetra_p2_shape_gradients(lam_q, grad_phi)       # (Q,6,10,3)

    # ------------------------------------------------------------
    # Block 1: weak form of u_t = Delta(mu)
    #   ∫ u_t N_i + ∫ grad(mu)·grad(N_i) = 0
    # ------------------------------------------------------------
    Ut_p2 = (U1_p2 - U0_p2) / dt                              # (B,Cx,Cy,Cz,6,10)
    ut_q = _tetra_p2_eval_scalar(Ut_p2, Nq)                  # (B,Cx,Cy,Cz,6,Q)
    time_vec = _tetra_p2_load_vector_from_scalar_q(ut_q, Nq, w_q, vol)

    grad_mu_q = _tetra_p2_eval_grad(MUta_p2, gradNq)         # (B,Cx,Cy,Cz,6,Q,3)
    diff_mu_vec = _tetra_p2_stiffness_action_from_grad(grad_mu_q, gradNq, w_q, vol)

    r_u_vec = time_vec + diff_mu_vec

    # ------------------------------------------------------------
    # Block 2: weak form of mu = -eps2 Delta(u) + (u^3-u)
    #   ∫ mu N_i - eps2 ∫ grad(u)·grad(N_i) - ∫ (u^3-u) N_i = 0
    # ------------------------------------------------------------
    mu_q = _tetra_p2_eval_scalar(MUta_p2, Nq)
    mu_vec = _tetra_p2_load_vector_from_scalar_q(mu_q, Nq, w_q, vol)

    grad_u_q = _tetra_p2_eval_grad(Uta_p2, gradNq)
    diff_u_vec = _tetra_p2_stiffness_action_from_grad(grad_u_q, gradNq, w_q, vol)

    u_tau_q = _tetra_p2_eval_scalar(Uta_p2, Nq)
    nonlin_q = u_tau_q**3 - u_tau_q
    react_vec = _tetra_p2_load_vector_from_scalar_q(nonlin_q, Nq, w_q, vol)

    r_mu_vec = mu_vec - eps2 * diff_u_vec - react_vec

    if normalize:
        s_ru = torch.sqrt((r_u_vec**2).mean(dim=(1, 2, 3, 4, 5), keepdim=True).detach() + eps)
        s_rmu = torch.sqrt((r_mu_vec**2).mean(dim=(1, 2, 3, 4, 5), keepdim=True).detach() + eps)

        r_u_vec = r_u_vec / (s_ru + eps)
        r_mu_vec = r_mu_vec / (s_rmu + eps)

    # exact mass conservation defect
    m_prev = u0.mean(dim=(1, 2, 3))
    m_next = u1.mean(dim=(1, 2, 3))
    mass_err = (m_next - m_prev) ** 2

    return r_u_vec.to(u_next.dtype), r_mu_vec.to(u_next.dtype), mass_err.to(u_next.dtype)


def ch_weak_fe_loss_tetra_neumann_p2(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    tau=0.5,
    normalize=True,
    robust=False,
    mass_weight=0.0,
    eps=1e-8,
):
    """
    Scalar mixed weak-form FE loss for Cahn-Hilliard using quadratic tetrahedral (P2) elements.

    Loss =
        mean(r_u^2) + mean(r_mu^2) + mass_weight * mass_err
    """
    r_u_vec, r_mu_vec, mass_err = ch_weak_fe_residual_tetra_neumann_p2(
        u_prev,
        u_next,
        dt,
        dx,
        eps2,
        tau=tau,
        normalize=normalize,
        eps=eps,
    )

    if robust:
        loss_u = torch.sqrt(r_u_vec * r_u_vec + eps).mean()
        loss_mu = torch.sqrt(r_mu_vec * r_mu_vec + eps).mean()
        loss_mass = torch.sqrt(mass_err + eps).mean()
    else:
        loss_u = (r_u_vec * r_u_vec).mean()
        loss_mu = (r_mu_vec * r_mu_vec).mean()
        loss_mass = mass_err.mean()

    loss = loss_u + loss_mu + mass_weight * loss_mass
    return loss.to(u_next.dtype)



##



def mixed_form_CH_physical_gauss_lobatto_single_step(
    u_prev,
    u_next,
    dt,
    dx,
    eps2,
    normalize=True,
    robust=False,
    mass_weight=0.1,
    eps=1e-8,
):
    """
    Mixed-form CH loss with two symmetric Gauss-Lobatto collocation points.

    Mixed CH:
        u_t = Delta(mu)
        mu  = -eps2 * Delta(u) + (u^3 - u)

    Residual at collocation state u_tau:
        R_tau = u_t - Delta(mu_tau)

    where
        u_tau = (1-tau) u_prev + tau u_next

    We use the two symmetric Gauss-Lobatto nodes:
        tau_{1,2} = 1/2 ± 1/(2*sqrt(5))

    and average the resulting FEM-style tetrahedral Gauss residuals.

    Inputs
    ------
    u_prev, u_next : (B,S,S,S,1)

    Returns
    -------
    scalar tensor
    """
    if u_prev.dim() != 5 or u_prev.shape[-1] != 1:
        raise ValueError("u_prev must have shape (B,S,S,S,1)")
    if u_next.dim() != 5 or u_next.shape[-1] != 1:
        raise ValueError("u_next must have shape (B,S,S,S,1)")

    u0 = u_prev.squeeze(-1).float()
    u1 = u_next.squeeze(-1).float()

    ut = (u1 - u0) / dt

    tau_off = 1.0 / (2.0 * math.sqrt(5.0))
    taus = [0.5 - tau_off, 0.5 + tau_off]

    loss_res_all = []

    for tau in taus:
        u_tau = (1.0 - tau) * u0 + tau * u1

        # mixed split in physical space
        lap_u_tau = laplacian_neumann_fd_3d(u_tau, dx)
        mu_tau = -eps2 * lap_u_tau + (u_tau**3 - u_tau)
        lap_mu_tau = laplacian_neumann_fd_3d(mu_tau, dx)

        R = ut - lap_mu_tau

        # CH residual should be mean-free
        R = R - R.mean(dim=(1, 2, 3), keepdim=True)

        if normalize:
            s_ut = ut.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
            s_lm = lap_mu_tau.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
            R = ut / s_ut - lap_mu_tau / s_lm
            R = R - R.mean(dim=(1, 2, 3), keepdim=True)

        if robust:
            loss_tau = tetra_gauss_integral_scalar(torch.sqrt(R * R + eps), dx).mean()
        else:
            loss_tau = tetra_gauss_l2_scalar(R, dx).mean()

        loss_res_all.append(loss_tau)

    loss_res = 0.5 * (loss_res_all[0] + loss_res_all[1])

    # mass conservation
    m0 = u0.mean(dim=(1, 2, 3))
    m1 = u1.mean(dim=(1, 2, 3))
    mass_err = (m1 - m0) ** 2

    if robust:
        loss_mass = torch.sqrt(mass_err + eps).mean()
    else:
        loss_mass = mass_err.mean()

    return (loss_res + mass_weight * loss_mass).to(u_next.dtype)











#######
######


def ac_neumann_bc_loss(u_pred, dx, normalize=True):
    """
    Enforce homogeneous Neumann BC on predicted rollout.

    u_pred: (B,Sx,Sy,Sz,T) or (B,Sx,Sy,Sz,1)

    Neumann BC:
        du/dn = 0
    approximated by one-sided finite differences on the 6 faces.
    """
    up = u_pred.float()

    # x-faces
    dudx_l = (up[:, 1:2, :, :, :] - up[:, 0:1, :, :, :]) / dx
    dudx_r = (up[:, -1:, :, :, :] - up[:, -2:-1, :, :, :]) / dx

    # y-faces
    dudy_l = (up[:, :, 1:2, :, :] - up[:, :, 0:1, :, :]) / dx
    dudy_r = (up[:, :, -1:, :, :] - up[:, :, -2:-1, :, :]) / dx

    # z-faces
    dudz_l = (up[:, :, :, 1:2, :] - up[:, :, :, 0:1, :]) / dx
    dudz_r = (up[:, :, :, -1:, :] - up[:, :, :, -2:-1, :]) / dx

    if normalize:
        scale = up.pow(2).mean(dim=(1, 2, 3, 4), keepdim=True).sqrt().detach() / max(dx, 1e-12) + 1e-8

        dudx_l = dudx_l / scale
        dudx_r = dudx_r / scale
        dudy_l = dudy_l / scale
        dudy_r = dudy_r / scale
        dudz_l = dudz_l / scale
        dudz_r = dudz_r / scale

    loss_bc = (
        dudx_l.pow(2).mean() + dudx_r.pow(2).mean() +
        dudy_l.pow(2).mean() + dudy_r.pow(2).mean() +
        dudz_l.pow(2).mean() + dudz_r.pow(2).mean()
    ) / 6.0

    return loss_bc.to(u_pred.dtype)


def ac_initial_anchor_loss(u_in_last, u_pred, y_true=None, use_teacher=False):
    """
    Optional PINTO-like start-of-rollout anchor.

    If y_true is given:
        compares first predicted frame to ground-truth first frame.
    If use_teacher=True:
        compares first predicted frame to one semi-implicit teacher step.
    """
    y_first = u_pred[..., 0:1]

    if use_teacher:
        y_target = semi_implicit_step_AC_neumann(
            u_in_last, config.DT, config.DX, config.EPS2
        )[..., 0:1]
    else:
        if y_true is None:
            raise ValueError("y_true must be provided when use_teacher=False")
        y_target = y_true[..., 0:1]

    return F.mse_loss(y_first, y_target)

def project_neumann_cosine(u):
    """
    Project u onto Neumann-consistent cosine space.
    Works for:
        (B,S,S,S)
        (B,S,S,S,1)
        (B,S,S,S,T)
    """
    if u.dim() == 5:
        B, Sx, Sy, Sz, T = u.shape
        u_flat = u.permute(0, 4, 1, 2, 3).reshape(B * T, Sx, Sy, Sz)
        u_proj = idct3_neumann(dct3_neumann(u_flat))
        return u_proj.view(B, T, Sx, Sy, Sz).permute(0, 2, 3, 4, 1)

    elif u.dim() == 4:
        return idct3_neumann(dct3_neumann(u))

    else:
        raise ValueError("Unsupported shape")

def neumann_bc_residual_3d(u, dx):
    """
    u: (B,S,S,S,1) or (B,S,S,S,T)
    Returns scalar loss enforcing du/dn = 0 on all faces
    """

    # handle multi-step or single-step
    if u.dim() == 5:
        pass
    else:
        raise ValueError("u must be (B,S,S,S,T)")

    # finite differences (forward/backward)
    # x-direction
    res_x0 = (u[:, 1, :, :, :] - u[:, 0, :, :, :]) / dx
    res_x1 = (u[:, -1, :, :, :] - u[:, -2, :, :, :]) / dx

    # y-direction
    res_y0 = (u[:, :, 1, :, :] - u[:, :, 0, :, :]) / dx
    res_y1 = (u[:, :, -1, :, :] - u[:, :, -2, :, :]) / dx

    # z-direction
    res_z0 = (u[:, :, :, 1, :] - u[:, :, :, 0, :]) / dx
    res_z1 = (u[:, :, :, -1, :] - u[:, :, :, -2, :]) / dx

    loss = (
        res_x0.pow(2).mean() + res_x1.pow(2).mean() +
        res_y0.pow(2).mean() + res_y1.pow(2).mean() +
        res_z0.pow(2).mean() + res_z1.pow(2).mean()
    )

    return loss
def energy_ac_neumann(u, dx, eps2):
    """
    Allen–Cahn free energy:
    E = ∫ (eps^2/2 |∇u|^2 + (u^2 - 1)^2 / 4) dx
    """

    # gradients (central differences)
    ux = (u[:, 2:, :, :, :] - u[:, :-2, :, :, :]) / (2 * dx)
    uy = (u[:, :, 2:, :, :] - u[:, :, :-2, :, :]) / (2 * dx)
    uz = (u[:, :, :, 2:, :] - u[:, :, :, :-2, :]) / (2 * dx)

    # crop u to match gradient shape
    u_c = u[:, 1:-1, 1:-1, 1:-1, :]

    grad_sq = ux[:, :, 1:-1, 1:-1, :]**2 + \
              uy[:, 1:-1, :, 1:-1, :]**2 + \
              uz[:, 1:-1, 1:-1, :, :]**2

    potential = ((u_c**2 - 1.0)**2) / 4.0

    energy_density = 0.5 * eps2 * grad_sq + potential

    return energy_density.mean()


def ch_rollout_residual_neumann(u_hist_last, u_pred, dt, dx, eps2, mean_free=True):
    """
    u_hist_last: (B,S,S,S,1)
    u_pred     : (B,S,S,S,T)
    returns    : (B,S,S,S,T)

    Residual at each rollout step:
        (u_next - u_prev)/dt - RHS_CH(u_next)
    """
    T = u_pred.shape[-1]
    res_list = []
    u_prev = u_hist_last

    for t in range(T):
        u_next = u_pred[..., t:t+1]  # (B,S,S,S,1)

        ut = (u_next - u_prev) / dt
        rhs = pde_rhs_ch_neumann(u_next.squeeze(-1), dx, eps2).unsqueeze(-1)

        R = ut - rhs

        # CH residual should be mean-free
        if mean_free:
            R = R - R.mean(dim=(1, 2, 3), keepdim=True)

        res_list.append(R)
        u_prev = u_next

    return torch.cat(res_list, dim=-1)


def physics_collocation_tau_L2_CH_neumann_twopoints(u_in, u_pred, tau, normalize=True):
    """
    Two-point symmetric collocation for CH:
      tau and (1 - tau), averaged.
    Same usage style as AC.
    """
    l1 = physics_collocation_tau_L2_CH_neumann(
        u_in, u_pred, tau=tau, normalize=normalize
    )
    l2 = physics_collocation_tau_L2_CH_neumann(
        u_in, u_pred, tau=1.0 - tau, normalize=normalize
    )
    return 0.5 * (l1 + l2)


def energy_residual_ac(u_prev, u_next, dx, eps2):
    """
    Penalize energy increase: E(u_next) <= E(u_prev)
    """

    E_prev = energy_ac_neumann(u_prev, dx, eps2)
    E_next = energy_ac_neumann(u_next, dx, eps2)

    return torch.relu(E_next - E_prev)


def physics_collocation_multi_rule_AC_neumann_rollout(
    u_in,
    u_pred,
    rules=("lobatto", "jacobi"),
    weights=None,
    normalize=True,
    gegenbauer_lam=1.0,
    jacobi_alpha=0.0,
    jacobi_beta=0.0,
):
    """
    Multi-rule temporal collocation for AC3D + Neumann BC over a rollout block.
    Same style as your existing collocation wrappers.
    """
    if weights is None:
        weights = [1.0 / len(rules)] * len(rules)

    if len(weights) != len(rules):
        raise ValueError("weights must have the same length as rules")

    total_w = float(sum(weights))
    if total_w <= 0:
        raise ValueError("sum(weights) must be positive")

    loss = torch.zeros((), device=u_pred.device, dtype=u_pred.dtype)

    for rule_name, rule_weight in zip(rules, weights):
        taus, tau_weights = temporal_collocation_rule(
            rule=rule_name,
            device=u_pred.device,
            dtype=u_pred.dtype if u_pred.is_floating_point() else torch.float32,
            gegenbauer_lam=gegenbauer_lam,
            jacobi_alpha=jacobi_alpha,
            jacobi_beta=jacobi_beta,
        )

        loss_rule = torch.zeros((), device=u_pred.device, dtype=u_pred.dtype)

        for tau, tau_w in zip(taus, tau_weights):
            l_tau = physics_collocation_tau_L2_AC_neumann(
                u_in,
                u_pred,
                tau=float(tau),
                normalize=normalize,
            )
            loss_rule = loss_rule + tau_w * l_tau

        loss = loss + float(rule_weight) * loss_rule

    return loss / total_w



def physics_collocation_tau_L2_AC_neumann_twopoints(
    u_in,
    u_pred,
    tau=0.5 - 1.0 / (2.0 * math.sqrt(5.0)),
    normalize=True,
):
    """
    AC3D Neumann collocation at u_tau:

        R_tau = (u^{n+1} - u^n)/dt - RHS_AC_Neumann(u_tau)

    where
        u_tau = (1 - tau) u^n + tau u^{n+1}.

    Supports:
      - single-step prediction: u_pred shape (B,S,S,S,1)
      - multi-step rollout block: u_pred shape (B,S,S,S,T)

    Parameters
    ----------
    u_in : torch.Tensor
        Shape (B,S,S,S,1), last observed state.
    u_pred : torch.Tensor
        Shape (B,S,S,S,1) or (B,S,S,S,T), predicted state/block.
    tau : float
        Collocation point inside the time interval.
    normalize : bool
        If True, normalize temporal derivative and RHS per sample.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    dt, dx, eps2 = config.DT, config.DX, config.EPS2

    u0 = u_in.squeeze(-1).float()
    up = u_pred.float()

    # -----------------------------------------
    # Case 1: single-step prediction
    # -----------------------------------------
    if up.shape[-1] == 1:
        up1 = up.squeeze(-1)                     # (B,S,S,S)
        ut = (up1 - u0) / dt                     # (B,S,S,S)
        u_tau = (1.0 - tau) * u0 + tau * up1    # (B,S,S,S)

        rhs_tau = pde_rhs_ac_neumann(u_tau, dx, eps2)  # (B,S,S,S)

        if normalize:
            s_t = ut.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
            s_r = rhs_tau.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
            R = ut / s_t - rhs_tau / s_r
        else:
            R = ut - rhs_tau

        return (R ** 2).mean().to(u_pred.dtype)

    # -----------------------------------------
    # Case 2: multi-step rollout block
    # -----------------------------------------
    B, Sx, Sy, Sz, T = up.shape

    # build consecutive pairs:
    # [u^n, u^{n+1}, ..., u^{n+T}]
    u_all = torch.cat([u0.unsqueeze(-1), up], dim=-1)   # (B,S,S,S,T+1)
    u_prev = u_all[..., :-1]                             # (B,S,S,S,T)
    u_next = u_all[..., 1:]                              # (B,S,S,S,T)

    ut = (u_next - u_prev) / dt
    u_tau = (1.0 - tau) * u_prev + tau * u_next

    # evaluate RHS for all time slices efficiently
    u_tau_flat = u_tau.permute(0, 4, 1, 2, 3).reshape(B * T, Sx, Sy, Sz)
    rhs_tau_flat = pde_rhs_ac_neumann(u_tau_flat, dx, eps2)
    rhs_tau = rhs_tau_flat.view(B, T, Sx, Sy, Sz).permute(0, 2, 3, 4, 1)

    if normalize:
        s_t = ut.pow(2).mean((1, 2, 3, 4), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean((1, 2, 3, 4), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau

    return (R ** 2).mean().to(u_pred.dtype)


def _ac_pairwise_states_from_rollout(u_hist_last, u_pred):
    """
    Build consecutive state pairs from rollout:
      [u^n, u^{n+1}, ..., u^{n+T}]
    Returns:
      u_prev, u_next with shape (B,S,S,S,T)
    """
    if u_hist_last.dim() != 5 or u_hist_last.shape[-1] != 1:
        raise ValueError("u_hist_last must have shape (B,S,S,S,1)")
    if u_pred.dim() != 5:
        raise ValueError("u_pred must have shape (B,S,S,S,T)")

    u0 = u_hist_last.float()
    up = u_pred.float()

    u_all = torch.cat([u0, up], dim=-1)   # (B,S,S,S,T+1)
    u_prev = u_all[..., :-1]              # (B,S,S,S,T)
    u_next = u_all[..., 1:]               # (B,S,S,S,T)
    return u_prev, u_next

def chemical_potential_ac_neumann(u, dx, eps2):
    """
    Allen-Cahn chemical potential / variational derivative:
        mu = -eps2 * Lap(u) + (u^3 - u)

    u: (B,S,S,S)
    returns: (B,S,S,S)
    """
    lap_u = laplacian_neumann_cosine_3d(u, dx)
    return -eps2 * lap_u + (u**3 - u)


def ac_gradient_flow_alignment_loss(u_hist_last, u_pred, dt, dx, eps2, eps=1e-8):
    """
    Pure-physics gradient-flow direction alignment for Allen-Cahn rollout.

    Allen-Cahn is an L2-gradient flow:
        u_t = -mu,
    where
        mu = -eps2 * Lap(u) + (u^3 - u).

    We enforce that the predicted time increment points in the same
    direction as the physical descent direction -mu.

    Parameters
    ----------
    u_hist_last : torch.Tensor
        Shape (B,S,S,S,1), last observed state before rollout.
    u_pred : torch.Tensor
        Shape (B,S,S,S,T), rollout prediction block.
    dt, dx, eps2 : float
        Physical parameters.
    eps : float
        Small number for numerical stability.

    Returns
    -------
    torch.Tensor
        Scalar loss. Smaller is better.
    """
    if u_hist_last.dim() != 5 or u_hist_last.shape[-1] != 1:
        raise ValueError("u_hist_last must have shape (B,S,S,S,1)")
    if u_pred.dim() != 5:
        raise ValueError("u_pred must have shape (B,S,S,S,T)")

    u_prev, u_next = _ac_pairwise_states_from_rollout(u_hist_last, u_pred)  # (B,S,S,S,T)

    B, Sx, Sy, Sz, T = u_next.shape

    # predicted time direction
    d_t = (u_next - u_prev) / dt   # (B,S,S,S,T)

    # evaluate chemical potential at u_next
    u_next_flat = u_next.permute(0, 4, 1, 2, 3).reshape(B * T, Sx, Sy, Sz)
    mu_flat = chemical_potential_ac_neumann(u_next_flat, dx, eps2)
    mu = mu_flat.view(B, T, Sx, Sy, Sz).permute(0, 2, 3, 4, 1)  # (B,S,S,S,T)

    # physical gradient-flow direction
    g_t = -mu

    # cosine alignment per sample and per rollout step
    d_flat = d_t.permute(0, 4, 1, 2, 3).reshape(B * T, -1)
    g_flat = g_t.permute(0, 4, 1, 2, 3).reshape(B * T, -1)

    inner = torch.sum(d_flat * g_flat, dim=1)
    d_norm = torch.sqrt(torch.sum(d_flat ** 2, dim=1) + eps)
    g_norm = torch.sqrt(torch.sum(g_flat ** 2, dim=1) + eps)

    cos_sim = inner / (d_norm * g_norm + eps)

    # want cos_sim -> 1
    loss = 1.0 - cos_sim.mean()

    return loss.to(u_pred.dtype)


def ac_gradient_flow_alignment_loss_weighted(
    u_hist_last,
    u_pred,
    dt,
    dx,
    eps2,
    end_weight=2.0,
    eps=1e-8,
):
    """
    Same as ac_gradient_flow_alignment_loss, but later rollout steps
    receive larger weight.

    Useful if your dominant error is long-time drift.
    """
    if u_hist_last.dim() != 5 or u_hist_last.shape[-1] != 1:
        raise ValueError("u_hist_last must have shape (B,S,S,S,1)")
    if u_pred.dim() != 5:
        raise ValueError("u_pred must have shape (B,S,S,S,T)")

    u_prev, u_next = _ac_pairwise_states_from_rollout(u_hist_last, u_pred)

    B, Sx, Sy, Sz, T = u_next.shape

    d_t = (u_next - u_prev) / dt

    u_next_flat = u_next.permute(0, 4, 1, 2, 3).reshape(B * T, Sx, Sy, Sz)
    mu_flat = chemical_potential_ac_neumann(u_next_flat, dx, eps2)
    mu = mu_flat.view(B, T, Sx, Sy, Sz).permute(0, 2, 3, 4, 1)

    g_t = -mu

    # per-time cosine similarity averaged over batch
    d_bt = d_t.permute(0, 4, 1, 2, 3).reshape(B, T, -1)
    g_bt = g_t.permute(0, 4, 1, 2, 3).reshape(B, T, -1)

    inner = torch.sum(d_bt * g_bt, dim=2)                   # (B,T)
    d_norm = torch.sqrt(torch.sum(d_bt ** 2, dim=2) + eps)  # (B,T)
    g_norm = torch.sqrt(torch.sum(g_bt ** 2, dim=2) + eps)  # (B,T)

    cos_sim = inner / (d_norm * g_norm + eps)               # (B,T)
    loss_t = 1.0 - cos_sim.mean(dim=0)                      # (T,)

    w = torch.linspace(
        1.0, float(end_weight), T,
        device=u_pred.device,
        dtype=u_pred.dtype
    )

    return ((w * loss_t).sum() / w.sum()).to(u_pred.dtype)






### @@@

def semi_implicit_step_AC_neumann_single(u_in, dt, dx, eps2):
    """
    One AC semi-implicit step for input of shape (B,S,S,S,1).
    Returns shape (B,S,S,S,1).
    """
    u_cur = u_in.squeeze(-1).float()
    _, nx, ny, nz = u_cur.shape

    k2 = _neumann_k2_grid(nx, ny, nz, dx, u_cur.device, u_cur.dtype)
    denom = 1.0 + dt * k2

    nl = u_cur**3 - u_cur
    u_hat = dct3_neumann(u_cur)
    nl_hat = dct3_neumann(nl)

    u_next_hat = (u_hat - (dt / eps2) * nl_hat) / denom
    u_next = idct3_neumann(u_next_hat)

    return u_next.unsqueeze(-1).to(u_in.dtype)

#
def ac_weak_collocation_residual_neumann(
    u_hist_last,
    u_pred,
    dt,
    dx,
    eps2,
    taus=(0.0, 1.0),
    weights=(0.5, 0.5),
    normalize=True,
):
    """
    Weak Gauss-Lobatto collocation residual for Allen-Cahn with Neumann BC.

    By default this uses the 2-point Gauss-Lobatto rule on [0,1]:
        tau = 0, 1
        weight = 1/2, 1/2

    Parameters
    ----------
    u_hist_last : torch.Tensor
        Shape (B,S,S,S,1), last observed state before rollout.
    u_pred : torch.Tensor
        Shape (B,S,S,S,T), predicted rollout block.
    dt, dx, eps2 : float
        Time step, grid spacing, epsilon^2.
    taus : tuple
        Collocation points inside each step.
        Default: (0.0, 1.0)  -> 2-point Gauss-Lobatto.
    weights : tuple
        Quadrature weights corresponding to taus.
        Default: (0.5, 0.5)
    normalize : bool
        If True, normalize temporal derivative and PDE RHS scales.

    Returns
    -------
    torch.Tensor
        Scalar collocation loss.
    """
    if u_hist_last.dim() != 5 or u_hist_last.shape[-1] != 1:
        raise ValueError("u_hist_last must have shape (B,S,S,S,1)")
    if u_pred.dim() != 5:
        raise ValueError("u_pred must have shape (B,S,S,S,T)")
    if len(taus) != len(weights):
        raise ValueError("taus and weights must have the same length")

    B, Sx, Sy, Sz, T = u_pred.shape

    total_w = float(sum(weights))
    if total_w <= 0:
        raise ValueError("sum(weights) must be positive")

    loss = torch.zeros((), device=u_pred.device, dtype=u_pred.dtype)

    u_prev = u_hist_last
    for t in range(T):
        u_next = u_pred[..., t:t+1]  # (B,S,S,S,1)

        # finite-difference time derivative over the whole interval
        ut = (u_next - u_prev) / dt  # (B,S,S,S,1)

        step_loss = torch.zeros((), device=u_pred.device, dtype=u_pred.dtype)

        for tau, w in zip(taus, weights):
            tau = float(tau)

            # affine state inside the step
            u_tau = (1.0 - tau) * u_prev + tau * u_next  # (B,S,S,S,1)

            rhs_tau = pde_rhs_ac_neumann(
                u_tau.squeeze(-1), dx, eps2
            ).unsqueeze(-1)  # (B,S,S,S,1)

            if normalize:
                s_t = ut.pow(2).mean(dim=(1, 2, 3, 4), keepdim=True).sqrt().detach() + 1e-8
                s_r = rhs_tau.pow(2).mean(dim=(1, 2, 3, 4), keepdim=True).sqrt().detach() + 1e-8
                R = ut / s_t - rhs_tau / s_r
            else:
                R = ut - rhs_tau

            step_loss = step_loss + float(w) * (R ** 2).mean()

        loss = loss + step_loss / total_w
        u_prev = u_next

    return loss / T

########   @@@@ @@@@

# ============================================================
# Architecture helper functions for PhysicsGuidedTNO3d
# Switch by config.PROBLEM:
#   - "AC3D"
#   - "CH3D"
# ============================================================

def semi_implicit_step_CH_neumann_single(u_in, dt, dx, eps2):
    """
    One CH semi-implicit step for input of shape (B,S,S,S,1).
    Returns shape (B,S,S,S,1).
    """
    u_cur = u_in.squeeze(-1).float()
    _, nx, ny, nz = u_cur.shape

    k2 = _neumann_k2_grid(nx, ny, nz, dx, u_cur.device, u_cur.dtype)
    denom = 1.0 + dt * (2.0 * k2 + eps2 * (k2 ** 2))

    nl = u_cur**3 - 3.0 * u_cur
    u_hat = dct3_neumann(u_cur)
    nl_hat = dct3_neumann(nl)

    rhs_hat = u_hat - dt * k2 * nl_hat
    u_next_hat = rhs_hat / (denom + 1e-12)
    u_next = idct3_neumann(u_next_hat)

    return u_next.unsqueeze(-1).to(u_in.dtype)


# ----------------------------------------------------------
# Physics scaffold
# ----------------------------------------------------------
def phys_step(model, u):
    """
    u: (B, X, Y, Z, 1)
    returns: (B, X, Y, Z, 1)
    """
    if config.PROBLEM == "AC3D":
        p = semi_implicit_step_AC_neumann_single(u, model.dt, model.dx, model.eps2)
    elif config.PROBLEM == "CH3D":
        p = semi_implicit_step_CH_neumann_single(u, model.dt, model.dx, model.eps2)
    else:
        raise ValueError(f"Unknown PROBLEM '{config.PROBLEM}'")

    return p.detach() if model.detach_phys else p


# ----------------------------------------------------------
# Midpoint raw defect
# ----------------------------------------------------------
def midpoint_raw_defect(model, u_prev, u_next):
    """
    R = (u_next - u_prev)/dt - f((u_prev + u_next)/2)

    u_prev, u_next: (B, X, Y, Z, 1)
    returns:
        R_cf: (B, 1, X, Y, Z)
    """
    u0 = u_prev.squeeze(-1).float()
    u1 = u_next.squeeze(-1).float()

    ut = (u1 - u0) / model.dt
    u_mid = 0.5 * (u0 + u1)

    if config.PROBLEM == "AC3D":
        rhs_mid = pde_rhs_ac_neumann(u_mid, model.dx, model.eps2)
        R = ut - rhs_mid

    elif config.PROBLEM == "CH3D":
        rhs_mid = pde_rhs_ch_neumann(u_mid, model.dx, model.eps2)
        R = ut - rhs_mid
        R = _remove_spatial_mean_4d(R)

    else:
        raise ValueError(f"Unknown PROBLEM '{config.PROBLEM}'")

    # mild smoothing
    R = 0.9 * R + 0.1 * F.avg_pool3d(
        R.unsqueeze(1), kernel_size=3, stride=1, padding=1
    ).squeeze(1)

    return R.unsqueeze(1).to(u_next.dtype)


# ----------------------------------------------------------
# Collocation correction direction
# ----------------------------------------------------------
def collocation_direction(model, u_prev, u_base):
    """
    Pure-physics collocation correction:
        delta_coll = -dt * R_mid
    scaled relative to the physical step magnitude.
    """
    R_cf = midpoint_raw_defect(model, u_prev, u_base)  # (B,1,X,Y,Z)
    delta_cf = -model.dt * R_cf

    phys_inc_cf = (u_base - u_prev).permute(0, 4, 1, 2, 3)  # (B,1,X,Y,Z)

    rms_phys = phys_inc_cf.pow(2).mean((1, 2, 3, 4), keepdim=True).sqrt().detach() + 1e-8
    rms_d = delta_cf.pow(2).mean((1, 2, 3, 4), keepdim=True).sqrt().detach() + 1e-8

    delta_cf = (delta_cf / rms_d) * (model.colloc_scale * rms_phys)
    return delta_cf.to(u_base.dtype)


# ----------------------------------------------------------
# Collocation score
# ----------------------------------------------------------
def collocation_score(model, u_prev, u_next):
    """
    Lower is better.
    """
    R_cf = midpoint_raw_defect(model, u_prev, u_next)
    R = R_cf.squeeze(1)
    return R.pow(2).mean((1, 2, 3))  # (B,)


# ----------------------------------------------------------
# Collocation candidate selection
# ----------------------------------------------------------
def select_candidate(model, u_prev, u_base, delta_coll, t):
    """
    Build collocation-corrected candidates and softly select among them.
    """
    device = u_base.device
    dtype = u_base.dtype

    alpha_t = torch.sigmoid(model.colloc_gates[t])
    eta_vals = torch.tensor([0.0, 0.5, 1.0, 1.5], device=device, dtype=dtype)

    candidates = []
    scores = []

    temp = F.softplus(model.score_temp[t]) + 1e-6

    for eta in eta_vals:
        u_k = u_base + alpha_t * eta * delta_coll.permute(0, 2, 3, 4, 1)
        candidates.append(u_k)
        scores.append(collocation_score(model, u_prev, u_k))

    scores = torch.stack(scores, dim=1)  # (B,K)
    weights = F.softmax(-temp * scores, dim=1)

    u_final = 0.0
    for k, u_k in enumerate(candidates):
        w = weights[:, k].view(-1, 1, 1, 1, 1)
        u_final = u_final + w * u_k

    return u_final.to(u_base.dtype)


# ----------------------------------------------------------
# Energy gate
# ----------------------------------------------------------
def energy_gate(model, u_prev, u_coll, p_t, t):
    """
    Scalar gate for energy correction.
    Larger when the current state is less dissipative than the scaffold
    or too far from scaffold energy level.
    """
    if config.PROBLEM == "AC3D":
        E_prev = total_free_energy_AC_neumann(u_prev, model.dx, model.eps2)  # (B,)
        E_coll = total_free_energy_AC_neumann(u_coll, model.dx, model.eps2)  # (B,)
        E_phys = total_free_energy_AC_neumann(p_t, model.dx, model.eps2)     # (B,)

    elif config.PROBLEM == "CH3D":
        E_prev = total_free_energy_CH_neumann(u_prev, model.dx, model.eps2)  # (B,)
        E_coll = total_free_energy_CH_neumann(u_coll, model.dx, model.eps2)  # (B,)
        E_phys = total_free_energy_CH_neumann(p_t, model.dx, model.eps2)     # (B,)

    else:
        raise ValueError(f"Unknown PROBLEM '{config.PROBLEM}'")

    dE_coll = E_prev - E_coll
    dE_phys = E_prev - E_phys

    scale = dE_phys.abs().detach() + 0.05 * E_prev.abs().detach() + 1e-6

    lack_of_drop = torch.relu(dE_phys - dE_coll) / scale
    energy_increase = torch.relu(E_coll - E_prev) / scale
    scaffold_gap = (E_coll - E_phys).abs() / scale

    signal = lack_of_drop + 0.45 * energy_increase + 0.25 * scaffold_gap

    strength = F.softplus(model.energy_gate_strength[t]) + 1e-6
    gate = torch.tanh(strength * signal)  # in [0,1)

    return gate.view(-1, 1, 1, 1, 1).to(u_coll.dtype)


# ----------------------------------------------------------
# Energy confidence
# ----------------------------------------------------------
def energy_confidence(model, u_prev, u_coll, p_t):
    """
    Measures how much energy correction is needed.
    """
    if config.PROBLEM == "AC3D":
        E_prev = total_free_energy_AC_neumann(u_prev, model.dx, model.eps2)
        E_coll = total_free_energy_AC_neumann(u_coll, model.dx, model.eps2)
        E_phys = total_free_energy_AC_neumann(p_t, model.dx, model.eps2)

    elif config.PROBLEM == "CH3D":
        E_prev = total_free_energy_CH_neumann(u_prev, model.dx, model.eps2)
        E_coll = total_free_energy_CH_neumann(u_coll, model.dx, model.eps2)
        E_phys = total_free_energy_CH_neumann(p_t, model.dx, model.eps2)

    else:
        raise ValueError(f"Unknown PROBLEM '{config.PROBLEM}'")

    dE_coll = E_prev - E_coll
    dE_phys = E_prev - E_phys

    scale = dE_phys.abs().detach() + 0.05 * E_prev.abs().detach() + 1e-6

    mismatch = (dE_coll - dE_phys).abs() / scale
    scaffold_gap = (E_coll - E_phys).abs() / scale

    w = 0.7 * mismatch + 0.3 * scaffold_gap
    return torch.clamp(w, 0.0, 1.0).view(-1, 1, 1, 1, 1).to(u_coll.dtype)


# ----------------------------------------------------------
# Energy correction direction
# ----------------------------------------------------------
def energy_direction(model, u_prev, u_coll, p_t, delta_coll, t):
    """
    Pure-physics energy correction:
    blend of
      - PDE gradient-flow direction
      - pull-back to scaffold

    scaled relative to the physical step magnitude,
    then clamped relative to collocation strength.
    """
    u = u_coll.squeeze(-1).float()

    if config.PROBLEM == "AC3D":
        rhs = pde_rhs_ac_neumann(u, model.dx, model.eps2)

    elif config.PROBLEM == "CH3D":
        rhs = pde_rhs_ch_neumann(u, model.dx, model.eps2)
        rhs = _remove_spatial_mean_4d(rhs)

    else:
        raise ValueError(f"Unknown PROBLEM '{config.PROBLEM}'")

    d_rhs = (model.dt * rhs).unsqueeze(1)                # (B,1,X,Y,Z)
    d_scaf = (p_t - u_coll).permute(0, 4, 1, 2, 3)       # (B,1,X,Y,Z)

    mix = torch.sigmoid(model.energy_mix[t])
    delta_cf = mix * d_rhs + (1.0 - mix) * d_scaf

    phys_inc_cf = (p_t - u_prev).permute(0, 4, 1, 2, 3)
    rms_phys = phys_inc_cf.pow(2).mean((1, 2, 3, 4), keepdim=True).sqrt().detach() + 1e-8
    rms_d = delta_cf.pow(2).mean((1, 2, 3, 4), keepdim=True).sqrt().detach() + 1e-8
    delta_cf = (delta_cf / rms_d) * (model.energy_scale * rms_phys)

    # clamp relative to collocation correction
    rms_coll = delta_coll.pow(2).mean((1, 2, 3, 4), keepdim=True).sqrt().detach() + 1e-8
    rms_eng = delta_cf.pow(2).mean((1, 2, 3, 4), keepdim=True).sqrt().detach() + 1e-8

    max_ratio = 0.35
    scale = torch.minimum(torch.ones_like(rms_eng), max_ratio * rms_coll / rms_eng)
    delta_cf = delta_cf * scale

    return delta_cf.to(u_coll.dtype)
















#####
####

# -------------------------
# FFT helpers and operators
# -------------------------
@torch.no_grad()
def _fft_wavenumbers_3d(nx, ny, nz, dx):
    kx = torch.fft.fftfreq(nx, d=dx)
    ky = torch.fft.fftfreq(ny, d=dx)
    kz = torch.fft.fftfreq(nz, d=dx)
    kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    minus_k2 = -((2*np.pi)**2) * (kx**2 + ky**2 + kz**2)
    return minus_k2

# add near your other helpers
def _k_spectrum(nx, ny, nz, dx, device, dtype=torch.float32):
    kx = torch.fft.fftfreq(nx, d=dx).to(device)
    ky = torch.fft.fftfreq(ny, d=dx).to(device)
    kz = torch.fft.fftfreq(nz, d=dx).to(device)
    kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    k2 = (2*np.pi)**2 * (kx**2 + ky**2 + kz**2)
    return k2.to(dtype)

# ---- existing k-spectrum helper is already present ----
# def _k_spectrum(...)

import torch, math
import torch.fft
import torch.nn.functional as F
import numpy as np
import config



def _fft_rk2(nx, ny, nz, dx, device, dtype=torch.float32):
    """Return k^2 (=(2π)^2 |k|^2) and a safe inverse (1/(k^2+ε0)) for H^{-1} norm."""
    k2 = _k_spectrum(nx, ny, nz, dx, device, dtype=dtype)  # (S,S,S)
    # small floor for k=0 to avoid division by zero; choose physical smallest mode ~ (2π/L)^2
    # L = S*dx; minimal nonzero k^2 is about (2π/L)^2. Use a small fraction of that as ε0.
    L = nx * dx
    eps_k2 = (2*np.pi / L)**2 * 1e-2
    inv_k2_safe = 1.0 / (k2 + eps_k2)
    return k2, inv_k2_safe

def _charbonnier_mean(x, eps=1e-8):
    # Robust L2 ~1 when x small, ~|x| when large; stabilizes max spikes.
    return torch.sqrt(x*x + eps).mean()

# === NEW: L2 Gauss–Lobatto collocation for AC3D (identical form to SH/PFC/MBE) ===
import math


def physics_collocation_tau_L2_AC(u_in, u_pred,
                                  tau=0.5 - 1.0/(2.0*math.sqrt(5.0)),
                                  normalize=True):
    """
    AC3D collocation at u_tau:
      R_tau = (u^{n+1}-u^n)/dt - RHS_AC(u_tau), scored in L2,
      with optional per-sample normalization (same as SH/PFC/MBE).
    """
    assert config.PROBLEM == 'AC3D'
    dt, dx = config.DT, config.DX

    u0 = u_in.squeeze(-1).float()   # (B,S,S,S)
    up = u_pred.squeeze(-1).float()
    ut = (up - u0) / dt
    u_tau = (1.0 - tau) * u0 + tau * up

    rhs_tau = pde_rhs(u_tau, dx, config.EPSILON_PARAM)  # -> AC RHS under AC3D

    if normalize:
        s_t = ut.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau
    return (R**2).mean().to(u_pred.dtype)

def physics_collocation_tau_L2_AC_Tout(u_in, u_pred,
                                  tau=0.5 - 1.0/(2.0*math.sqrt(5.0)),
                                  normalize=True):
    """
    AC3D collocation at u_tau:
      R_tau = (u^{n+1}-u^n)/dt - RHS_AC(u_tau), scored in L2.

    Extended to handle:
      u_pred: (B,S,S,S,1)  or  (B,S,S,S,T_out)

    For T_out > 1, we interpret u_pred[...,k] as u^{n+k} and build
    a temporal chain u^n, u^{n+1}, ..., u^{n+T_out}, computing
    residuals for each step.
    """
    assert config.PROBLEM == 'AC3D'
    dt, dx = config.DT, config.DX

    # u0: last known state, shape (B,S,S,S)
    u0 = u_in.squeeze(-1).float()   # (B,S,S,S)

    up = u_pred.float()
    # Case 1: original single-step case
    if up.shape[-1] == 1:
        up = up.squeeze(-1)  # (B,S,S,S)

        ut = (up - u0) / dt
        u_tau = (1.0 - tau) * u0 + tau * up     # (B,S,S,S)

        rhs_tau = pde_rhs(u_tau, dx, config.EPSILON_PARAM)  # (B,S,S,S)

        if normalize:
            s_t = ut.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
            s_r = rhs_tau.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
            R = ut / s_t - rhs_tau / s_r
        else:
            R = ut - rhs_tau

    # Case 2: multi-step prediction, up: (B,S,S,S,T_out)
    else:
        B, Sx, Sy, Sz, T = up.shape

        # Build the chain [u^n, u^{n+1}, ..., u^{n+T}]
        u0_exp = u0.unsqueeze(-1)                  # (B,Sx,Sy,Sz,1)
        u_all = torch.cat([u0_exp, up], dim=-1)    # (B,Sx,Sy,Sz,T+1)

        # Differences: u^{k} - u^{k-1} for k=1..T
        u_prev = u_all[..., :-1]   # (B,Sx,Sy,Sz,T)
        u_next = u_all[...,  1:]   # (B,Sx,Sy,Sz,T)

        ut = (u_next - u_prev) / dt               # (B,Sx,Sy,Sz,T)
        u_tau = (1.0 - tau) * u_prev + tau * u_next   # (B,Sx,Sy,Sz,T)

        # Flatten time into batch for pde_rhs
        u_tau_flat = u_tau.permute(0, 4, 1, 2, 3).reshape(B * T, Sx, Sy, Sz)
        rhs_tau_flat = pde_rhs(u_tau_flat, dx, config.EPSILON_PARAM)  # (B*T,Sx,Sy,Sz)
        rhs_tau = rhs_tau_flat.view(B, T, Sx, Sy, Sz).permute(0, 2, 3, 4, 1)
        # rhs_tau: (B,Sx,Sy,Sz,T)

        if normalize:
            s_t = ut.pow(2).mean((1,2,3,4), keepdim=True).sqrt().detach() + 1e-8
            s_r = rhs_tau.pow(2).mean((1,2,3,4), keepdim=True).sqrt().detach() + 1e-8
            R = ut / s_t - rhs_tau / s_r
        else:
            R = ut - rhs_tau

    return (R**2).mean().to(u_pred.dtype)


# === NEW: L2 Gauss–Lobatto collocation for CH3D (to match SH/PFC/MBE exactly) ===

def spec_high_energy(u, frac_cut=0.6):
    """
    Tiny CH3D-only regularizer: penalize energy in the top (1-frac_cut) fraction of modes.
    u: (B,S,S,S)
    """
    B, S, _, _ = u.shape
    uhat = torch.fft.rfftn(u, dim=(1,2,3))
    # radial mask in index space
    kx = torch.fft.fftfreq(S, d=1.0).to(u.device)
    ky = torch.fft.fftfreq(S, d=1.0).to(u.device)
    kz = torch.fft.rfftfreq(S, d=1.0).to(u.device)
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    r = torch.sqrt(KX*KX + KY*KY + KZ*KZ)
    rmax = r.max()
    mask = (r > frac_cut * rmax).float()
    energy = (uhat.real**2 + uhat.imag**2) * mask
    return energy.mean().to(u.dtype)



import torch
import torch.fft as tfft
import math

def _kx_grid(S, dx, device, dtype):
    # physical-size L = S*dx; match MATLAB’s 2π/L * [0..S/2, -S/2+1..-1]
    L = S * dx
    half = S // 2
    kvec = torch.cat([
        torch.arange(0, half + 1, device=device),
        torch.arange(-half + 1, 0, device=device)
    ], dim=0).to(dtype)
    return (2.0 * math.pi / L) * kvec  # (S,)
'''''
def semi_implicit_step_sh(u_in, dt, dx, eps):
    """
    One semi-implicit Swift–Hohenberg step that mirrors your MATLAB:
      s_hat = FFT(u/dt) - FFT(u^3) + 2 * (kxx+kyy+kzz) * FFT(u)
      v_hat = s_hat / (1/dt + (1 - eps) + (kxx+kyy+kzz)^2)
      u^{n+1} = IFFT(v_hat)
    Inputs:
      u_in: (B,S,S,S,1)  real
      dt, dx: scalars (float)
      eps: your config.EPSILON_PARAM (same as MATLAB 'epsilon', not eps^2)
    Returns:
      (B,S,S,S,1)
    """
    assert u_in.ndim == 5 and u_in.shape[-1] == 1
    u = u_in.squeeze(-1)

    B, Sx, Sy, Sz = u.shape
    device, dtype = u.device, u.dtype

    kx = _kx_grid(Sx, dx, device, dtype)
    ky = _kx_grid(Sy, dx, device, dtype)
    kz = _kx_grid(Sz, dx, device, dtype)
    kxx = kx**2; kyy = ky**2; kzz = kz**2
    Kxx, Kyy, Kzz = torch.meshgrid(kxx, kyy, kzz, indexing='ij')
    K2 = (Kxx + Kyy + Kzz)  # (S,S,S)

    U   = tfft.fftn(u, dim=(1,2,3))
    S1  = tfft.fftn(u / dt, dim=(1,2,3))
    Nl  = tfft.fftn(u**3,  dim=(1,2,3))
    s_hat = S1 - Nl + 2.0 * K2 * U

    denom = (1.0/dt) + (1.0 - float(eps)) + (K2**2)
    v_hat = s_hat / denom
    up = tfft.ifftn(v_hat, dim=(1,2,3)).real
    return up.unsqueeze(-1)
'''

def _semi_implicit_step_sh(u_in, dt, dx, eps):
    """
    SH3D teacher step (matches MATLAB):
      û^{n+1} = [ (1/dt) û^n - FFT((u^n)^3) + 2 k^2 û^n ] / [ 1/dt + (1-ε) + k^4 ]
    Notes:
      - no dealiasing on u^3 (to match the generator)
      - k^2 = (2π/L)^2 |k|^2 from _k_spectrum(...)
    """
    u0 = u_in.squeeze(-1).float()
    B, S, _, _ = u0.shape
    k2 = _k_spectrum(S, S, S, dx, u0.device)     # (2π/L)^2 |k|^2
    k4 = k2 * k2

    u0_hat = torch.fft.fftn(u0, dim=(1,2,3))
    u3_hat = torch.fft.fftn(u0**3, dim=(1,2,3))  # no dealias (data)

    denom  = (1.0/dt) + (1.0 - eps) + k4
    numer  = (1.0/dt) * u0_hat - u3_hat + 2.0 * k2 * u0_hat

    u1_hat = numer / denom
    u1     = torch.fft.ifftn(u1_hat, dim=(1,2,3)).real
    return u1.unsqueeze(-1).to(u_in.dtype)


###### CH3D

def pde_rhs_CH(u, dx, eps2):
    """
    Standard CH3D RHS (continuous PDE):
    u_t = Δ μ,   μ = -eps2 Δ u + (u^3 - u)
    """
    lap_u = laplacian_fourier_3d_phys(u, dx)
    mu    = -eps2 * lap_u + (u**3 - 1.0*u)
    lap_mu = laplacian_fourier_3d_phys(mu, dx)
    return lap_mu

#####
def _get_k2_grid_cached(S, dx, device):
    """
    Cache |k|^2 grid per (S, dx, device) to avoid recomputing every batch.
    """
    key = (S, float(dx), str(device))
    if not hasattr(_get_k2_grid_cached, "_cache"):
        _get_k2_grid_cached._cache = {}
    cache = _get_k2_grid_cached._cache
    if key in cache:
        return cache[key]

    import math, torch
    k = 2.0 * math.pi * torch.fft.fftfreq(S, d=dx).to(device)  # (S,)
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k2 = (kx**2 + ky**2 + kz**2)
    cache[key] = k2
    return k2

def _split_hist_coords(x):
    """
    Split x into (hist, coords) assuming the last 3 channels are coordinates
    when present. Falls back to (x, None) if no coords.
    x: (B, S, S, S, C)
    """
    C = x.shape[-1]
    T_in = getattr(config, "T_IN_CHANNELS", 4)
    if C >= T_in + 3:
        hist   = x[..., :T_in]     # time/history channels
        coords = x[..., T_in:]     # 3 coord channels
    else:
        hist, coords = x, None
    return hist, coords
import torch
import torch.nn.functional as F



# ===== Swift–Hohenberg (SH3D) =====

def _r_sh():
    # MATLAB step uses (1 - epsilon) in the implicit linear term → r = 1 - epsilon
    return 1.0 - float(config.EPSILON_PARAM)

def energy_density_SH(u, dx):
    """
    Lyapunov density for SH (gradient flow, unit mobility):
      E(u) = ∫ [ 1/2 |(1 + ∇^2) u|^2  - (r/2) u^2  + (1/4) u^4 ] dx,
    with r = 1 - EPSILON_PARAM.
    """
    r = _r_sh()
    lap_u = laplacian_fourier_3d_phys(u, dx)
    one_plus_lap_u = u + lap_u
    term_lin = 0.5 * (one_plus_lap_u ** 2)
    term_r   = -0.5 * r * (u ** 2)
    term_nl  = 0.25 * (u ** 4)
    return term_lin + term_r + term_nl


def semi_implicit_step_sh(u_in, dt, dx, eps_param):
    """
    One semi-implicit SH step that matches the MATLAB generator:
        s_hat = FFT(u/dt) - FFT(u^3) + 2 k^2 FFT(u)
        v_hat = s_hat / (1/dt + (1 - epsilon) + k^4)
    """
    u0 = u_in.squeeze(-1).float()
    B, S, _, _ = u0.shape

    k = 2.0 * math.pi * torch.fft.fftfreq(S, d=dx).to(u0.device)
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k2 = (kx**2 + ky**2 + kz**2)
    k4 = k2**2

    U0   = torch.fft.fftn(u0, dim=(1,2,3))
    U3   = torch.fft.fftn(u0**3, dim=(1,2,3))
    sHat = U0 / dt - U3 + 2.0 * k2 * U0
    denom = (1.0/dt) + (1.0 - float(eps_param)) + k4
    VHat  = sHat / (denom + 1e-12)

    u1 = torch.fft.ifftn(VHat, dim=(1,2,3)).real
    return u1.unsqueeze(-1).to(u_in.dtype)

#####################################

@torch.no_grad()
def _fft_freqs(S, dx, device):
    k = 2*math.pi*torch.fft.fftfreq(S, d=dx).to(device)  # physical wavenumbers
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k2 = kx*kx + ky*ky + kz*kz
    return k2  # (S,S,S)


def low_k_mse(u_pred, u_ref, frac=0.45):
    """
    MSE between u_pred and u_ref restricted to low spatial frequencies.
    Shapes: (B,S,S,S,1).
    """
    up = u_pred.squeeze(-1).float()
    ur = u_ref.squeeze(-1).float()
    B, S, _, _ = up.shape

    Up = torch.fft.fftn(up, dim=(1,2,3))
    Ur = torch.fft.fftn(ur, dim=(1,2,3))

    fx = torch.fft.fftfreq(S, d=1.0).to(up.device)
    fy = torch.fft.fftfreq(S, d=1.0).to(up.device)
    fz = torch.fft.fftfreq(S, d=1.0).to(up.device)
    FX, FY, FZ = torch.meshgrid(fx, fy, fz, indexing='ij')
    r = torch.sqrt(FX*FX + FY*FY + FZ*FZ)
    rmax = r.max()
    mask = (r <= frac * rmax).to(Up.real.dtype)

    Dh = Up - Ur
    spec_mse = (Dh.real**2 + Dh.imag**2) * mask
    # normalize by # of active modes to keep scale stable across S
    denom = mask.sum().clamp_min(1.0)
    return spec_mse.sum() / denom
def low_k_mse_T_out(u_pred, u_ref, frac=0.45):
    """
    MSE between u_pred and u_ref restricted to low spatial frequencies.

    Supports shapes:
      - (B,S,S,S)          single-step
      - (B,S,S,S,1)        single-step with channel
      - (B,S,S,S,T_out)    multi-step (time/channel last)

    For multi-step, we treat each time slice as an extra batch element
    (so the loss aggregates over all steps).
    """
    up = u_pred
    ur = u_ref

    # --- ensure 5D with time/channel last ---
    if up.dim() == 4:
        up = up.unsqueeze(-1)   # (B,S,S,S,1)
    if ur.dim() == 4:
        ur = ur.unsqueeze(-1)

    assert up.dim() == 5 and ur.dim() == 5, "low_k_mse expects 4D or 5D inputs"

    # --- align time dimension ---
    T_p = up.shape[-1]
    T_r = ur.shape[-1]

    if T_p != T_r:
        # broadcast the one-step teacher across time if needed
        if T_p > 1 and T_r == 1:
            ur = ur.expand(*ur.shape[:-1], T_p)
            T_r = T_p
        elif T_r > 1 and T_p == 1:
            up = up.expand(*up.shape[:-1], T_r)
            T_p = T_r
        else:
            raise ValueError(f"low_k_mse: incompatible time dims {T_p} vs {T_r}")

    # now up, ur: (B,S,S,S,T) with same T
    B, S, _, _, T = up.shape

    # flatten time into batch: (B*T,S,S,S,1)
    up_flat = up.permute(0, 4, 1, 2, 3).reshape(B*T, S, S, S, 1)
    ur_flat = ur.permute(0, 4, 1, 2, 3).reshape(B*T, S, S, S, 1)

    up4 = up_flat.squeeze(-1).float()  # (B*T,S,S,S)
    ur4 = ur_flat.squeeze(-1).float()

    # --- original spectral low-k MSE on the flattened batch ---
    Up = torch.fft.fftn(up4, dim=(1, 2, 3))
    Ur = torch.fft.fftn(ur4, dim=(1, 2, 3))

    fx = torch.fft.fftfreq(S, d=1.0).to(up4.device)
    fy = torch.fft.fftfreq(S, d=1.0).to(up4.device)
    fz = torch.fft.fftfreq(S, d=1.0).to(up4.device)
    FX, FY, FZ = torch.meshgrid(fx, fy, fz, indexing='ij')
    r = torch.sqrt(FX*FX + FY*FY + FZ*FZ)
    rmax = r.max()
    mask = (r <= frac * rmax).to(Up.real.dtype)

    Dh = Up - Ur
    spec_mse = (Dh.real**2 + Dh.imag**2) * mask

    # same normalization style as before: by # of active modes only
    denom = mask.sum().clamp_min(1.0)
    return spec_mse.sum() / denom

# --- CH3D projection with mixed chemistry (uses the same MATLAB-matched kernel) ---

def _k_vectors(nx, ny, nz, dx, device, dtype=torch.float32):
    kx = torch.fft.fftfreq(nx, d=dx).to(device)
    ky = torch.fft.fftfreq(ny, d=dx).to(device)
    kz = torch.fft.fftfreq(nz, d=dx).to(device)
    kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    # no in-place ops
    kx = kx * (2*np.pi); ky = ky * (2*np.pi); kz = kz * (2*np.pi)
    return kx.to(dtype), ky.to(dtype), kz.to(dtype)

# --- H^{-1} distance between two scalar fields (B,S,S,S) ---
def hminus1_mse(u, v, dx, eps_floor_scale=1e-2):
    """
    Return mean_{batch,k} |Û-Ṽ|^2 / (k^2 + eps0), i.e., an H^{-1} metric.
    This emphasizes low-k agreement (CH3D gradient flow is H^{-1}).
    """
    with torch.amp.autocast(device_type='cuda', enabled=False):
        B, S, _, _ = u.shape
        # safe inverse k^2 (reuse your k-spectrum convention)
        kx = torch.fft.fftfreq(S, d=dx).to(u.device)
        ky = torch.fft.fftfreq(S, d=dx).to(u.device)
        kz = torch.fft.fftfreq(S, d=dx).to(u.device)
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        k2 = (2*np.pi)**2 * (KX**2 + KY**2 + KZ**2)
        # floor ~ small fraction of (2π/L)^2
        L = S * dx
        eps0 = (2*np.pi/L)**2 * eps_floor_scale
        inv_k2 = 1.0 / (k2 + eps0)

        d  = (u.float() - v.float())
        dh = torch.fft.fftn(d, dim=(1,2,3))
        pow_spec = dh.real**2 + dh.imag**2
        val = (pow_spec * inv_k2).mean()
        return val.to(u.dtype)


# --- Simple spectral low-pass (keep lowest frac of modes) ---
def lowpass_field(u, frac=0.35):
    """
    Keep modes with radius <= frac*rmax (index space). Good enough to
    isolate the coarse μ that drives CH3D coarsening.
    """
    with torch.amp.autocast(device_type='cuda', enabled=False):
        B, S, _, _ = u.shape
        fx = torch.fft.fftfreq(S, d=1.0).to(u.device)
        fy = torch.fft.fftfreq(S, d=1.0).to(u.device)
        fz = torch.fft.fftfreq(S, d=1.0).to(u.device)
        FX, FY, FZ = torch.meshgrid(fx, fy, fz, indexing='ij')
        r = torch.sqrt(FX*FX + FY*FY + FZ*FZ)
        rmax = r.max()
        mask = (r <= frac * rmax).float()

        uh = torch.fft.fftn(u.float(), dim=(1,2,3))
        ul = torch.fft.ifftn(uh * mask, dim=(1,2,3)).real
        return ul.to(u.dtype)


def interface_weight(u, dx, eps=1e-12):
    """
    Weight in [0,1] that highlights interfaces.
    We use normalized |∇u| as a proxy for interface location.
    """
    ux, uy, uz = grad_fourier(u, dx)     # spectral gradient (cheap & alias-free)
    g = torch.sqrt(ux*ux + uy*uy + uz*uz + eps)
    gmax = g.amax(dim=(1,2,3), keepdim=True) + 1e-12
    return (g / gmax).clamp(0, 1)



def laplacian_fourier_3d_phys(u, dx):
    with torch.amp.autocast(device_type='cuda', enabled=False):
        B, nx, ny, nz = u.shape
        minus_k2 = _fft_wavenumbers_3d(nx, ny, nz, dx).to(u.device).to(torch.float32)
        u32 = u.float()
        u_ft = torch.fft.fftn(u32, dim=[1,2,3])
        lap  = torch.fft.ifftn(minus_k2 * u_ft, dim=[1,2,3]).real
    return lap.to(u.dtype)

def biharmonic(u, dx):
    # ∇⁴u = ∇²(∇²u)
    return laplacian_fourier_3d_phys(laplacian_fourier_3d_phys(u, dx), dx)

def triharmonic(u, dx):
    # ∇⁶u = ∇²(∇⁴u)
    return laplacian_fourier_3d_phys(biharmonic(u, dx), dx)

def grad_fourier(u, dx):
    """
    Spectral gradient: returns (ux, uy, uz) same dtype/device as u
    """
    B, nx, ny, nz = u.shape
    kx, ky, kz = _k_vectors(nx, ny, nz, dx, u.device, dtype=torch.float32)
    uhat = torch.fft.fftn(u.float(), dim=[1,2,3])
    i = torch.complex(torch.tensor(0.0, device=u.device), torch.tensor(1.0, device=u.device))
    ux = torch.fft.ifftn(i * kx * uhat, dim=[1,2,3]).real
    uy = torch.fft.ifftn(i * ky * uhat, dim=[1,2,3]).real
    uz = torch.fft.ifftn(i * kz * uhat, dim=[1,2,3]).real
    return ux.to(u.dtype), uy.to(u.dtype), uz.to(u.dtype)
def div_fourier(vx, vy, vz, dx):
    """
    Spectral divergence of vector field v = (vx, vy, vz)
    """
    B, nx, ny, nz = vx.shape
    kx, ky, kz = _k_vectors(nx, ny, nz, dx, vx.device, dtype=torch.float32)
    i = torch.complex(torch.tensor(0.0, device=vx.device), torch.tensor(1.0, device=vx.device))
    vxhat = torch.fft.fftn(vx.float(), dim=[1,2,3])
    vyhat = torch.fft.fftn(vy.float(), dim=[1,2,3])
    vzhat = torch.fft.fftn(vz.float(), dim=[1,2,3])
    div_hat = i * (kx * vxhat + ky * vyhat + kz * vzhat)
    return torch.fft.ifftn(div_hat, dim=[1,2,3]).real.to(vx.dtype)

# -------------------------
# Dealiasing (kept)
# -------------------------
def dealias_two_thirds(u):
    S = u.shape[1]
    kcut = S // 3
    filt = torch.zeros((S, S, S//2 + 1), device=u.device, dtype=torch.float32)
    filt[:2*kcut, :2*kcut, :kcut+1] = 1.0
    uhat = torch.fft.rfftn(u, dim=(1,2,3))
    return torch.fft.irfftn(uhat * filt, s=(S,S,S), dim=(1,2,3)).real

# -------------------------
# RHS dispatch per PROBLEM

def _rhs_ac3d(u, dx, eps2):
    # Dataset-consistent: NO dealiasing in the cubic (matches MATLAB generator)
    lap_u = laplacian_fourier_3d_phys(u, dx)
    return lap_u - (1.0/eps2) * (u**3 - u)


def _rhs_mbe3d(u, dx, eps):
    """
    MBE3D (slope-selection, data-consistent):
        u_t = -Δ u  -  eps ∇^4 u  +  ∇·(|∇u|^2 ∇u)
             = -ε Δ^2 u - ∇·((1 - |∇u|^2)∇u)
    """
    ux, uy, uz = grad_fourier(u, dx)
    s = ux*ux + uy*uy + uz*uz
    vx, vy, vz = s*ux, s*uy, s*uz
    div_f = div_fourier(vx, vy, vz, dx)    # ∇·(|∇u|^2 ∇u)
    lap   = laplacian_fourier_3d_phys(u, dx)
    bi    = biharmonic(u, dx)
    return -lap - eps * bi + div_f



def _rhs_pfc3d(u, dx, eps):
    # u_t = (1-ε)∇²u + 2∇⁴u + ∇⁶u - ∇²(u^3)
    lap_u = laplacian_fourier_3d_phys(u, dx)
    bi_u  = biharmonic(u, dx)
    tri_u = triharmonic(u, dx)
    # match data: NO dealiasing in the cubic
    lap_u3 = laplacian_fourier_3d_phys(u**3, dx)
    return (1.0 - eps) * lap_u + 2.0 * bi_u + tri_u - lap_u3

def _rhs_sh3d(u, dx, eps):
    """
    Swift–Hohenberg (data-consistent split):
        u_t = (1-ε) u - 2 Δ u - Δ^2 u - u^3
    NOTE:
      • NO dealiasing on u^3 (matches the MATLAB generator).
      • Clamp inside the cubic to avoid overflow when the net overshoots early.
    """
    lap_u = laplacian_fourier_3d_phys(u, dx)     # Δu
    bi_u  = biharmonic(u, dx)                    # Δ^2 u
    u_safe = torch.clamp(u, -5.0, 5.0)           # safe cubic; no in-place to keep graph
    return (1.0 - eps) * u - 2.0 * lap_u - bi_u - (u_safe ** 3)

def semi_implicit_step_pfc(u_in, dt, dx, eps):
    """
    One PFC3D semi-implicit step (matches your MATLAB generator).
    Inputs:  u_in (B,S,S,S,1), dt, dx, eps
    Returns: (B,S,S,S,1)
    """
    u0 = u_in.squeeze(-1).float()
    B, S, _, _ = u0.shape
    k2 = _k_spectrum(S, S, S, dx, u0.device)  # (2π/L)^2 |k|^2  >= 0

    U0     = torch.fft.fftn(u0,    dim=(1,2,3))
    U3_hat = torch.fft.fftn(u0**3, dim=(1,2,3))  # no dealias (data)

    numer = (1.0/dt) * U0 - k2 * U3_hat + 2.0 * (k2**2) * U0
    denom = (1.0/dt) + (1.0 - eps) * k2 + (k2**3)

    U1 = numer / denom
    u1 = torch.fft.ifftn(U1, dim=(1,2,3)).real
    return u1.unsqueeze(-1).to(u_in.dtype)


def _scheme_residual_fourier_pfc(u0, up, dt, dx, eps):
    # u0, up: (B,S,S,S)
    k2 = _k_spectrum(u0.shape[1], u0.shape[2], u0.shape[3], dx, u0.device)
    U0     = torch.fft.fftn(u0,    dim=(1,2,3))
    UP     = torch.fft.fftn(up,    dim=(1,2,3))
    U3_hat = torch.fft.fftn(u0**3, dim=(1,2,3))    # data-consistent (no dealias)

    denom  = (1.0/dt) + (1.0 - eps)*k2 + (k2**3)
    rhs    = (1.0/dt)*U0 - k2*U3_hat + 2.0*(k2**2)*U0
    rhat   = denom*UP - rhs
    rhat   = rhat / (denom + 1e-12)                # precondition like SH
    return (rhat.real**2 + rhat.imag**2).mean()


def pde_rhs(u, dx, eps_param):
    """
    Generic RHS(u) for the active problem such that u_t = RHS(u).
    """
    P = config.PROBLEM
    if P == 'AC3D':
        return _rhs_ac3d(u, dx, config.EPS2)
    elif P == 'CH3D':
        return _rhs_ch3d(u, dx, eps_param)
    elif P == 'SH3D':
        return _rhs_sh3d(u, dx, eps_param)
    elif P == 'MBE3D':
        return _rhs_mbe3d(u, dx, eps_param)
    elif P == 'PFC3D':
        return _rhs_pfc3d(u, dx, eps_param)
    else:
        raise ValueError(f"Unknown PROBLEM '{P}'")

# -------------------------
# Allen–Cahn chemical potential alias (compat)
# For non-AC problems, we return RHS(u) so your debug prints still work.
# -------------------------
def mu_ac(u, dx, eps2, dealias=True):
    if config.PROBLEM == 'AC3D':
        lap_u = laplacian_fourier_3d_phys(u, dx)
        if dealias:
            u = dealias_two_thirds(u)
        return lap_u - (1.0/eps2) * (u**3 - u)
    # For other PDEs, use RHS(u) as a general “μ-like” term for logging
    return pde_rhs(u, dx, config.EPSILON_PARAM)

# -------------------------
# Physics residuals (now generic)
# -------------------------
def physics_residual_matlab(u_in, u_pred):
    # Preserved for backward compat (Allen–Cahn form). Kept but now generic.
    dt, dx = config.DT, config.DX
    u0 = u_in.squeeze(-1); up = u_pred.squeeze(-1)
    ut = (up - u0) / dt
    mu = pde_rhs(up, dx, config.EPSILON_PARAM)
    R = ut - mu
    mse_phys = F.mse_loss(R, torch.zeros_like(R))
    debug_ut_mse = torch.mean(ut**2)
    debug_muspatial_mse = config.DEBUG_MU_SCALE * torch.mean(mu**2)
    return mse_phys, debug_ut_mse, debug_muspatial_mse

def physics_residual_normalized(u_in, u_pred):
    dt, dx = config.DT, config.DX
    u0 = u_in.squeeze(-1); up = u_pred.squeeze(-1)
    ut = (up - u0) / dt
    mu = pde_rhs(up, dx, config.EPSILON_PARAM)
    s_t  = ut.pow(2).mean(dim=(1,2,3), keepdim=True).sqrt().detach() + 1e-8
    s_mu = mu.pow(2).mean(dim=(1,2,3), keepdim=True).sqrt().detach() + 1e-8
    R_tilde = ut / s_t - mu / s_mu
    loss = R_tilde.pow(2).mean()
    return loss, s_t.mean(), s_mu.mean()

def physics_residual_midpoint(u_in, u_pred):
    dt, dx = config.DT, config.DX
    u0 = u_in.squeeze(-1); up = u_pred.squeeze(-1)
    um = 0.5 * (u0 + up)
    ut = (up - u0) / dt
    mu_m = pde_rhs(um, dx, config.EPSILON_PARAM)
    Rm = ut - mu_m
    return (Rm**2).mean()

# -------------------------
# Semi-implicit / teacher step
# -------------------------
def semi_implicit_step(u_in, dt, dx, eps2):
    """
    AC3D: keep your previous semi-implicit.
    Others: safe explicit Euler teacher (one-step) using RHS(u^n).
    """
    P = config.PROBLEM
    if P == 'AC3D':
        # original AC3D semi-implicit step
        u0 = u_in.squeeze(-1).float()
        B,S,_,_ = u0.shape[:4]
        kx = torch.fft.fftfreq(S, d=dx).to(u0.device)
        ky = torch.fft.fftfreq(S, d=dx).to(u0.device)
        kz = torch.fft.fftfreq(S, d=dx).to(u0.device)
        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
        k2 = (2*np.pi)**2 * (kx**2 + ky**2 + kz**2)
        nl = u0**3 - u0
        u0_hat = torch.fft.fftn(u0, dim=[1,2,3])
        nl_hat = torch.fft.fftn(nl, dim=[1,2,3])
        num = u0_hat - (dt/eps2) * nl_hat
        den = (1.0 + dt * k2)
        u1_hat = num / den
        u1 = torch.fft.ifftn(u1_hat, dim=[1,2,3]).real

        return u1.unsqueeze(-1)

    else:
        # fallback (unchanged) explicit Euler
        u0 = u_in.squeeze(-1)
        rhs = pde_rhs(u0, dx, config.EPSILON_PARAM)
        return (u0 + dt * rhs).unsqueeze(-1)

def semi_implicit_step_T_out(u_in, dt, dx, eps2):
    """
    AC3D: semi-implicit scheme.
    - If T_OUT == 1: behaves as before, returns one step u^{n+1}.
    - If T_OUT > 1 : returns [u^{n+1}, ..., u^{n+T_OUT}] in the last dim.

    Others: explicit Euler teacher (also extended to T_OUT steps).
    """
    P = config.PROBLEM
    T_out = int(getattr(config, "T_OUT", 1))

    # last known state u^n: (B,S,S,S)
    u0 = u_in.squeeze(-1).float()

    if P == 'AC3D':
        # original AC3D semi-implicit, extended to multiple steps
        B, S, _, _ = u0.shape[:4]

        # spectral grid (same for all steps)
        kx = torch.fft.fftfreq(S, d=dx).to(u0.device)
        ky = torch.fft.fftfreq(S, d=dx).to(u0.device)
        kz = torch.fft.fftfreq(S, d=dx).to(u0.device)
        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
        k2 = (2 * np.pi) ** 2 * (kx**2 + ky**2 + kz**2)
        den = (1.0 + dt * k2)

        u_cur = u0
        steps = []
        for _ in range(T_out):
            nl = u_cur**3 - u_cur
            u0_hat = torch.fft.fftn(u_cur, dim=[1, 2, 3])
            nl_hat = torch.fft.fftn(nl, dim=[1, 2, 3])
            num = u0_hat - (dt / eps2) * nl_hat
            u1_hat = num / den
            u_next = torch.fft.ifftn(u1_hat, dim=[1, 2, 3]).real  # (B,S,S,S)

            steps.append(u_next)
            u_cur = u_next

        u_all = torch.stack(steps, dim=-1)  # (B,S,S,S,T_out)
        return u_all  # for T_out=1, shape is (B,S,S,S,1)

    else:
        # fallback explicit Euler, extended to multiple steps
        u_cur = u0
        steps = []
        for _ in range(T_out):
            rhs = pde_rhs(u_cur, dx, config.EPSILON_PARAM)
            u_next = u_cur + dt * rhs
            steps.append(u_next)
            u_cur = u_next

        u_all = torch.stack(steps, dim=-1)  # (B,S,S,S,T_out)
        return u_all

# -------------------------
# Energy utilities
# For AC3D use your original. Otherwise return zero penalty (no-op).
# -------------------------
def energy_density(u, dx, eps2):
    lap = laplacian_fourier_3d_phys(u, dx)
    grad2_term = -0.5 * eps2 * (u * lap)
    pot_term   = 0.25 * (u**2 - 1.0)**2
    return grad2_term + pot_term

def energy_penalty(u_in, u_pred, dx, eps2):
    # CH3D is also a (mass-conserving) gradient flow of the same free energy.
    if config.PROBLEM not in ('AC3D', 'CH3D'):
        return torch.zeros((), device=u_pred.device, dtype=u_pred.dtype)
    u0 = u_in.squeeze(-1); up = u_pred.squeeze(-1)
    E0 = energy_density(u0, dx, eps2).mean(dim=(1,2,3))
    Ep = energy_density(up, dx, eps2).mean(dim=(1,2,3))
    inc = torch.relu(Ep - E0)
    return inc.mean()


''''
def energy_penalty(u_ref, u_pred, dx, eps2):
    """
    Match free energy of u_pred to u_ref at each time step.

    u_ref, u_pred shapes:
        - (B,S,S,S,T)  multi-step
        - or (B,S,S,S) single-step (then treated as T=1)
    """
    if config.PROBLEM not in ('AC3D', 'CH3D'):
        return torch.zeros((), device=u_pred.device, dtype=u_pred.dtype)

    ur = u_ref
    up = u_pred

    # unify shapes
    if ur.dim() == 5 and ur.shape[-1] == 1:
        ur = ur.squeeze(-1)
    if up.dim() == 5 and up.shape[-1] == 1:
        up = up.squeeze(-1)

    # single-step: just L2 on energy
    if ur.dim() == 4 and up.dim() == 4:
        E_ref = energy_density(ur, dx, eps2).mean(dim=(1, 2, 3))  # (B,)
        E_pred = energy_density(up, dx, eps2).mean(dim=(1, 2, 3)) # (B,)
        return ((E_pred - E_ref)**2).mean()

    # multi-step: (B,S,S,S,T)
    if ur.dim() == 5 and up.dim() == 5:
        assert ur.shape == up.shape
        B, Sx, Sy, Sz, T = ur.shape

        ur_flat = ur.permute(0, 4, 1, 2, 3).reshape(B*T, Sx, Sy, Sz)
        up_flat = up.permute(0, 4, 1, 2, 3).reshape(B*T, Sx, Sy, Sz)

        E_ref_all = energy_density(ur_flat, dx, eps2).mean(dim=(1, 2, 3))  # (B*T,)
        E_pred_all = energy_density(up_flat, dx, eps2).mean(dim=(1, 2, 3)) # (B*T,)

        return ((E_pred_all - E_ref_all)**2).mean()

    # fallback
    E_ref = energy_density(ur, dx, eps2).mean(dim=tuple(range(1, ur.dim())))
    E_pred = energy_density(up, dx, eps2).mean(dim=tuple(range(1, up.dim())))
    return ((E_pred - E_ref)**2).mean()
'''

def energy_penalty_T_out(u_ref, u_pred, dx, eps2):
    """
    Match free energy of u_pred to u_ref.

    Supported shapes:
      - u_ref, u_pred: (B,S,S,S)                 single-step
      - u_ref, u_pred: (B,S,S,S,T)               multi-step
      - u_ref: (B,S,S,S),    u_pred: (B,S,S,S,T) baseline vs multi-step
      - also accepts trailing singleton time dims (B,S,S,S,1).
    """
    if config.PROBLEM not in ('AC3D', 'CH3D'):
        return torch.zeros((), device=u_pred.device, dtype=u_pred.dtype)

    ur = u_ref
    up = u_pred

    # strip useless trailing singleton time dims
    if ur.dim() == 5 and ur.shape[-1] == 1:
        ur = ur.squeeze(-1)
    if up.dim() == 5 and up.shape[-1] == 1:
        up = up.squeeze(-1)

    # ---- case 1: both single-step (B,S,S,S) ----
    if ur.dim() == 4 and up.dim() == 4:
        E_ref  = energy_density(ur, dx, eps2).mean(dim=(1, 2, 3))  # (B,)
        E_pred = energy_density(up, dx, eps2).mean(dim=(1, 2, 3))  # (B,)
        return ((E_pred - E_ref) ** 2).mean()

    # ---- case 2: both multi-step (B,S,S,S,T) ----
    if ur.dim() == 5 and up.dim() == 5:
        assert ur.shape == up.shape
        B, Sx, Sy, Sz, T = ur.shape

        ur_flat = ur.permute(0, 4, 1, 2, 3).reshape(B * T, Sx, Sy, Sz)
        up_flat = up.permute(0, 4, 1, 2, 3).reshape(B * T, Sx, Sy, Sz)

        E_ref_all  = energy_density(ur_flat, dx, eps2).mean(dim=(1, 2, 3))  # (B*T,)
        E_pred_all = energy_density(up_flat, dx, eps2).mean(dim=(1, 2, 3))  # (B*T,)

        return ((E_pred_all - E_ref_all) ** 2).mean()

    # ---- case 3: ref single-step, pred multi-step ----
    if ur.dim() == 4 and up.dim() == 5:
        B, Sx, Sy, Sz, T = up.shape

        # broadcast u_ref over time steps
        ur_flat = ur.unsqueeze(1).expand(B, T, Sx, Sy, Sz).reshape(B * T, Sx, Sy, Sz)
        up_flat = up.permute(0, 4, 1, 2, 3).reshape(B * T, Sx, Sy, Sz)

        E_ref_all  = energy_density(ur_flat, dx, eps2).mean(dim=(1, 2, 3))  # (B*T,)
        E_pred_all = energy_density(up_flat, dx, eps2).mean(dim=(1, 2, 3))  # (B*T,)

        return ((E_pred_all - E_ref_all) ** 2).mean()

    # ---- case 4: ref multi-step, pred single-step (rare) ----
    if ur.dim() == 5 and up.dim() == 4:
        # just swap roles → reuse case 3
        return energy_penalty(up, ur, dx, eps2)

    # ---- fallback: very unusual shapes, just compare global energies ----
    E_ref  = energy_density(ur, dx, eps2).mean(dim=tuple(range(1, ur.dim())))
    E_pred = energy_density(up, dx, eps2).mean(dim=tuple(range(1, up.dim())))
    return ((E_pred - E_ref) ** 2).mean()


def mass_penalty(u_in, u_pred):
    """Penalize change in spatial mean (mass) between steps."""
    u0 = u_in.squeeze(-1)
    up = u_pred.squeeze(-1)
    m0 = u0.mean(dim=(1,2,3))
    mp = up.mean(dim=(1,2,3))
    return ((mp - m0)**2).mean()

def mass_project_pred(y_pred, u_in_last):
    """
    Hard projection for CH mass conservation.

    y_pred:    (B,S,S,S,T_out)
    u_in_last: (B,S,S,S,1)

    Enforces each predicted time slice to have the same spatial mean
    as the last observed input state.
    """
    m_in = u_in_last.mean(dim=(1, 2, 3), keepdim=True)   # (B,1,1,1,1)
    m_out = y_pred.mean(dim=(1, 2, 3), keepdim=True)     # (B,1,1,1,T_out)
    return y_pred - m_out + m_in


# -------------------------
# Minimizing-movement projection (generic: use RHS)
# -------------------------
def mm_projection(u_in, u_pred, dt, dx, eps2, steps=5, eta=None):
    up = u_pred.squeeze(-1).float()
    u0 = u_in.squeeze(-1).float()
    if eta is None:
        eta = 0.5 * dt
    for _ in range(steps):
        g = (up - u0) / dt - pde_rhs(up, dx, config.EPSILON_PARAM).float()
        up = up - eta * g
    return up.unsqueeze(-1), g

def loss_mm_projection(u_in, u_pred):
    dt, dx = config.DT, config.DX
    up_ref, g_last = mm_projection(u_in, u_pred, dt, dx, config.EPS2, steps=5)
    l_proj = F.mse_loss(u_pred, up_ref)
    l_stat = (g_last**2).mean()
    return l_proj + l_stat, l_proj.detach(), l_stat.detach()

# -------------------------
# Scheme residual (teacher consistency)
# Generic: compare to explicit Euler from u^n.
# -------------------------
def scheme_residual_fourier(u_in, u_pred):
    dt, dx = config.DT, config.DX
    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()

    if config.PROBLEM == 'CH3D':
        # ----- EXACT MATLAB SEMI-IMPLICIT RESIDUAL (minimal change) -----
        B, S, _, _ = u0.shape
        k2 = _k_spectrum(S, S, S, dx, u0.device)       # (2π/L)^2 |k|^2  >= 0
        k4 = k2 * k2

        u0_hat    = torch.fft.fftn(u0, dim=(1,2,3))
        up_hat    = torch.fft.fftn(up, dim=(1,2,3))
        chem0_hat = torch.fft.fftn(u0**3 - 3.0*u0, dim=(1,2,3))

        eps = config.EPSILON_PARAM
        denom = 1.0 + dt * (2.0 * k2 + (eps * eps) * k4)            # 1 + dt(2k^2 + eps^2 k^4)

        # R^ = denom * û^{n+1} - [ û^n - dt * k^2 * (u^3 - 3u)^n^ ]
        rhat = denom * up_hat - (u0_hat - dt * k2 * chem0_hat)

        # Ignore k=0 mass mode (nullspace) exactly as in CH3D
        rhat = rhat * (k2 > 0)

        # Precondition by the same denom to avoid high-k domination
        rhat = rhat / (denom + 1e-12)

        # Parseval: ||R||^2 ~ sum |R^|^2. Use mean to keep scale stable.
        r2 = rhat.real**2 + rhat.imag**2
        return r2.mean()
    if config.PROBLEM == 'SH3D':  # <-- NEW
        B, S, _, _ = u0.shape
        k2 = _k_spectrum(S, S, S, dx, u0.device)
        k4 = k2 * k2

        u0_hat = torch.fft.fftn(u0, dim=(1, 2, 3))
        up_hat = torch.fft.fftn(up, dim=(1, 2, 3))
        u3_hat = torch.fft.fftn(u0 ** 3, dim=(1, 2, 3))  # data-consistent (no dealias)

        eps = config.EPSILON_PARAM
        denom = (1.0 / dt) + (1.0 - eps) + k4

        # residual in Fourier of the semi-implicit update
        rhs_hat = (1.0 / dt) * u0_hat - u3_hat + 2.0 * k2 * u0_hat
        rhat = denom * up_hat - rhs_hat

        # precondition to avoid high-k domination (same idea as CH3D)
        rhat = rhat / (denom + 1e-12)

        r2 = rhat.real ** 2 + rhat.imag ** 2
        return r2.mean()

    elif config.PROBLEM == 'PFC3D':  # NEW
        return _scheme_residual_fourier_pfc(u0, up, dt, dx, config.EPSILON_PARAM)

    elif config.PROBLEM == 'MBE3D':
        # Semi-implicit MBE residual (preconditioned in Fourier)
        B, S, _, _ = u0.shape
        k2 = _k_spectrum(S, S, S, dx, u0.device)  # (2π/L)^2 |k|^2
        k4 = k2 * k2

        # denom = 1/dt - k^2 + eps * k^4   (matches MATLAB: 1/dt - (pp2+qq2+rr2) + eps*(...)^2 )
        eps = config.EPSILON_PARAM
        denom = (1.0 / dt) - k2 + eps * k4

        # rhs_hat = FFT( u0/dt + div( |∇u0|^2 ∇u0 ) )
        ux, uy, uz = grad_fourier(u0, dx)  # ∇u0 in real space
        s = ux * ux + uy * uy + uz * uz
        vx, vy, vz = s * ux, s * uy, s * uz
        div_term = div_fourier(vx, vy, vz, dx)  # real space
        rhs_real = (1.0 / dt) * u0 + div_term

        U0p = torch.fft.fftn(up, dim=(1, 2, 3))
        RHS = torch.fft.fftn(rhs_real, dim=(1, 2, 3))

        rhat = denom * U0p - RHS
        # precondition to avoid high-k domination (same style as other PDEs)
        rhat = rhat / (denom + 1e-12)

        r2 = rhat.real ** 2 + rhat.imag ** 2
        return r2.mean()



    if config.PROBLEM == 'AC3D':
        u_explicit = u0 + dt * pde_rhs(u0, dx, config.EPSILON_PARAM).float()
        r = up - u_explicit
        return (r**2).mean()


##
##
# ---------- Fourier grid ----------
def _kgrid(nx, dx, device):
    # like MATLAB: [0:N/2, -N/2+1:-1]
    k = 2.0*torch.pi/(nx*dx) * torch.cat([
        torch.arange(0, nx//2 + 1, device=device),
        torch.arange(-nx//2 + 1, 0,        device=device)
    ])
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k4 = k2**2
    return k2, k4

# ---------- CH3D semi-implicit residual from your MATLAB step ----------
@torch.no_grad()
def _safe_den(den):
    return den + 1e-12

def ch3d_semiimplicit_residual(u_n, u_np1, dt, dx, eps, huber_beta=1.0):
    """
    u_n, u_np1: (B,S,S,S) in real space
    returns scalar physics loss for CH3D (semi-implicit residual, robust & preconditioned)
    """
    _, S, _, _ = u_n.shape
    device = u_n.device
    k2, k4 = _kgrid(S, dx, device)
    eps2 = eps**2

    u0_hat   = torch.fft.fftn(u_n,   dim=(1,2,3))
    up_hat   = torch.fft.fftn(u_np1, dim=(1,2,3))
    chem0    = u_n**3 - 3.0*u_n                 # f'(u) = u^3 - 3u  (your code)
    chem0_hat= torch.fft.fftn(chem0, dim=(1,2,3))

    denom = 1.0 + dt*(2.0*k2 + eps2*k4)         # semi-implicit operator
    lhs   = denom * up_hat
    rhs   = u0_hat - dt * k2 * chem0_hat
    rhat  = lhs - rhs

    # CH3D specifics:
    mask = (k2 > 0).to(rhat.real.dtype)         # ignore k=0 mass mode (nullspace)
    rhat = rhat * mask

    # precondition: whiten spectrum so high-k doesn’t dominate
    rhat = rhat / _safe_den(denom)

    # robust penalty in Fourier (Parseval)
    r2 = rhat.real**2 + rhat.imag**2
    loss_res = F.smooth_l1_loss(r2, torch.zeros_like(r2), beta=huber_beta)
    return loss_res

# ---------- tiny soft anchor for mass (k=0 mode) ----------
def ch_mass_anchor(u_n, u_np1):
    m0 = u_n.mean(dim=(1,2,3))
    m1 = u_np1.mean(dim=(1,2,3))
    return ((m1 - m0)**2).mean()

# ---------- one-sided (hinge) free-energy decrease ----------
def _periodic_grad2(u, dx):
    # periodic forward differences; avoids FFT normalization pitfalls
    gx = (torch.roll(u, shifts=-1, dims=1) - u) / dx
    gy = (torch.roll(u, shifts=-1, dims=2) - u) / dx
    gz = (torch.roll(u, shifts=-1, dims=3) - u) / dx
    return gx*gx + gy*gy + gz*gz

def ch_free_energy_density(u, dx, eps):
    bulk = 0.25*(u**2 - 1.0)**2
    grad2 = _periodic_grad2(u, dx)
    return (bulk + 0.5*(eps**2)*grad2).mean()

def ch_energy_hinge(u_n, u_np1, dx, eps):
    Fn  = ch_free_energy_density(u_n,   dx, eps)
    Fnp = ch_free_energy_density(u_np1, dx, eps)
    return torch.relu(Fnp - Fn)    # penalize only increases


##

def _mid_residual_norm_sh(u_in, u_pred):
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM
    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()
    um = 0.5 * (u0 + up)
    ut = (up - u0) / dt
    lap = laplacian_fourier_3d_phys(um, dx)
    bi = biharmonic(um, dx)
    rhs = (1.0 - eps) * um - 2.0 * lap - bi - (um ** 3)
    s_t = ut.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
    s_r = rhs.pow(2).mean((1, 2, 3), keepdim=True).sqrt().detach() + 1e-8
    R = ut / s_t - rhs / s_r
    return (R ** 2).mean()


##
##############



def _hm1_mean(R, dx, eps_floor_scale=1e-2):
    with torch.amp.autocast(device_type='cuda', enabled=False):
        B, S, _, _ = R.shape
        k2, inv_k2 = _fft_rk2(S, S, S, dx, R.device, dtype=torch.float32)
        R_hat = torch.fft.fftn(R.float(), dim=(1,2,3))
        pow_spec = R_hat.real**2 + R_hat.imag**2
        return (pow_spec * inv_k2).mean().to(R.dtype)

# === NEW: one extra Gauss–Lobatto collocation residual in H^{-1} (CH3D) ===
import math


def physics_collocation_tau_L2_SH(u_in, u_pred,
                                  tau=0.5 - 1.0/(2.0*math.sqrt(5.0)),
                                  normalize=True):
    """
    SH3D collocation residual at interior time u_tau:
      R_tau = (u^{n+1}-u^n)/dt - RHS_SH(u_tau)
    with
      RHS_SH(u) = (1-ε) u - 2 Δ u - Δ^2 u - u^3
    Notes:
      • NO dealiasing on u^3 (matches your MATLAB generator).
      • Optional per-sample normalization balances ut vs RHS in L2.
      • Scored in L2 (SH is an L2-type gradient flow).
    """
    assert config.PROBLEM == 'SH3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM

    u0 = u_in.squeeze(-1).float()   # (B,S,S,S)
    up = u_pred.squeeze(-1).float()

    ut = (up - u0) / dt
    u_tau = (1.0 - tau) * u0 + tau * up

    # Dataset-consistent SH RHS (no dealias on cubic)
    lap_u   = laplacian_fourier_3d_phys(u_tau, dx)
    bi_u    = biharmonic(u_tau, dx)
    rhs_tau = (1.0 - eps) * u_tau - 2.0 * lap_u - bi_u - (u_tau ** 3)

    if normalize:
        s_t = ut.pow(2).mean(dim=(1,2,3), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean(dim=(1,2,3), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau

    return (R ** 2).mean().to(u_pred.dtype)

def sh_free_energy_density(u, dx, eps):
    """
    Per-voxel energy density for Swift–Hohenberg:
      f_SH(u) = - (1-ε)/2 * u^2  - |∇u|^2  + 1/2 * (Δu)^2  + 1/4 * u^4
    Uses periodic finite differences for |∇u|^2 and spectral Δ for Δu.
    Shapes: u (B,S,S,S), returns (B,S,S,S).
    """
    # |∇u|^2 with periodic forward diffs (same style as CH3D helper)
    grad2 = _periodic_grad2(u, dx)  # already in your file

    # Δu via your spectral Laplacian
    lap = laplacian_fourier_3d_phys(u, dx)

    bulk_quad  = -0.5 * (1.0 - eps) * (u * u)
    grad_term  = -grad2
    bih_term   = 0.5 * (lap * lap)
    quartic    = 0.25 * (u * u * u * u)

    return (bulk_quad + grad_term + bih_term + quartic)

def energy_penalty_sh(u_in, u_pred, dx, eps):
    """
    One-sided hinge on F_SH increase: penalize only if F(u^{n+1}) > F(u^n).
    Returns a scalar (mean over batch).
    """
    u0 = u_in.squeeze(-1)
    up = u_pred.squeeze(-1)
    F0 = sh_free_energy_density(u0, dx, eps).mean(dim=(1,2,3))
    Fp = sh_free_energy_density(up, dx, eps).mean(dim=(1,2,3))
    inc = torch.relu(Fp - F0)  # only increases are penalized
    return inc.mean()



# === Optimal Physics-Guided Update for CH3D (minimizes SI residual in H^{-1}) ===
import torch, math
import torch.nn.functional as F
import numpy as np
import config

import torch, math
import torch.nn.functional as F


#############
def physics_collocation_tau_L2_PFC(u_in, u_pred, tau=0.5 - 1.0/(2.0*math.sqrt(5.0)),
                                   normalize=True):
    """
    PFC3D collocation at u_tau: R_tau = (u^{n+1}-u^n)/dt - RHS_PFC(u_tau).
    Scored in L2; optional per-sample normalization.
    """
    assert config.PROBLEM == 'PFC3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM
    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()
    ut   = (up - u0) / dt
    u_tau = (1.0 - tau) * u0 + tau * up
    # RHS that matches data (no dealias):
    rhs_tau = _rhs_pfc3d(u_tau, dx, eps)

    if normalize:
        s_t = ut.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau
    return (R**2).mean().to(u_pred.dtype)




def pfc_free_energy_density(u, dx, eps):
    """
    F[u] = ∫ [ 1/2(1-ε)u^2  - |∇u|^2  + 1/2(Δu)^2  - 1/4 u^4 ] dx
    δF/δu = (1-ε)u + 2Δu + Δ^2 u - u^3  ⇒  u_t = Δ(δF/δu) matches your PFC step.
    """
    grad2 = _periodic_grad2(u, dx)                      # |∇u|^2
    lap   = laplacian_fourier_3d_phys(u, dx)            # Δu

    bulk_quad = 0.5 * (1.0 - eps) * (u * u)
    grad_term = -grad2
    bih_term  = 0.5 * (lap * lap)
    quartic   = -0.25 * (u ** 4)

    return bulk_quad + grad_term + bih_term + quartic

def energy_penalty_pfc(u_in, u_pred, dx, eps):
    """One-sided hinge: penalize F(u^{n+1}) > F(u^n)."""
    u0 = u_in.squeeze(-1)
    up = u_pred.squeeze(-1)
    F0 = pfc_free_energy_density(u0, dx, eps).mean(dim=(1,2,3))
    Fp = pfc_free_energy_density(up, dx, eps).mean(dim=(1,2,3))
    return torch.relu(Fp - F0).mean()


###########
## MBE3D

# ==== MBE3D helpers (append to functions.py) =================================
import math as _math

def _k_axes_phys(S, dx, device, dtype):
    # physical k like MATLAB: 2π/L * [0..S/2, -S/2+1..-1]
    L = S * dx
    half = S // 2
    kv = torch.cat([torch.arange(0, half + 1, device=device),
                    torch.arange(-half + 1, 0, device=device)], dim=0).to(dtype)
    return (2.0 * _math.pi / L) * kv  # (S,)



# --- at top of functions.py (near imports) ---
_MBES2_MAX = 36.0  # gentle cap for |∇u|^2 (tunable: 16..64 works well)

def _cap_s2(s2, cap=_MBES2_MAX):
    # inplace-free, differentiable cap
    return torch.clamp(s2, max=cap)

def semi_implicit_step_mbe(u_in, dt, dx, eps):
    """
    Robust MBE3D semi-implicit, computed in float64 to avoid overflow/underflow.
    """
    u = u_in.squeeze(-1).to(torch.float64)      # promote to float64
    B, Sx, Sy, Sz = u.shape
    device = u.device

    # physical k like MATLAB
    def _k_axes_phys64(S, dx):
        L = S * dx
        half = S // 2
        kv = torch.cat([torch.arange(0, half + 1, device=device),
                        torch.arange(-half + 1, 0, device=device)], dim=0).to(torch.float64)
        return (2.0 * np.pi / L) * kv

    kx = _k_axes_phys64(Sx, dx); ky = _k_axes_phys64(Sy, dx); kz = _k_axes_phys64(Sz, dx)
    PX, QY, RZ = torch.meshgrid(1j*kx, 1j*ky, 1j*kz, indexing='ij')
    K2 = (PX/(1j))**2 + (QY/(1j))**2 + (RZ/(1j))**2  # real, >=0 (float64)

    U = torch.fft.fftn(u, dim=(1,2,3))

    fx = torch.fft.ifftn(PX * U, dim=(1,2,3)).real
    fy = torch.fft.ifftn(QY * U, dim=(1,2,3)).real
    fz = torch.fft.ifftn(RZ * U, dim=(1,2,3)).real

    s2 = fx*fx + fy*fy + fz*fz
    s2 = _cap_s2(s2)  # <--- important OOD stabilizer

    f1 = s2 * fx; f2 = s2 * fy; f3 = s2 * fz
    div_hat = PX * torch.fft.fftn(f1, dim=(1,2,3)) \
            + QY * torch.fft.fftn(f2, dim=(1,2,3)) \
            + RZ * torch.fft.fftn(f3, dim=(1,2,3))

    s_hat = torch.fft.fftn(u / dt, dim=(1,2,3)) + div_hat
    denom = (1.0 / dt) - K2 + float(eps) * (K2 ** 2)

    v_hat = s_hat / (denom + 1e-16)  # slightly larger epsilon in 64-bit
    up = torch.fft.ifftn(v_hat, dim=(1,2,3)).real
    return up.to(torch.float32).unsqueeze(-1)   # return in float32

def _scheme_residual_fourier_mbe(u0, up, dt, dx, eps):
    """
    Semi-implicit residual in Fourier matching semi_implicit_step_mbe:
      R̂ = [1/dt - k^2 + eps k^4] * Û^{n+1} - { FFT(u^n/dt) + i k · FFT((|∇u^n|^2 ∇u^n)) }
    Precondition by denom to whiten spectrum, then L2 mean.
    """
    u0 = u0.float(); up = up.float()
    B, S, _, _ = u0.shape
    device, dtype = u0.device, u0.dtype

    kv = _k_axes_phys(S, dx, device, dtype)
    PX, QY, RZ = torch.meshgrid(1j*kv, 1j*kv, 1j*kv, indexing='ij')
    K2 = (PX/(1j))**2 + (QY/(1j))**2 + (RZ/(1j))**2

    U0 = torch.fft.fftn(u0, dim=(1,2,3))
    UP = torch.fft.fftn(up, dim=(1,2,3))

    fx = torch.fft.ifftn(PX * U0, dim=(1,2,3)).real
    fy = torch.fft.ifftn(QY * U0, dim=(1,2,3)).real
    fz = torch.fft.ifftn(RZ * U0, dim=(1,2,3)).real
    s2 = fx*fx + fy*fy + fz*fz
    s2 = _cap_s2(s2)
    f1, f2, f3 = s2*fx, s2*fy, s2*fz
    div_hat = PX * torch.fft.fftn(f1, dim=(1,2,3)) \
            + QY * torch.fft.fftn(f2, dim=(1,2,3)) \
            + RZ * torch.fft.fftn(f3, dim=(1,2,3))

    denom = (1.0/dt) - K2 + float(eps) * (K2**2)
    rhs   = (1.0/dt) * U0 + div_hat
    rhat  = denom * UP - rhs
    rhat  = rhat / (denom + 1e-12)

    r2 = rhat.real**2 + rhat.imag**2
    return r2.mean()

def physics_collocation_tau_L2_MBE(u_in, u_pred,
                                   tau=0.5 - 1.0/(2.0*_math.sqrt(5.0)),
                                   normalize=True):
    """
    MBE collocation (L2) at u_tau:
      R_tau = (u^{n+1}-u^n)/dt - [ -Δu_tau - eps ∇^4 u_tau + ∇·(|∇u_tau|^2 ∇u_tau) ].
    """
    assert config.PROBLEM == 'MBE3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM
    u0 = u_in.squeeze(-1).float()
    up = u_pred.squeeze(-1).float()

    ut    = (up - u0) / dt
    u_tau = (1.0 - tau) * u0 + tau * up

    ux, uy, uz = grad_fourier(u_tau, dx)
    s2 = _cap_s2(ux * ux + uy * uy + uz * uz)  # <--- cap here too
    #s2 = ux*ux + uy*uy + uz*uz
    # (optional) gentle clamp to avoid rare spiky batches; remove if undesired:
    # s2 = torch.clamp(s2, max=36.0)

    vx, vy, vz = s2*ux, s2*uy, s2*uz
    div_term   = div_fourier(vx, vy, vz, dx)
    lap_u      = laplacian_fourier_3d_phys(u_tau, dx)
    bi_u       = biharmonic(u_tau, dx)

    rhs_tau = -lap_u - eps * bi_u + div_term

    if normalize:
        s_t = ut.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau
    return (R**2).mean().to(u_pred.dtype)




def mbe_free_energy_density(u, dx, eps):
    """
    Slope-selection MBE energy density:
        f = (eps/2) |Δu|^2 + 1/4 (|∇u|^2 - 1)^2    (constant shift ignored).
    """
    grad2 = _periodic_grad2(u, dx)       # |∇u|^2
    lap   = laplacian_fourier_3d_phys(u, dx)
    return 0.5 * eps * (lap * lap) + 0.25 * (grad2 - 1.0) * (grad2 - 1.0)


def energy_penalty_mbe(u_in, u_pred, dx, eps):
    """One-sided hinge: penalize F(u^{n+1}) > F(u^n)."""
    u0 = u_in.squeeze(-1)
    up = u_pred.squeeze(-1)
    F0 = mbe_free_energy_density(u0, dx, eps).mean(dim=(1,2,3))
    Fp = mbe_free_energy_density(up, dx, eps).mean(dim=(1,2,3))
    return torch.relu(Fp - F0).mean()
# ==================

def semi_implicit_step_ch(u_in, dt, dx, eps):
    """
    CORRECTED to exactly match MATLAB:
        (1 + dt*(2k² + ε²k⁴)) û^{n+1} = û^n - dt*k² * FFT(u^n³ - 3u^n)
    """
    u0 = u_in.squeeze(-1).float()
    B, S, _, _ = u0.shape
    k2 = _k_spectrum(S, S, S, dx, u0.device)  # (2π/L)²|k|²

    # Exact MATLAB match
    chem0_hat = torch.fft.fftn(u0 ** 3 - 3.0 * u0, dim=(1, 2, 3))
    U0 = torch.fft.fftn(u0, dim=(1, 2, 3))

    denom = 1.0 + dt * (2.0 * k2 + (eps ** 2) * (k2 ** 2))
    numer = U0 - dt * k2 * chem0_hat  # Note: k2 here is (kxx+kyy+kzz) equivalent

    U1 = numer / (denom + 1e-12)
    u1 = torch.fft.ifftn(U1, dim=(1, 2, 3)).real
    return u1.unsqueeze(-1).to(u_in.dtype)

def semi_implicit_step_ch_T_out(u_in, dt, dx, eps):
    """
    CORRECTED to exactly match MATLAB:
        (1 + dt*(2k² + ε²k⁴)) û^{n+1} = û^n - dt*k² * FFT(u^n³ - 3u^n)

    Multi-step extension:
      - If T_out == 1: original one-step behavior.
      - If T_out > 1: returns (B,S,S,S,T_out) with T_out semi-implicit steps.
    """
    u0 = u_in.squeeze(-1).float()   # (B,S,S,S)
    B, S, _, _ = u0.shape
    k2 = _k_spectrum(S, S, S, dx, u0.device)  # (2π/L)²|k|²

    # read T_out from config (support either T_OUT or T_out name)
    T_out = int(getattr(config, "T_OUT", getattr(config, "T_out", 1)))

    # ---------- single-step: original behavior ----------
    if T_out <= 1:
        chem0_hat = torch.fft.fftn(u0 ** 3 - 3.0 * u0, dim=(1, 2, 3))
        U0 = torch.fft.fftn(u0, dim=(1, 2, 3))

        denom = 1.0 + dt * (2.0 * k2 + (eps ** 2) * (k2 ** 2))
        numer = U0 - dt * k2 * chem0_hat

        U1 = numer / (denom + 1e-12)
        u1 = torch.fft.ifftn(U1, dim=(1, 2, 3)).real
        return u1.unsqueeze(-1).to(u_in.dtype)  # (B,S,S,S,1)

    # ---------- multi-step: iterate T_out times ----------
    steps = torch.empty(B, S, S, S, T_out, device=u0.device, dtype=u0.dtype)
    u_curr = u0

    # denom does not depend on u, only on k2
    denom = 1.0 + dt * (2.0 * k2 + (eps ** 2) * (k2 ** 2))

    for t in range(T_out):
        chem_hat = torch.fft.fftn(u_curr ** 3 - 3.0 * u_curr, dim=(1, 2, 3))
        U0 = torch.fft.fftn(u_curr, dim=(1, 2, 3))

        numer = U0 - dt * k2 * chem_hat
        U1 = numer / (denom + 1e-12)
        u_next = torch.fft.ifftn(U1, dim=(1, 2, 3)).real

        steps[..., t] = u_next
        u_curr = u_next

    # (B,S,S,S,T_out), no extra singleton channel, matches TNO output
    return steps.to(u_in.dtype)

def _rhs_ch3d(u, dx, eps):
    """
    CORRECTED CH3D RHS to match MATLAB generator:
        u_t = -Δ[2u + ε²Δu + (u³ - 3u)]
    This matches the MATLAB semi-implicit splitting.
    """
    lap_u = laplacian_fourier_3d_phys(u, dx)  # Δu
    bi_u = biharmonic(u, dx)  # Δ²u
    chem = u ** 3 - 3.0 * u  # f'(u) = u³ - 3u

    # The RHS is -Δ of everything
    return -laplacian_fourier_3d_phys(2.0 * u + (eps ** 2) * lap_u + chem, dx)

import math as _m

def physics_collocation_tau_L2_CH(u_in, u_pred,
                                  tau=0.5 - 1.0/(2.0*_m.sqrt(5.0)),
                                  normalize=True):
    """
    CH3D L2 collocation at u_tau (match SH/PFC/MBE structure):
      R_tau = (u^{n+1}-u^n)/dt - RHS_CH(u_tau)
    scored in **L2** with optional per-sample normalization.
    """
    assert config.PROBLEM == 'CH3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM

    u0 = u_in.squeeze(-1).float()   # (B,S,S,S)
    up = u_pred.squeeze(-1).float()
    ut = (up - u0) / dt
    u_tau = (1.0 - tau) * u0 + tau * up

    rhs_tau = _rhs_ch3d(u_tau, dx, eps)

    if normalize:
        s_t = ut.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau

    return (R**2).mean().to(u_pred.dtype)


def physics_collocation_tau_L2_CH_T_out(u_in, u_pred,
                                  tau=0.5 - 1.0/(2.0*_m.sqrt(5.0)),
                                  normalize=True):
    """
    CH3D L2 collocation at u_tau (now supports multi-step T_out):
      R_tau = (u^{n+1}-u^n)/dt - RHS_CH(u_tau)
    If u_pred has shape (B,S,S,S,T_out), we apply the same formula
    for each predicted step, broadcasting u^n.
    """
    assert config.PROBLEM == 'CH3D'
    dt, dx, eps = config.DT, config.DX, config.EPSILON_PARAM

    u0 = u_in.squeeze(-1).float()   # (B,S,S,S)
    up = u_pred.squeeze(-1).float() # (B,S,S,S) or (B,S,S,S,T)

    # ----- single-step (backward compatible) -----
    if up.dim() == 4:
        ut   = (up - u0) / dt              # (B,S,S,S)
        u_tau = (1.0 - tau) * u0 + tau * up
        rhs_tau = _rhs_ch3d(u_tau, dx, eps)  # (B,S,S,S)

        if normalize:
            s_t = ut.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
            s_r = rhs_tau.pow(2).mean((1,2,3), keepdim=True).sqrt().detach() + 1e-8
            R = ut / s_t - rhs_tau / s_r
        else:
            R = ut - rhs_tau
        return (R**2).mean().to(u_pred.dtype)

    # ----- multi-step: up: (B,S,S,S,T_out) -----
    assert up.dim() == 5, "u_pred must be 4D or 5D"

    B, Sx, Sy, Sz, T = up.shape
    u0_exp = u0.unsqueeze(-1)                  # (B,S,S,S,1)

    ut    = (up - u0_exp) / dt                 # (B,S,S,S,T)
    u_tau = (1.0 - tau) * u0_exp + tau * up    # (B,S,S,S,T)

    # RHS_CH expects (B,S,S,S), so flatten time into batch:
    u_tau_flat = u_tau.permute(0,4,1,2,3).reshape(B*T, Sx, Sy, Sz)
    rhs_flat   = _rhs_ch3d(u_tau_flat, dx, eps)           # (B*T,S,S,S)
    rhs_tau    = rhs_flat.view(B, T, Sx, Sy, Sz).permute(0,2,3,4,1)  # (B,S,S,S,T)

    if normalize:
        # normalize over all spatial + time dims
        s_t = ut.pow(2).mean((1,2,3,4), keepdim=True).sqrt().detach() + 1e-8
        s_r = rhs_tau.pow(2).mean((1,2,3,4), keepdim=True).sqrt().detach() + 1e-8
        R = ut / s_t - rhs_tau / s_r
    else:
        R = ut - rhs_tau

    return (R**2).mean().to(u_pred.dtype)
