import torch
import torch.nn.functional as F
import numpy as np
import config
import math
######
#####


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



def idct3_neumann(uhat):
    out = _idct_axis(uhat, 1)
    out = _idct_axis(out, 2)
    out = _idct_axis(out, 3)
    return out
def laplacian_neumann_cosine_3d(u, dx):
    _, nx, ny, nz = u.shape
    k2 = _neumann_k2_grid(nx, ny, nz, dx, u.device, u.dtype)
    u_hat = dct3_neumann(u)
    lap_hat = -k2 * u_hat
    return idct3_neumann(lap_hat)

def pde_rhs_ac_neumann(u, dx, eps2):
    lap_u = laplacian_neumann_cosine_3d(u, dx)
    return lap_u - (1.0 / eps2) * (u**3 - u)
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
def tetra_gauss_l2_scalar(field, dx):
    """
    Integrate field^2 over the domain using tetrahedral centroid Gauss quadrature.

    field: (B,Sx,Sy,Sz)

    returns: (B,)
    """
    centroids = _tet6_centroid_values(field)          # (B,Sx-1,Sy-1,Sz-1,6)
    tet_volume = (dx ** 3) / 6.0
    return tet_volume * (centroids ** 2).sum(dim=(1, 2, 3, 4))
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

# It is a PDE residual (least-squares) integral
# Global term
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

###########
# Local term


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


## loss_weak_fe  ====>  Weak form

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



######
#####
# CH case from here:
### integrated with a single centroid point tetrahedral quadrature.





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



######
#####
# CH case from here:
### integrated with 4-point tetrahedral quadrature instead of a single centroid point.

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

''''
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
'''



###################
#################
# SH functions


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



####### PENCO --> SH functions


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


#


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

#


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

#

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
