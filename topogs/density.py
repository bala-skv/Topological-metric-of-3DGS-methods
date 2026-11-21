"""Sample a 3DGS scene as a scalar density field on a regular grid.

    d(x) = sum_k  o_k * exp(-1/2 * m_k(x)^2),   m_k(x)^2 = || diag(1/s_k) R_k^T (x - mu_k) ||^2

which equals the Gaussian Mahalanobis distance with Sigma_k = R_k S_k S_k^T R_k^T,
computed without inverting any matrix.

Each Gaussian is truncated at Mahalanobis radius `cutoff` (default 4 sigma,
i.e. at exp(-8) ~ 3e-4 of its peak) so only nearby voxels are touched.

The grid lives in a (possibly rotated) crop box:
    box frame coords  u = R_box @ (x - center)
    voxel centres     u_i = -half + (i + 0.5 + offset) * h
`offset` (in voxels) is used for the half-voxel-shift noise floor.

Anti-aliasing (filter_sigma > 0, world units): each covariance is widened by an
isotropic Gaussian, Sigma' = R (S^2 + f^2 I) R^T, i.e. every scale becomes
s' = sqrt(s^2 + f^2) with the rotation unchanged. 3DGS surfaces are made of flat
"pancake" / needle Gaussians far thinner than any affordable voxel; without this,
whether a primitive shows up depends on where the voxel centres happen to fall.
  mode="peak": opacity kept (each primitive becomes at least ~f thick; analogous to
               the vanilla 3DGS rasterizer's +0.3 px screen-space dilation).
  mode="mass": opacity *= prod(s/s'), an exact convolution of the field with a
               normalised Gaussian (Mip-Splatting-style 3D filter).
filter_sigma must be fixed in world units across resolutions and variants.
Array index order is [ix, iy, iz] along the box's u, v, w axes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

from .io_ply import GaussianScene


@dataclass
class Grid:
    center: np.ndarray            # (3,) world coordinates of the box centre
    half: np.ndarray              # (3,) half-extents in the box frame
    shape: tuple                  # (nx, ny, nz)
    R_box: np.ndarray = field(default_factory=lambda: np.eye(3))  # rows = box axes in world coords
    offset: float = 0.0           # shift of all sample points, in voxel units

    def __post_init__(self):
        self.center = np.asarray(self.center, float)
        self.half = np.broadcast_to(np.asarray(self.half, float), (3,)).copy()
        self.shape = tuple(int(s) for s in np.broadcast_to(self.shape, (3,)))
        self.R_box = np.asarray(self.R_box, float)

    @property
    def spacing(self) -> np.ndarray:
        return 2 * self.half / np.array(self.shape)

    @property
    def origin(self) -> np.ndarray:
        """Box-frame coordinate of voxel (0,0,0)."""
        return -self.half + (0.5 + self.offset) * self.spacing

    def axes(self):
        return [self.origin[a] + self.spacing[a] * np.arange(self.shape[a]) for a in range(3)]

    def points_box(self) -> np.ndarray:
        U, V, W = np.meshgrid(*self.axes(), indexing="ij")
        return np.stack([U, V, W], axis=-1)

    def to_dict(self) -> dict:
        return dict(center=self.center.tolist(), half=self.half.tolist(), shape=list(self.shape),
                    R_box=self.R_box.tolist(), offset=self.offset)

    @classmethod
    def cubic(cls, center, half, res, R_box=None, offset=0.0):
        """Isotropic voxels: `res` voxels along the longest box axis."""
        half = np.broadcast_to(np.asarray(half, float), (3,))
        h = 2 * half.max() / res
        shape = tuple(int(np.ceil(2 * x / h - 1e-9)) for x in half)
        half_adj = np.array(shape) * h / 2  # make spacing exactly isotropic
        return cls(center, half_adj, shape, np.eye(3) if R_box is None else R_box, offset)


def to_box_frame(scene: GaussianScene, grid: Grid):
    """Means and rotation matrices expressed in the box frame."""
    mu = (scene.means - grid.center) @ grid.R_box.T
    R = np.einsum("ij,njk->nik", grid.R_box, scene.rotmats)
    return mu, R


def _aabb_half(R, s, cutoff):
    # Sigma' = R S^2 R^T ; axis-aligned half-extent of the cutoff ellipsoid = cutoff*sqrt(diag)
    return cutoff * np.sqrt(np.einsum("nij,nj->ni", R ** 2, s ** 2))


@njit(parallel=True, fastmath=False, cache=True)
def _splat(grid, origin, h, mu, R, inv_s, op, lo_idx, hi_idx, r2, nslab):
    nx, ny, nz = grid.shape
    G = mu.shape[0]
    bounds = np.linspace(0, nx, nslab + 1).astype(np.int64)
    for s in prange(nslab):
        x0, x1 = bounds[s], bounds[s + 1]
        for k in range(G):
            i0 = max(lo_idx[k, 0], x0)
            i1 = min(hi_idx[k, 0], x1 - 1)
            if i0 > i1:
                continue
            for i in range(i0, i1 + 1):
                dx = origin[0] + h[0] * i - mu[k, 0]
                for j in range(lo_idx[k, 1], hi_idx[k, 1] + 1):
                    dy = origin[1] + h[1] * j - mu[k, 1]
                    for l in range(lo_idx[k, 2], hi_idx[k, 2] + 1):
                        dz = origin[2] + h[2] * l - mu[k, 2]
                        m2 = 0.0
                        for a in range(3):  # (R^T d)_a / s_a
                            y = (R[k, 0, a] * dx + R[k, 1, a] * dy + R[k, 2, a] * dz) * inv_s[k, a]
                            m2 += y * y
                        if m2 <= r2:
                            grid[i, j, l] += op[k] * np.exp(-0.5 * m2)
    return grid


def filtered_params(scene: GaussianScene, filter_sigma: float = 0.0, mode: str = "peak"):
    s, op = scene.scales, scene.opacities
    if filter_sigma > 0:
        s_new = np.sqrt(s ** 2 + filter_sigma ** 2)
        if mode == "mass":
            op = op * np.prod(s / s_new, axis=1)
        elif mode != "peak":
            raise ValueError(mode)
        s = s_new
    return s, op


def density_field(scene: GaussianScene, grid: Grid, cutoff: float = 4.0, nslab: int = 64,
                  filter_sigma: float = 0.0, mode: str = "peak"):
    """Fast truncated density on `grid`. Returns (field[nx,ny,nz] float64, n_gaussians_used)."""
    mu, R = to_box_frame(scene, grid)
    s, op = filtered_params(scene, filter_sigma, mode)
    ext = _aabb_half(R, s, cutoff)
    origin, h, shape = grid.origin, grid.spacing, np.array(grid.shape)
    lo = np.ceil((mu - ext - origin) / h).astype(np.int64)
    hi = np.floor((mu + ext - origin) / h).astype(np.int64)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, shape - 1)
    keep = np.all(lo <= hi, axis=1)
    out = np.zeros(grid.shape, dtype=np.float64)
    if keep.any():
        _splat(out, origin, h, np.ascontiguousarray(mu[keep]), np.ascontiguousarray(R[keep]),
               np.ascontiguousarray(1.0 / s[keep]), np.ascontiguousarray(op[keep]),
               np.ascontiguousarray(lo[keep]), np.ascontiguousarray(hi[keep]),
               float(cutoff) ** 2, int(min(nslab, grid.shape[0])))
    return out, int(keep.sum())


def density_bruteforce(scene: GaussianScene, grid: Grid, cutoff: float | None = 4.0,
                       filter_sigma: float = 0.0, mode: str = "peak"):
    """Every Gaussian at every voxel (tiny grids only). cutoff=None means no truncation."""
    mu, R = to_box_frame(scene, grid)
    s, op = filtered_params(scene, filter_sigma, mode)
    P = grid.points_box().reshape(-1, 3)
    out = np.zeros(len(P))
    for k in range(scene.n):
        y = ((P - mu[k]) @ R[k]) / s[k]           # rows: (R^T (x-mu))^T / s
        m2 = np.sum(y * y, axis=1)
        val = op[k] * np.exp(-0.5 * m2)
        if cutoff is not None:
            val[m2 > cutoff ** 2] = 0.0
        out += val
    return out.reshape(grid.shape)
