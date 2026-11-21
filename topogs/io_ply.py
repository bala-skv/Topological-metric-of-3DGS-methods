"""Load / save 3D Gaussian Splatting scenes in the INRIA .ply convention.

Stored (raw) vs. activated values:
  opacity   stored as a logit         -> sigmoid
  scale_i   stored as log(scale)      -> exp
  rot_0..3  quaternion (w, x, y, z), not necessarily unit -> normalize
The raw structured array is kept untouched so that variants can be written
back with every property (including colour / SH) preserved in order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from plyfile import PlyData, PlyElement


@dataclass
class GaussianScene:
    raw: np.ndarray  # structured array, one record per Gaussian, all original properties

    @property
    def n(self) -> int:
        return len(self.raw)

    @property
    def means(self) -> np.ndarray:
        return np.stack([self.raw["x"], self.raw["y"], self.raw["z"]], axis=1).astype(np.float64)

    @property
    def opacities(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.raw["opacity"].astype(np.float64)))

    @property
    def scales(self) -> np.ndarray:
        return np.exp(np.stack([self.raw[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64))

    @property
    def quats(self) -> np.ndarray:
        q = np.stack([self.raw[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
        return q / np.linalg.norm(q, axis=1, keepdims=True)

    @property
    def rotmats(self) -> np.ndarray:
        return quat_to_rotmat(self.quats)

    def subset(self, mask_or_idx) -> "GaussianScene":
        return GaussianScene(self.raw[mask_or_idx].copy())


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(N,4) unit quaternions in (w,x,y,z) order -> (N,3,3) rotation matrices.
    Same formula as build_rotation() in the INRIA gaussian-splatting code."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


REQUIRED = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def load_ply(path) -> GaussianScene:
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    missing = [p for p in REQUIRED if p not in v.dtype.names]
    if missing:
        raise ValueError(f"{path}: missing 3DGS properties {missing}")
    return GaussianScene(np.array(v))


def save_ply(scene: GaussianScene, path) -> None:
    el = PlyElement.describe(scene.raw, "vertex")
    PlyData([el], text=False).write(str(path))


def make_scene(means, scales, quats, opacities, sh_rest: int = 45) -> GaussianScene:
    """Build an INRIA-format scene from *activated* values (for synthetic tests).
    Writes logit(opacity), log(scale), and dummy colour fields."""
    means = np.asarray(means, float).reshape(-1, 3)
    n = len(means)
    scales = np.broadcast_to(np.asarray(scales, float), (n, 3))
    quats = np.broadcast_to(np.asarray(quats, float), (n, 4))
    opac = np.clip(np.broadcast_to(np.asarray(opacities, float), (n,)), 1e-7, 1 - 1e-7)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(sh_rest)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    raw = np.zeros(n, dtype=[(nm, "f4") for nm in names])
    raw["x"], raw["y"], raw["z"] = means.T
    raw["opacity"] = np.log(opac / (1 - opac))
    for i in range(3):
        raw[f"scale_{i}"] = np.log(scales[:, i])
    for i in range(4):
        raw[f"rot_{i}"] = quats[:, i]
    return GaussianScene(raw)
