"""Cubical persistent homology of a density field, and diagram distances.

We care about *superlevel* sets {d >= t} (high-density regions = matter).
GUDHI filters by *sublevel* sets, so we feed it -d. A feature that appears when
the threshold drops to density b and disappears at density a < b is stored by
GUDHI as the pair (birth=-b, death=-a); persistence = b - a in density units.

Each voxel value is assigned to a top-dimensional cube; lower-dimensional faces
take the min of their cofaces (in -d), so two voxels touching only at a corner
are connected as soon as both are "in" (26-connectivity of the superlevel set).

The single infinite H0 bar (the whole crop eventually becomes one component) is
dropped from the diagrams and its birth is returned separately.
"""
from __future__ import annotations

import time

import gudhi
import gudhi.hera
import numpy as np


def persistence(field: np.ndarray, min_persistence: float = 0.0):
    """Returns dict with 'dgms' {0,1,2: (n,2) arrays in -density units, finite only},
    'essential' {dim: births of infinite bars}, and 'seconds'."""
    t0 = time.perf_counter()
    cc = gudhi.CubicalComplex(top_dimensional_cells=-np.asarray(field, dtype=np.float64))
    cc.persistence(homology_coeff_field=2, min_persistence=min_persistence)
    dgms, essential = {}, {}
    for dim in range(field.ndim):
        iv = np.asarray(cc.persistence_intervals_in_dimension(dim)).reshape(-1, 2)
        fin = np.isfinite(iv[:, 1])
        dgms[dim] = iv[fin]
        essential[dim] = iv[~fin, 0]
    return dict(dgms=dgms, essential=essential, seconds=time.perf_counter() - t0)


def lifetimes(dgm: np.ndarray) -> np.ndarray:
    return dgm[:, 1] - dgm[:, 0] if len(dgm) else np.zeros(0)


def filter_dgm(dgm: np.ndarray, min_pers: float) -> np.ndarray:
    return dgm[lifetimes(dgm) > min_pers] if len(dgm) else dgm


def count_long(dgm: np.ndarray, min_pers: float) -> int:
    return int(np.sum(lifetimes(dgm) > min_pers))


def bottleneck(a: np.ndarray, b: np.ndarray) -> float:
    """Exact bottleneck distance (GUDHI / CGAL); ~2 s for 27k-point diagrams."""
    return float(gudhi.bottleneck_distance(a, b))


def wasserstein2(a: np.ndarray, b: np.ndarray, delta: float = 0.01) -> float:
    """2-Wasserstein (internal L2) via Hera, relative error <= delta.
    (POT's exact solver does not finish on real diagrams with ~10^4-10^5 points.)"""
    return float(gudhi.hera.wasserstein_distance(a, b, order=2, internal_p=2, delta=delta))


def distances(res_a: dict, res_b: dict, min_pers: float = 0.0, w2: bool = True) -> dict:
    """Bottleneck (and optionally 2-Wasserstein) per homology dimension (finite points only)."""
    out = {}
    for dim in res_a["dgms"]:
        a = filter_dgm(res_a["dgms"][dim], min_pers)
        b = filter_dgm(res_b["dgms"][dim], min_pers)
        out[f"bottleneck_H{dim}"] = bottleneck(a, b)
        if w2:
            out[f"wasserstein2_H{dim}"] = wasserstein2(a, b)
    return out


def load_dgms(npz_path) -> dict:
    """Read diagrams saved by scripts/run_persistence.py back into the persistence() format."""
    z = np.load(npz_path)
    return dict(dgms={d: z[f"H{d}"] for d in range(3)}, essential={0: z["essential_H0"]})


def count_air_loops(dgm: np.ndarray, min_pers: float, death_max: float) -> int:
    """H1 loops above the cutoff that close only when the threshold drops below `death_max`
    density, i.e. loops whose hole is (nearly) empty space. GUDHI convention: death density = -dgm[:,1]."""
    if not len(dgm):
        return 0
    return int(np.sum((lifetimes(dgm) > min_pers) & (-dgm[:, 1] < death_max)))
