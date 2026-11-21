"""Self-made compression baselines. Every function returns a new
GaussianScene in INRIA .ply format (raw stored values), so the result can be written
with save_ply and loaded by any 3DGS renderer, plus a dict describing what was done
(pruning info includes `keep`, the kept indices into the original, so a variant can be
rebuilt exactly elsewhere without relying on the RNG of another numpy version).

- prune_opacity: drop the k lowest-opacity Gaussians of the WHOLE scene.
- prune_random:  drop k Gaussians chosen uniformly at random (control), fixed seed.
- quantize:      uniform min-max quantization of chosen raw properties (default:
                 positions and log-scales), per property, to `bits` bits, then
                 de-quantized back to float32 so the .ply stays standard.

Kept free of newer Python/numpy features so it also runs in the FCGS env (py3.7)
on ada, where variants are regenerated in memory for rendering instead of copied.
"""
from __future__ import annotations

import numpy as np

from .io_ply import GaussianScene

QUANT_FIELDS = ("x", "y", "z", "scale_0", "scale_1", "scale_2")


def _n_drop(n, frac):
    return int(np.floor(frac * n))


def prune_opacity(scene, frac):
    """Remove the floor(frac*N) lowest-opacity Gaussians; survivors keep their file order."""
    k = _n_drop(scene.n, frac)
    order = np.argsort(scene.raw["opacity"], kind="stable")  # sigmoid is monotone: sort raw logits
    keep = np.sort(order[k:])
    info = dict(kind="prune_opacity", frac=frac, n_in=scene.n, n_out=int(len(keep)),
                opacity_threshold=float(1.0 / (1.0 + np.exp(-scene.raw["opacity"][order[k]]))) if k < scene.n else None,
                keep=keep.astype(np.uint32))
    return scene.subset(keep), info


def prune_random(scene, frac, seed=0):
    """Remove floor(frac*N) Gaussians uniformly at random (numpy PCG64 with `seed`)."""
    k = _n_drop(scene.n, frac)
    rng = np.random.Generator(np.random.PCG64(seed))
    drop = rng.choice(scene.n, size=k, replace=False)
    mask = np.ones(scene.n, dtype=bool); mask[drop] = False
    return scene.subset(mask), dict(kind="prune_random", frac=frac, seed=seed, n_in=scene.n,
                                    n_out=int(mask.sum()), keep=np.flatnonzero(mask).astype(np.uint32))


def quantize(scene, bits, fields=QUANT_FIELDS):
    """Per-property uniform min-max quantization of the RAW stored values to 2**bits levels.
    Log-scales are quantized in log space (that is what the file stores)."""
    raw = scene.raw.copy()
    levels = 2 ** bits - 1
    steps = {}
    for f in fields:
        v = raw[f].astype(np.float64)
        lo, hi = float(v.min()), float(v.max())
        step = (hi - lo) / levels if hi > lo else 0.0
        if step > 0:
            q = np.rint((v - lo) / step)
            raw[f] = (lo + q * step).astype(np.float32)
        steps[f] = dict(min=lo, max=hi, step=step)
    info = dict(kind="quantize", bits=bits, fields=list(fields), n_in=scene.n, n_out=scene.n,
                steps=steps, nominal_payload_bits_per_gaussian=bits * len(fields))
    return GaussianScene(raw), info


def make_variant(scene, spec):
    """spec: dict(kind=..., frac=... | bits=..., seed=...)."""
    kind = spec["kind"]
    if kind == "prune_opacity":
        return prune_opacity(scene, spec["frac"])
    if kind == "prune_random":
        return prune_random(scene, spec["frac"], spec.get("seed", 0))
    if kind == "quantize":
        return quantize(scene, spec["bits"], tuple(spec.get("fields", QUANT_FIELDS)))
    raise ValueError(kind)
