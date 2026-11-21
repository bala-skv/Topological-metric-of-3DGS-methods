"""Build the crop grid from configs, compute + save density fields with a JSON sidecar."""
from __future__ import annotations

import json
import pathlib
import subprocess
import time

import numpy as np
import yaml

from .density import Grid, density_field


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def crop_grid(crop_cfg_path, box_name: str, res: int, offset: float = 0.0) -> Grid:
    crop = yaml.safe_load(pathlib.Path(crop_cfg_path).read_text())
    box = crop["boxes"][box_name]
    R = np.array(crop["frame"]["R_box"]); c = np.array(crop["frame"]["centre"])
    cw = c + R.T @ np.array(box["center"])
    return Grid.cubic(cw, box["half"], res, R_box=R, offset=offset)


def compute_and_save(scene, grid: Grid, out_npy, *, source: str, filter_sigma: float,
                     mode: str = "peak", cutoff: float = 4.0, extra: dict | None = None):
    t0 = time.perf_counter()
    field, n_used = density_field(scene, grid, cutoff=cutoff, filter_sigma=filter_sigma, mode=mode)
    secs = time.perf_counter() - t0
    out_npy = pathlib.Path(out_npy); out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, field.astype(np.float32))
    meta = dict(grid=grid.to_dict(), spacing=grid.spacing.tolist(), source=str(source),
                n_gaussians_scene=int(scene.n), n_gaussians_used=n_used, cutoff_sigma=cutoff,
                filter_sigma=filter_sigma, filter_mode=mode, dtype_saved="float32",
                seconds=round(secs, 2), git_commit=git_commit(), **(extra or {}))
    out_npy.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return field.astype(np.float32), meta
