"""Matching-free check of WHAT a variant weakened, at fixed places in the wheel box:

 (a) typical spoke level: all spoke-band voxels (radius spoke_r in the u-w plane) where the
     ORIGINAL's density is in the spoke range [1, 10]; per-voxel ratio variant/original, median.
     (Voxel probes, not Gaussian centres: centres of surviving Gaussians would be biased towards
     places pruning cannot touch.)
 (b) weakest links: the birth cell of each of the original's air-enclosing H1 loops (the voxel
     whose arrival closes the ring around a spoke gap, i.e. the lowest-density point of that ring);
     ratio variant/original at that voxel: median, 10th percentile, min, max.

Density is a sum over Gaussians and pruning only removes terms, so for pruning a ratio r means
that (1 - r) of the density at that point came from the removed Gaussians.

    python scripts/structure_levels.py 256 384
"""
import json
import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.explain_change import pairs_with_cells  # noqa: E402
from topogs.grids import crop_grid  # noqa: E402

VARIANTS = ["fcgs_1e-4", "fcgs_4e-4", "fcgs_16e-4", "quant16", "quant8", "opac30", "opac50", "opac70",
            "rand30", "rand30_s1", "rand30_s2", "rand50", "rand70"]


def field(v, res):
    return np.load(f"results/fields/{v}/{v}_r{res}_off0.0.npy")


def main(*resolutions):
    an = yaml.safe_load(open("configs/analysis.yaml"))
    out = {}
    for res in map(int, resolutions):
        f0 = field("original", res)
        g = crop_grid(an["crop_config"], an["box"], res); au, av, aw = g.axes()
        U, W = np.meshgrid(au, aw, indexing="ij")
        rad = np.broadcast_to(np.hypot(U, W)[:, None, :], f0.shape)
        lo, hi = an["spoke_r"]
        typical = (rad > lo) & (rad < hi) & (f0 >= 1) & (f0 <= 10)
        D, B, _ = pairs_with_cells(f0)
        b, d = -D[:, 0], -D[:, 1]
        air = ((b - d) > an["persistence_cutoff"][res]["H1"]) & (d < an["air_loop_death_max"])
        Bc = B[air]; b0 = f0[Bc[:, 0], Bc[:, 1], Bc[:, 2]]
        out[res] = dict(n_typical_voxels=int(typical.sum()), n_weak_links=int(air.sum()), variants={})
        print(f"{res}^3: {int(typical.sum())} spoke-band voxels in [1,10]; {int(air.sum())} weakest links "
              f"(original density median {np.median(b0):.2f})")
        for v in VARIANTS:
            p = pathlib.Path(f"results/fields/{v}/{v}_r{res}_off0.0.npy")
            if not p.exists():
                continue
            f1 = np.load(p)
            t = np.median(f1[typical] / f0[typical])
            w = f1[Bc[:, 0], Bc[:, 1], Bc[:, 2]] / b0
            row = dict(typical_median=round(float(t), 3), weak_median=round(float(np.median(w)), 3),
                       weak_p10=round(float(np.percentile(w, 10)), 3), weak_min=round(float(w.min()), 3),
                       weak_max=round(float(w.max()), 3))
            out[res]["variants"][v] = row
            print(f"   {v:11s} typical spoke {row['typical_median']:.2f}x | weakest links median {row['weak_median']:.2f}x "
                  f"(p10 {row['weak_p10']:.2f}, min {row['weak_min']:.2f}, max {row['weak_max']:.2f})")
    pth = pathlib.Path("results/explain"); pth.mkdir(parents=True, exist_ok=True)
    f = pth / "structure_levels.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    old.update({str(k): v for k, v in out.items()})        # keep other resolutions' entries
    f.write_text(json.dumps(old, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
