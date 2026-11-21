"""Write the subset of a scene that can touch the crop box, as a standard .ply (all properties).

Keeps every Gaussian whose truncated (cutoff-sigma) ellipsoid, after the anti-aliasing filter,
has an axis-aligned box-frame extent that comes within `margin` of the crop box. That is a
superset of what density.density_field uses (voxel centres lie inside the box, +/- half a voxel
for the shifted grid), so the density field of the subset is identical to that of the full
scene (checked on the original: scripts/check_crop_subset.py). File order is preserved.

Used because ada -> laptop transfer is too slow for full 1.5 GB decoded scenes. Needs only numpy
+ plyfile (runs in the FCGS env on ada, py3.7). Box parameters come from a small JSON written by
scripts/check_crop_subset.py --write-box (no yaml needed on ada).

    python scripts/extract_crop_subset.py box.json in.ply out.ply
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from topogs.io_ply import GaussianScene, load_ply, save_ply  # noqa: E402


def crop_mask(scene, box, chunk=1_000_000):
    R_box = np.array(box["R_box"]); cw = np.array(box["centre_world"]); half = np.array(box["half"])
    f, cut, m = box["filter_sigma"], box["cutoff_sigma"], box["margin"]
    keep = np.zeros(scene.n, dtype=bool)
    for a in range(0, scene.n, chunk):
        sub = GaussianScene(scene.raw[a:a + chunk])
        mu = (sub.means - cw) @ R_box.T
        R = np.einsum("ij,njk->nik", R_box, sub.rotmats)
        s = np.sqrt(sub.scales ** 2 + f ** 2)                       # peak-mode filtered scales
        ext = cut * np.sqrt(np.einsum("nij,nj->ni", R ** 2, s ** 2))
        keep[a:a + chunk] = np.all((mu + ext > -half - m) & (mu - ext < half + m), axis=1)
    return keep


def main(box_json, src, dst):
    box = json.load(open(box_json))
    scene = load_ply(src)
    keep = crop_mask(scene, box)
    save_ply(GaussianScene(scene.raw[keep].copy()), dst)
    info = dict(source=src, n_source=int(scene.n), n_subset=int(keep.sum()), box=box)
    json.dump(info, open(dst + ".subset.json", "w"), indent=2)
    print(json.dumps(dict(n_source=info["n_source"], n_subset=info["n_subset"])))


if __name__ == "__main__":
    main(*sys.argv[1:4])
