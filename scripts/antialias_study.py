"""Anti-aliasing study on the ORIGINAL scene only (no compressed variants involved).

For each filter mode (peak / mass) and filter width f (multiples of the 256^3 voxel),
compute the density field on the crop box with and without a half-voxel grid shift,
then measure the topological noise floor: bottleneck / W2 distance between the two
persistence diagrams, per homology dimension. Also reports a "spoke level": the median
field value at the centres of spoke-zone Gaussians, to check the spokes stay above
the noise floor.

    python scripts/antialias_study.py configs/antialias.yaml
"""
import csv
import json
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from topogs.density import Grid, density_field  # noqa: E402
from topogs.io_ply import load_ply  # noqa: E402
from topogs.topology import distances, persistence  # noqa: E402


def main(cfg_path):
    cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
    crop = yaml.safe_load(pathlib.Path(cfg["crop_config"]).read_text())
    box = crop["boxes"][cfg["box"]]
    R = np.array(crop["frame"]["R_box"]); c = np.array(crop["frame"]["centre"])
    cw = c + R.T @ np.array(box["center"])
    half = np.array(box["half"])
    out = pathlib.Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    scene = load_ply(crop["ply"])
    v256 = 2 * half.max() / 256

    # spoke-zone Gaussians (in the box frame, relative to the box centre)
    P = (scene.means - cw) @ R.T
    rad = np.hypot(P[:, 0], P[:, 2])
    spoke = (np.abs(P[:, 1]) < half[1]) & (rad > cfg["spoke_r"][0]) & (rad < cfg["spoke_r"][1]) \
        & (scene.opacities > 0.02)

    rows, images = [], {}
    for run in cfg["runs"]:
        mode, mult = run["mode"], float(run["filter_mult"])
        f_sigma = mult * v256
        for res in run["res"]:
            res_ph = {}
            for off in (0.0, 0.5):
                g = Grid.cubic(cw, half, res, R_box=R, offset=off)
                t0 = time.perf_counter()
                field, _ = density_field(scene, g, filter_sigma=f_sigma, mode=mode)
                t_dens = time.perf_counter() - t0
                res_ph[off] = persistence(field)
                if off == 0.0:
                    idx = np.clip(np.round((P[spoke] - g.origin) / g.spacing).astype(int), 0,
                                  np.array(g.shape) - 1)
                    spoke_level = float(np.median(field[idx[:, 0], idx[:, 1], idx[:, 2]]))
                    fmax = float(field.max())
                    if res == 256:
                        images[(mode, mult)] = field.max(axis=1)
            d = distances(res_ph[0.0], res_ph[0.5], w2=cfg.get("w2", False))
            row = dict(mode=mode, filter_mult=mult, filter_sigma=round(f_sigma, 6), res=res,
                       field_max=round(fmax, 4), spoke_level=round(spoke_level, 5),
                       n_H0=len(res_ph[0.0]["dgms"][0]), n_H1=len(res_ph[0.0]["dgms"][1]),
                       n_H2=len(res_ph[0.0]["dgms"][2]),
                       t_density=round(t_dens, 1), t_persistence=round(res_ph[0.0]["seconds"], 1),
                       **{k: round(v, 6) for k, v in d.items()})
            rows.append(row)
            print(json.dumps(row), flush=True)
            with open(out / "noise_floor.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

    keys = sorted(images)
    fig, axs = plt.subplots(2, (len(keys) + 1) // 2, figsize=(5 * ((len(keys) + 1) // 2), 10))
    for ax, k in zip(np.ravel(axs), keys):
        ax.imshow(np.log10(images[k].T + 1e-3), origin="lower", cmap="magma", vmin=-3, vmax=1.5)
        ax.set_title(f"{k[0]}, f = {k[1]} x voxel256"); ax.set_xticks([]); ax.set_yticks([])
    for ax in np.ravel(axs)[len(keys):]:
        ax.axis("off")
    fig.suptitle(f"Box {cfg['box']} at 256^3: max over depth of log10 density, for each anti-aliasing filter")
    fig.tight_layout(); fig.savefig(out / "filters_maxproj.png", dpi=80); plt.close(fig)
    (out / "config_used.yaml").write_text(yaml.safe_dump({"study": cfg, "crop": crop}))


if __name__ == "__main__":
    main(sys.argv[1])
