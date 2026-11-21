"""HAC step 4: density fields of the exported neural Gaussians, with exactly the repo's field definition
(configs/analysis.yaml: peak filter f = 0.0076562, 4-sigma truncation, box B, same grids).

For each model and camera, the crop subset written by scripts/hac_validate_export.py
(data/hac/export/<model>/<cam>_crop.ply) is sampled; the model's representative field is the mean of its
per-camera fields (accumulated in float64 from the saved float32 fields, saved as float32 like every
other field). Fields go to results/fields/hac/<model>_<cam>_r<res>_off<off>.npy (+ JSON sidecar) and
results/fields/hac/<model>_mean_r<res>_off<off>.npy.

    python scripts/hac_fields.py MODEL --res 256 384 [--offsets 0.0 0.5] [--rep]

--rep: the identity re-export of the first camera (<cam>_rep_crop.ply), offset 0 only.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from topogs.grids import compute_and_save, crop_grid, git_commit  # noqa: E402
from topogs.io_ply import load_ply  # noqa: E402

OUT = pathlib.Path("results/fields/hac")
# model name -> crop subsets. Primary HAC set = zero-scale Gaussians KEPT (configs/hac_export.yaml, amendment 1,
# export job 95700). `hacdrop` = the pre-registered drop rule, which FAILED validation (b): sensitivity only.
# Scaffold-GS has no zero-scale Gaussians; its job-1 subsets are sha256-identical to job 2's.
SOURCES = {"scaffold": "data/hac/export/scaffold/{cam}_crop.ply",
           "hac": "data/hac/export2/hac/{cam}_keepzero_crop.ply",
           "hacdrop": "data/hac/export/hac/{cam}_crop.ply"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model")
    ap.add_argument("--res", type=int, nargs="+", required=True)
    ap.add_argument("--offsets", type=float, nargs="+", default=[0.0]); ap.add_argument("--rep", action="store_true")
    a = ap.parse_args()
    an = yaml.safe_load(open("configs/analysis.yaml"))
    cams = yaml.safe_load(open("configs/hac_export.yaml"))["cameras"]
    if a.rep:
        cams = [cams[0] + "_rep"]
    scenes = {}
    for c in cams:
        p = pathlib.Path(SOURCES[a.model].format(cam=c))
        scenes[c] = (load_ply(p), p)
    for res in a.res:
        for off in a.offsets:
            g = crop_grid(an["crop_config"], an["box"], res, offset=off)
            acc = None
            for c in cams:
                sc, p = scenes[c]
                f, meta = compute_and_save(sc, g, OUT / f"{a.model}_{c}_r{res}_off{off}.npy", source=str(p),
                                           filter_sigma=an["filter_sigma"], mode=an["filter_mode"],
                                           cutoff=an["cutoff_sigma"])
                assert np.isfinite(f).all(), f"non-finite field {a.model} {c} {res}"
                acc = f.astype(np.float64) if acc is None else acc + f
                print(a.model, c, res, off, f"n={sc.n} used={meta['n_gaussians_used']} max={f.max():.3f} {meta['seconds']}s",
                      flush=True)
            if a.rep:
                continue
            mean = (acc / len(cams)).astype(np.float32)
            mp = OUT / f"{a.model}_mean_r{res}_off{off}.npy"
            np.save(mp, mean)
            mp.with_suffix(".json").write_text(json.dumps(dict(
                grid=g.to_dict(), spacing=g.spacing.tolist(), source=[str(scenes[c][1]) for c in cams],
                definition="mean of the per-camera fields (float64 accumulation of the saved float32 fields)",
                filter_sigma=an["filter_sigma"], filter_mode=an["filter_mode"], cutoff_sigma=an["cutoff_sigma"],
                dtype_saved="float32", git_commit=git_commit()), indent=2))
            print(a.model, "mean", res, off, f"max={mean.max():.3f} sum={mean.sum():.1f}", flush=True)


if __name__ == "__main__":
    main()
