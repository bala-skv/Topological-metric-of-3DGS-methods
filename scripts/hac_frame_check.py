"""HAC step 1: do the HAC / Scaffold-GS models live in the same world frame as the pretrained 3DGS
scene (so the crop box B applies unchanged)? Compares camera centres and rotations in each model's
cameras.json with data/pretrained/bicycle/cameras.json, matched by image name.

    python scripts/hac_frame_check.py  -> results/hac/frame_check.json
"""
import json
import pathlib

import numpy as np

REF = "data/pretrained/bicycle/cameras.json"
MODELS = {"hac": "data/hac/hac_bicycle_l0.004/cameras.json", "scaffold": "data/hac/scaffold_bicycle/cameras.json"}


def load(p):
    return {c["img_name"]: c for c in json.load(open(p))}


def main():
    ref = load(REF); out = {}
    for m, p in MODELS.items():
        cams = load(p); common = sorted(set(ref) & set(cams))
        dp = [np.linalg.norm(np.subtract(ref[k]["position"], cams[k]["position"])) for k in common]
        dr = [np.abs(np.subtract(ref[k]["rotation"], cams[k]["rotation"])).max() for k in common]
        df = [abs(ref[k]["fx"] - cams[k]["fx"]) + abs(ref[k]["fy"] - cams[k]["fy"]) for k in common]
        out[m] = dict(n_ref=len(ref), n_model=len(cams), n_common=len(common), centre_diff_max=float(max(dp)),
                      centre_diff_median=float(np.median(dp)), rotation_diff_max=float(max(dr)), focal_diff_max=float(max(df)),
                      same_frame=bool(len(common) == len(ref) and max(dp) < 1e-6 and max(dr) < 1e-6))
        print(m, out[m])
    pathlib.Path("results/hac").mkdir(parents=True, exist_ok=True)
    pathlib.Path("results/hac/frame_check.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
