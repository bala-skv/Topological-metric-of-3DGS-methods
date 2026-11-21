"""Noise floors of the ORIGINAL scene: distance between its diagrams on the analysis grid and
on the same grid shifted by half a voxel. Bottleneck and 2-Wasserstein, without and with the
pre-registered persistence cutoff applied to both diagrams. Variants are never used here.

    python scripts/noise_floor.py configs/analysis.yaml results/persistence/original results/persistence/original_shift [RES ...]

With explicit RES (e.g. 512, a resolution added later) the output is noise_floor_r<RES...>.json, and
if that resolution has no frozen cutoff yet, tau_d = 2 * (full-diagram bottleneck floor), which is
how the frozen cutoffs were defined.
"""
import json
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from topogs.topology import distances, load_dgms  # noqa: E402


def main(an_path, d0, d1, *res_list):
    an = yaml.safe_load(pathlib.Path(an_path).read_text())
    out = {}
    for res in (list(map(int, res_list)) or an["resolutions"]):
        a = load_dgms(pathlib.Path(d0) / f"original_r{res}_off0.0_dgms.npz")
        b = load_dgms(pathlib.Path(d1) / f"original_r{res}_off0.5_dgms.npz")
        full = distances(a, b)
        cut = {}
        for d in range(3):
            tau = (an["persistence_cutoff"][res][f"H{d}"] if res in an["persistence_cutoff"]
                   else 2 * full[f"bottleneck_H{d}"])
            r = distances(dict(dgms={d: a["dgms"][d]}), dict(dgms={d: b["dgms"][d]}), min_pers=tau)
            cut.update(r)
        out[res] = dict(full={k: round(v, 6) for k, v in full.items()}, cut={k: round(v, 6) for k, v in cut.items()})
        print(res, json.dumps(out[res]), flush=True)
    p = pathlib.Path(d1) / ("noise_floor_r" + "_".join(res_list) + ".json" if res_list else "noise_floor.json")
    p.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
