"""Persistence diagrams for saved density fields (one variant, several resolutions).
Saves diagrams (.npz, GUDHI convention: (-birth_density, -death_density)), a stats JSON
(counts with/without the pre-registered cutoff, runtime, peak memory) and figures.

    python scripts/run_persistence.py configs/persistence_original.yaml
"""
import json
import pathlib
import resource
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from topogs.topology import count_long, lifetimes, persistence  # noqa: E402
from topogs.viz import diagram_and_barcode  # noqa: E402


def main(cfg_path):
    cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
    an = yaml.safe_load(pathlib.Path(cfg["analysis_config"]).read_text())
    fdir = pathlib.Path(cfg["field_dir"]); out = pathlib.Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for res in cfg["res"]:
        for off in cfg.get("offsets", [0.0]):
            tag = f"{cfg['name']}_r{res}_off{off}"
            field = np.load(fdir / f"{tag}.npy")
            r = persistence(field)
            np.savez(out / f"{tag}_dgms.npz", H0=r["dgms"][0], H1=r["dgms"][1], H2=r["dgms"][2],
                     essential_H0=r["essential"][0])
            cut = an["persistence_cutoff"].get(res)   # None before this resolution's noise floor exists
            st = dict(shape=list(field.shape), field_max=float(field.max()), seconds=round(r["seconds"], 1),
                      peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
                      essential_H0_birth_density=[float(-x) for x in r["essential"][0]])
            for d in range(3):
                L = lifetimes(r["dgms"][d])
                st[f"H{d}_all"] = int(len(L))
                st[f"H{d}_above_cutoff"] = count_long(r["dgms"][d], cut[f"H{d}"]) if cut else None
                st[f"H{d}_top5_persistence"] = np.sort(L)[::-1][:5].round(3).tolist()
            stats[tag] = st
            print(tag, json.dumps(st), flush=True)
            if off == 0.0 and cut:
                diagram_and_barcode(r, out / f"{tag}_diagram.png",
                                    title=f"{cfg['name']} {res}^3 (cutoff H1 = {cut['H1']:.3g})",
                                    cutoff=cut["H1"])
    sp = out / f"{cfg['name']}_stats.json"
    old = json.loads(sp.read_text()) if sp.exists() else {}
    old.update(stats)                                  # keep other resolutions' entries
    sp.write_text(json.dumps(old, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
