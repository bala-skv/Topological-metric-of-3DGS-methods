"""Persistence diagrams for HAC-audit fields (same topogs.topology.persistence as every other run).
One field per call, so each heavy run can sit in its own memory-capped process:

    systemd-run --user --scope -p MemoryMax=14G -p MemorySwapMax=4G \
        .venv/bin/python scripts/hac_persistence.py results/fields/hac/scaffold_mean_r256_off0.0.npy

Writes results/hac/dgms/<field stem>_dgms.npz (GUDHI convention, as run_persistence.py) and a
<stem>_stats.json (runtime, peak RSS, counts). Skips fields whose diagrams already exist.
"""
import json
import pathlib
import resource
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from topogs.topology import persistence  # noqa: E402

OUT = pathlib.Path("results/hac/dgms")


def main(*paths):
    OUT.mkdir(parents=True, exist_ok=True)
    for p in map(pathlib.Path, paths):
        dg = OUT / f"{p.stem}_dgms.npz"
        if dg.exists():
            print("exists", dg); continue
        f = np.load(p)
        r = persistence(f)
        np.savez(dg, H0=r["dgms"][0], H1=r["dgms"][1], H2=r["dgms"][2], essential_H0=r["essential"][0])
        st = dict(field=str(p), shape=list(f.shape), field_max=float(f.max()), seconds=round(r["seconds"], 1),
                  peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
                  n=[int(len(r["dgms"][d])) for d in range(3)])
        (OUT / f"{p.stem}_stats.json").write_text(json.dumps(st, indent=2))
        print(p.stem, json.dumps(st), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
