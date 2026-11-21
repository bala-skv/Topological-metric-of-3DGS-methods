"""HAC step 5: noise floors of the UNCOMPRESSED Scaffold-GS baseline, computed and frozen before any HAC
distance exists. Only Scaffold-GS diagrams are read here.

 (1) half-voxel floor eta: Scaffold-GS mean field on the analysis grid vs the same grid shifted by half a
     voxel (as scripts/noise_floor.py for the original); bottleneck and 2-Wasserstein per H0/H1/H2.
     Persistence cutoff tau_d = 2 * eta_d (bottleneck), the original's rule.
 (2) camera-spread floor: max over the 8 export cameras of the distance between that camera's field and
     the mean field (bottleneck and W2, per dimension).
 (3) effective floor = max(half-voxel, camera spread), per measure.
All four measures are computed on full diagrams and with tau applied to both diagrams.

    python scripts/hac_floors.py RES [RES ...]       -> results/hac/floors/floors_r<RES>.json
    python scripts/hac_floors.py --write-config      -> configs/analysis_hac.yaml (from all floors_r*.json)
"""
import json
import pathlib
import sys
from multiprocessing import Pool

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from topogs.topology import distances, load_dgms  # noqa: E402

DG = pathlib.Path("results/hac/dgms")
OUT = pathlib.Path("results/hac/floors")
MEASURES = [f"{m}_H{d}" for m in ("bottleneck", "wasserstein2") for d in range(3)]


def dists(a, b, tau):
    full = distances(a, b)
    cut = {}
    for d in range(3):
        cut.update(distances(dict(dgms={d: a["dgms"][d]}), dict(dgms={d: b["dgms"][d]}), min_pers=tau[f"H{d}"]))
    return dict(full={k: round(v, 6) for k, v in full.items()}, cut={k: round(v, 6) for k, v in cut.items()})


def _cam_job(args):
    cam, res, tau = args
    a = load_dgms(DG / f"scaffold_{cam}_r{res}_off0.0_dgms.npz")
    m = load_dgms(DG / f"scaffold_mean_r{res}_off0.0_dgms.npz")
    r = dists(a, m, tau); print(res, cam, json.dumps(r), flush=True)
    return cam, r


def floors(res):
    cams = yaml.safe_load(open("configs/hac_export.yaml"))["cameras"]
    m0 = load_dgms(DG / f"scaffold_mean_r{res}_off0.0_dgms.npz")
    m5 = load_dgms(DG / f"scaffold_mean_r{res}_off0.5_dgms.npz")
    full = distances(m0, m5)
    tau = {f"H{d}": round(2 * full[f"bottleneck_H{d}"], 6) for d in range(3)}
    hv = dists(m0, m5, tau)
    print(res, "half-voxel", json.dumps(hv), "tau", tau, flush=True)
    with Pool(4) as p:
        per_cam = dict(p.map(_cam_job, [(c, res, tau) for c in cams]))
    spread = {k: {m: max(per_cam[c][k][m] for c in cams) for m in MEASURES} for k in ("full", "cut")}
    argmax = {k: {m: max(cams, key=lambda c: per_cam[c][k][m]) for m in MEASURES} for k in ("full", "cut")}
    eff = {k: {m: max(hv[k][m], spread[k][m]) for m in MEASURES} for k in ("full", "cut")}
    src = {k: {m: ("half_voxel" if hv[k][m] >= spread[k][m] else "camera_spread") for m in MEASURES} for k in ("full", "cut")}
    rec = dict(res=res, tau=tau, half_voxel=hv, camera_spread=spread, camera_spread_argmax=argmax,
               effective=eff, effective_source=src, per_camera=per_cam)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"floors_r{res}.json").write_text(json.dumps(rec, indent=2))


def write_config():
    recs = {int(p.stem.split("_r")[1]): json.loads(p.read_text()) for p in sorted(OUT.glob("floors_r*.json"))}
    r6 = lambda d: {k: round(v, 6) for k, v in d.items()}
    hdr = """# HAC audit settings. Written by scripts/hac_floors.py --write-config from the UNCOMPRESSED Scaffold-GS
# baseline ONLY (results/hac/floors/), and committed BEFORE any HAC persistence diagram or distance was
# computed. configs/analysis.yaml (the original 3DGS model's frozen values) is NOT changed; the field
# definition (filter, truncation, box, grids) is taken from it unchanged.
# Frozen 21 Nov ~11:10 IST for 128/256/384 (commit before any HAC persistence). 512 may be appended later by
# the same script from Scaffold-GS only, again before any HAC 512 diagram (as was done for the original).
# Representative field of a model = mean of its 8 per-camera fields (configs/hac_export.yaml).
# Floors (per resolution, per measure; 'full' = whole diagrams, 'cut' = tau applied to both diagrams):
#   half_voxel:    Scaffold-GS mean field vs the same grid shifted by half a voxel
#   camera_spread: max over the 8 cameras of d(Scaffold-GS per-camera field, Scaffold-GS mean field)
#   effective:     max(half_voxel, camera_spread)  <- HAC-vs-Scaffold-GS distances are divided by this
# tau_d = 2 x half-voxel bottleneck floor (Scaffold-GS), the original's rule.
# Air loops: same definition as the original (persistence > tau_H1 and death density < air_loop_death_max).
#   Primary cutoff: Scaffold-GS's own tau_H1 (this file). Secondary, for comparison with the original's
#   31/35/37/37 loops: the original's tau_H1 from configs/analysis.yaml.
# Weak links: HAC density / Scaffold-GS density (mean fields) at the birth cells of Scaffold-GS's air loops.
#   (The original's 'typical spoke' band [1,10] is not used: it was set on the original's density scale.)
# Decision rule (pre-registered, unchanged): a change counts as real if the H1 bottleneck distance
#   HAC vs Scaffold-GS (full diagrams) exceeds the effective floor at BOTH 256^3 and 384^3.
#   128^3 and 512^3 are reported as extra columns.
"""
    cfg = dict(base_analysis_config="configs/analysis.yaml", export_config="configs/hac_export.yaml",
               reference="scaffold", representative_field="mean_of_cameras",
               air_loop_death_max=0.05, air_loop_cutoff_primary="scaffold_tau",
               decision_rule=dict(measure="bottleneck_H1", diagrams="full", resolutions=[256, 384]),
               persistence_cutoff={r: rec["tau"] for r, rec in recs.items()},
               floors={r: {k: {"half_voxel": r6(rec["half_voxel"][k]), "camera_spread": r6(rec["camera_spread"][k]),
                               "effective": r6(rec["effective"][k])} for k in ("full", "cut")}
                       for r, rec in recs.items()})
    pathlib.Path("configs/analysis_hac.yaml").write_text(hdr + yaml.safe_dump(cfg, sort_keys=False, width=200))
    print(open("configs/analysis_hac.yaml").read())


if __name__ == "__main__":
    if sys.argv[1:] == ["--write-config"]:
        write_config()
    else:
        for r in sys.argv[1:]:
            floors(int(r))
