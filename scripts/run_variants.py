"""Build each compressed variant, write it as a standard .ply, compute its density fields
on the frozen analysis grid, its persistence diagrams, and its distances to the ORIGINAL.
Every analysis setting comes from configs/analysis.yaml (frozen from the original only).

Per variant -> results/variants/<name>/<name>.json (+ diagrams .npz, figures at 256^3);
results/variants/summary.csv is rebuilt from all JSONs at the end.

    python scripts/run_variants.py configs/variants.yaml [--only name1,name2] [--reuse-dgms]

--reuse-dgms: if <out_dir>/<name>/<tag>_dgms.npz already exists (e.g. computed on ada because the
laptop ran out of memory), load it instead of recomputing; everything else is unchanged.

The 'identity' variant pushes the unmodified original through this same path: all its
distances must be exactly 0 (proves same grid, filter and cutoff as the original's run).
"""
import argparse
import csv
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from topogs.compress import make_variant  # noqa: E402
from topogs.grids import compute_and_save, crop_grid, git_commit  # noqa: E402
from topogs.io_ply import load_ply, save_ply  # noqa: E402
from topogs.topology import (count_air_loops, count_long, distances, load_dgms,  # noqa: E402
                             persistence)
from topogs.viz import density_panels, diagram_and_barcode  # noqa: E402


def box_frame(an):
    crop = yaml.safe_load(pathlib.Path(an["crop_config"]).read_text())
    box = crop["boxes"][an["box"]]
    R = np.array(crop["frame"]["R_box"]); c = np.array(crop["frame"]["centre"])
    return R, c + R.T @ np.array(box["center"]), np.array(box["half"])


def zone_counts(scene, an):
    """Gaussians whose centre is inside the crop box / in the spoke zone (cause evidence)."""
    R, cw, half = box_frame(an)
    P = (scene.means - cw) @ R.T
    in_box = np.all(np.abs(P) < half, axis=1)
    rad = np.hypot(P[:, 0], P[:, 2])
    lo, hi = an["spoke_r"]
    spoke = (np.abs(P[:, 1]) < half[1]) & (rad > lo) & (rad < hi)
    op = scene.opacities
    return dict(n_in_box=int(in_box.sum()), n_spoke_zone=int(spoke.sum()),
                n_spoke_zone_op02=int((spoke & (op > 0.02)).sum()),
                spoke_zone_opacity_sum=round(float(op[spoke].sum()), 3)), P, in_box


def position_error_in_box(orig_P, var_P, var_in, half, voxel256, margin=0.05):
    """For a re-ordered variant (FCGS): distance from each in-box variant Gaussian to the nearest
    original Gaussian (box frame; originals within the box + margin), in 256^3 voxel units."""
    tree = cKDTree(orig_P[np.all(np.abs(orig_P) < half + margin, axis=1)])
    d, _ = tree.query(var_P[var_in], workers=-1)
    d = d / voxel256
    return dict(nn_dist_vox256_median=round(float(np.median(d)), 4),
                nn_dist_vox256_p95=round(float(np.percentile(d, 95)), 4),
                nn_dist_vox256_max=round(float(d.max()), 4))


def run_one(v, cfg, an, orig, orig_zone, reuse_dgms=False):
    name = v["name"]; kind = v["kind"]
    out = pathlib.Path(cfg["out_dir"]) / name; out.mkdir(parents=True, exist_ok=True)
    fdir = pathlib.Path(cfg["field_dir"]) / name
    rec = dict(name=name, spec=v, git_commit=git_commit(), analysis_config=cfg["analysis_config"])
    t0 = time.perf_counter()
    if kind == "identity":
        scene, ply_path, info = orig, cfg["original_ply"], dict(kind="identity")
    elif kind == "file":
        ply_path = v["ply"]; scene = load_ply(ply_path); info = dict(kind="file")
    else:
        scene, info = make_variant(orig, v)
        vdir = pathlib.Path(cfg["variant_dir"]) / name; vdir.mkdir(parents=True, exist_ok=True)
        ply_path = str(vdir / "point_cloud.ply")
        save_ply(scene, ply_path)
        if "keep" in info:
            np.save(vdir / "keep.npy", info.pop("keep"))
    rec["variant_info"] = info
    rec["ply"] = str(ply_path)
    rec["n_gaussians"] = int(v.get("n_gaussians_full", scene.n))
    rec["n_gaussians_in_file"] = int(scene.n)          # differs from n_gaussians for crop subsets
    rec["ply_bytes"] = pathlib.Path(ply_path).stat().st_size if "n_gaussians_full" not in v else ""
    rec["raw_sha256"] = hashlib.sha256(scene.raw.tobytes()).hexdigest()
    if "bitstream_bytes" in v:
        rec["bitstream_bytes"] = int(v["bitstream_bytes"])
    zc, P, in_box = zone_counts(scene, an)
    rec["zones"] = zc
    if kind == "file":
        _, oP, _ = orig_zone
        half = box_frame(an)[2]
        rec["position_error_in_box"] = position_error_in_box(oP, P, in_box, half, 2 * half.max() / 256)
    rec["seconds_build"] = round(time.perf_counter() - t0, 1)
    rec["res"] = {}
    for res in cfg["res"]:
        g = crop_grid(an["crop_config"], an["box"], res)
        tag = f"{name}_r{res}_off0.0"
        field, meta = compute_and_save(scene, g, fdir / f"{tag}.npy", source=str(ply_path),
                                       filter_sigma=an["filter_sigma"], mode=an["filter_mode"],
                                       cutoff=an["cutoff_sigma"])
        dg_path = out / f"{tag}_dgms.npz"
        if reuse_dgms and dg_path.exists():
            r = load_dgms(dg_path); r["seconds"] = float("nan"); rec.setdefault("reused_dgms", []).append(str(dg_path))
            print("reusing", dg_path, flush=True)
        else:
            r = persistence(field)
        np.savez(out / f"{tag}_dgms.npz", H0=r["dgms"][0], H1=r["dgms"][1], H2=r["dgms"][2],
                 essential_H0=r["essential"][0])
        o = load_dgms(pathlib.Path(cfg["original_dgms"]) / f"original_r{res}_off0.0_dgms.npz")
        cut = an["persistence_cutoff"][res]
        dfull = distances(o, r, w2=cfg.get("w2", True))
        dcut = {}
        for d in range(3):
            dcut.update(distances(dict(dgms={d: o["dgms"][d]}), dict(dgms={d: r["dgms"][d]}),
                                  min_pers=cut[f"H{d}"], w2=cfg.get("w2", True)))
        eta_b, eta_w = an["noise_floor"][res], an["noise_floor_w2"][res]
        ratio = {}
        for d in range(3):
            ratio[f"bottleneck_H{d}"] = round(dfull[f"bottleneck_H{d}"] / eta_b[f"H{d}"], 3)
            if f"wasserstein2_H{d}" in dfull:
                ratio[f"wasserstein2_H{d}"] = round(dfull[f"wasserstein2_H{d}"] / eta_w[f"H{d}"], 3)
        rr = dict(field_max=round(float(field.max()), 4), field_sum=round(float(field.sum()), 2),
                  n_gaussians_used=meta["n_gaussians_used"], seconds_persistence=round(r["seconds"], 1),
                  dist_full={k: round(x, 6) for k, x in dfull.items()},
                  dist_cut={k: round(x, 6) for k, x in dcut.items()},
                  dist_over_noise_floor=ratio)
        for d in range(3):
            rr[f"H{d}_all"] = int(len(r["dgms"][d]))
            rr[f"H{d}_above_cutoff"] = count_long(r["dgms"][d], cut[f"H{d}"])
        rr["H1_air_loops"] = count_air_loops(r["dgms"][1], cut["H1"], an["air_loop_death_max"])
        rec["res"][res] = rr
        print(name, res, json.dumps(rr), flush=True)
        if res == cfg.get("figure_res", 256):
            density_panels(field, g, out / f"{tag}_slices.png",
                           title=f"{name}, box {an['box']}, {res}^3")
            diagram_and_barcode(r, out / f"{tag}_diagram.png",
                                title=f"{name} {res}^3 (cutoff H1 = {cut['H1']:.3g})", cutoff=cut["H1"])
    rec["seconds_total"] = round(time.perf_counter() - t0, 1)
    (out / f"{name}.json").write_text(json.dumps(rec, indent=2))
    return rec


def summarize(cfg):
    rows = []
    for p in sorted(pathlib.Path(cfg["out_dir"]).glob("*/*.json")):
        rec = json.loads(p.read_text())
        for res, rr in rec["res"].items():
            row = dict(name=rec["name"], res=int(res), n_gaussians=rec["n_gaussians"],
                       ply_bytes=rec["ply_bytes"], bitstream_bytes=rec.get("bitstream_bytes", ""),
                       nominal_bits=rec["variant_info"].get("nominal_payload_bits_per_gaussian", ""),
                       **rec["zones"])
            row.update({k: rr[k] for k in ["H0_above_cutoff", "H1_above_cutoff", "H2_above_cutoff",
                                            "H1_air_loops", "field_max"]})
            row.update({f"full_{k}": x for k, x in rr["dist_full"].items()})
            row.update({f"cut_{k}": x for k, x in rr["dist_cut"].items()})
            row.update({f"ratio_{k}": x for k, x in rr["dist_over_noise_floor"].items()})
            rows.append(row)
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(pathlib.Path(cfg["out_dir"]) / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--only", default=""); ap.add_argument("--reuse-dgms", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(a.config).read_text())
    an = yaml.safe_load(pathlib.Path(cfg["analysis_config"]).read_text())
    only = set(filter(None, a.only.split(",")))
    orig = load_ply(cfg["original_ply"])
    zc, oP, o_in = zone_counts(orig, an)
    print("original zones", zc, flush=True)
    for v in cfg["variants"]:
        if only and v["name"] not in only:
            continue
        if v["kind"] == "file" and not pathlib.Path(v["ply"]).exists():
            print("skip (missing file)", v["name"], flush=True); continue
        run_one(v, cfg, an, orig, (zc, oP, o_in), reuse_dgms=a.reuse_dgms)
        summarize(cfg)


if __name__ == "__main__":
    main()
