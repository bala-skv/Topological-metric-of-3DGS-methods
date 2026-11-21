"""Do image errors of opacity pruning concentrate on the spokes? (saved views only)

Spoke mask per view: project the ORIGINAL's spoke-zone Gaussians (box frame: radius in spoke_r in the
u-w plane, |v| < half_v, opacity > 0.02; same zone as run_variants.zone_counts) with the INRIA camera,
splat each centre to its pixel and dilate by 2 px. Inside the wheel bbox we report, per variant:
  share of squared error (vs the ORIGINAL render) on mask pixels, mask pixel fraction,
  PSNR vs ground truth on the mask and on the rest of the wheel box.

    python scripts/spoke_mask_check.py [RENDER_DIR]      # default results/render_wheel_png (3 saved views)

Uses every test view that has a saved PNG for the original in RENDER_DIR; the summary is the mean over
views (each view weighted equally, like the wheel-region metrics).
"""
import json
import pathlib
import sys

import numpy as np
import yaml
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from render_sheet import GT, load, project  # noqa: E402
from run_variants import box_frame  # noqa: E402
from topogs.io_ply import load_ply  # noqa: E402

VARS = ["fcgs_1e-4", "fcgs_4e-4", "fcgs_16e-4", "quant16", "opac30", "opac50", "rand30", "rand30_s1", "rand30_s2"]


def psnr(a, b):
    return float(-10 * np.log10(np.mean((a - b) ** 2)))


def main(render_dir="results/render_wheel_png"):
    R = pathlib.Path(render_dir)
    rows = {r["view"]: r for r in json.load(open(R / "original_metrics.json"))["per_view"]}
    VIEWS = [v for v in sorted(rows) if (R / f"original_{v}.png").exists() and rows[v].get("wheel")]
    vars_ = [v for v in VARS if (R / f"{v}_{VIEWS[0]}.png").exists()]
    an = yaml.safe_load(open("configs/analysis.yaml"))
    s = load_ply("data/pretrained/bicycle/point_cloud/iteration_30000/point_cloud.ply")
    Rb, cw, half = box_frame(an)
    P = (s.means - cw) @ Rb.T
    rad = np.hypot(P[:, 0], P[:, 2]); lo, hi = an["spoke_r"]
    m = (np.abs(P[:, 1]) < half[1]) & (rad > lo) & (rad < hi) & (s.opacities > 0.02)
    pts = s.means[m]; print("spoke-zone Gaussians", len(pts))
    cams = {c["img_name"]: c for c in json.load(open("data/pretrained/bicycle/cameras.json"))}
    out = {}
    for view in VIEWS:
        gt = load(GT / f"{view}.JPG"); H, W = gt.shape[:2]
        x0, x1, y0, y1 = rows[view]["wheel"]["bbox"]
        px = np.array([project(p, cams[view], W, H) for p in pts])
        mask = np.zeros((H, W), bool)
        ok = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
        mask[np.round(px[ok, 1]).astype(int).clip(0, H - 1), np.round(px[ok, 0]).astype(int).clip(0, W - 1)] = True
        mask = binary_dilation(mask, iterations=2)
        box = np.zeros((H, W), bool); box[y0:y1, x0:x1] = True
        sp, rest = mask & box, box & ~mask
        orig = load(R / f"original_{view}.png")
        row = dict(mask_frac_of_box=round(float(sp.sum() / box.sum()), 3),
                   original=dict(psnr_gt_spokes=round(psnr(orig[sp], gt[sp]), 2), psnr_gt_rest=round(psnr(orig[rest], gt[rest]), 2)))
        for v in vars_:
            img = load(R / f"{v}_{view}.png")
            e = ((img - orig) ** 2).sum(axis=2)
            row[v] = dict(err_share_on_spokes=round(float(e[sp].sum() / e[box].sum()), 3),
                          psnr_gt_spokes=round(psnr(img[sp], gt[sp]), 2), psnr_gt_rest=round(psnr(img[rest], gt[rest]), 2))
        out[view] = row
    summ = dict(n_views=len(VIEWS), mask_frac_of_box_mean=round(float(np.mean([out[w]["mask_frac_of_box"] for w in VIEWS])), 3))
    print(f"{len(VIEWS)} views; spoke mask = {summ['mask_frac_of_box_mean']:.3f} of the wheel box (mean)")
    print(f"   {'':12s} err share on spokes | ΔPSNR(gt) on spokes | ΔPSNR(gt) rest of wheel box   (means over views)")
    for v in vars_:
        sh = np.mean([out[w][v]["err_share_on_spokes"] for w in VIEWS])
        ds = np.mean([out[w]["original"]["psnr_gt_spokes"] - out[w][v]["psnr_gt_spokes"] for w in VIEWS])
        dr = np.mean([out[w]["original"]["psnr_gt_rest"] - out[w][v]["psnr_gt_rest"] for w in VIEWS])
        summ[v] = dict(err_share_on_spokes=round(float(sh), 3), d_psnr_spokes=round(float(ds), 3), d_psnr_rest=round(float(dr), 3))
        print(f"   {v:12s} {sh:.2f} | {ds:+.2f} | {dr:+.2f}")
    name = "spoke_mask_check.json" if len(VIEWS) <= 3 else "spoke_mask_all_views.json"
    pathlib.Path("results/explain", name).write_text(json.dumps(dict(summary=summ, per_view=out), indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
