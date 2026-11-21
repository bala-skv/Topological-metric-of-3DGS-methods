"""HAC step 7: image side of the HAC vs Scaffold-GS comparison (numpy only, laptop).

 (1) test-set PSNR / SSIM / LPIPS from each model's own results.json (25 views, 1600 px, the repos' evaluate()).
 (2) wheel box, all 25 test views, from the renders the training jobs saved (test/ours_30000/renders, 8-bit):
     - PSNR(HAC render, Scaffold-GS render), full frame and wheel box: how much HAC's image differs from its
       baseline's (the analogue of "PSNR vs our original render" used for FCGS);
     - EXTRA, approximate: PSNR vs ground truth on the wheel box, with both renders resized (PIL bicubic) to
       the images_4 photos (1237 x 822) we have locally; the 1600-px ground truth used by the repos is not on
       the laptop. Same resampling for both models, so only the difference HAC - Scaffold-GS is meaningful.
 Wheel box = projected original-model Gaussians in crop box B (opacity > 0.02), dilated 5%, exactly the
 definition of scripts/render_eval.py wheel_bbox (ported to numpy).

    python scripts/hac_image_side.py -> results/hac/image_side.json
"""
import json
import math
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from topogs.io_ply import load_ply  # noqa: E402

MODELS = {"scaffold": "data/hac/scaffold_bicycle", "hac": "data/hac/hac_bicycle_l0.004"}


def cam_mats(c):
    c2w = np.eye(4); c2w[:3, :3] = np.array(c["rotation"]); c2w[:3, 3] = np.array(c["position"])
    view = np.linalg.inv(c2w).T                                   # INRIA stores transposed
    fovx = 2 * math.atan(c["width"] / (2 * c["fx"])); fovy = 2 * math.atan(c["height"] / (2 * c["fy"]))
    zn, zf = 0.01, 100.0; tx, ty = math.tan(fovx / 2), math.tan(fovy / 2)
    P = np.zeros((4, 4)); P[0, 0] = 1 / tx; P[1, 1] = 1 / ty; P[3, 2] = 1.0
    P[2, 2] = zf / (zf - zn); P[2, 3] = -(zf * zn) / (zf - zn)
    return view, view @ P.T


def wheel_bbox(pts, c, W, H, dilate=0.05):
    view, full = cam_mats(c)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], 1)
    z = (hom @ view)[:, 2]; ph = hom @ full; ndc = ph[:, :3] / ph[:, 3:4]
    px = ((ndc[:, 0] + 1) * W - 1) * 0.5; py = ((ndc[:, 1] + 1) * H - 1) * 0.5
    ok = (z > 0.2) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    if ok.mean() < 0.5:
        return None
    x0, x1, y0, y1 = px[ok].min(), px[ok].max(), py[ok].min(), py[ok].max()
    dx, dy = dilate * (x1 - x0), dilate * (y1 - y0)
    return (int(max(0, x0 - dx)), int(min(W, x1 + dx + 1)), int(max(0, y0 - dy)), int(min(H, y1 + dy + 1)))


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2) / 255.0 ** 2
    return float("inf") if mse == 0 else 10 * np.log10(1 / mse)


def main():
    box = json.load(open("configs/crop_box.json")); s = load_ply("data/fcgs_crop/original_crop.ply")
    R = np.array(box["R_box"]); P = (s.means - np.array(box["centre_world"])) @ R.T
    pts = s.means[np.all(np.abs(P) < np.array(box["half"]), axis=1) & (s.opacities > 0.02)]
    cams = sorted(json.load(open(MODELS["scaffold"] + "/cameras.json")), key=lambda c: c["img_name"])
    test = [c for i, c in enumerate(cams) if i % 8 == 0]
    out = dict(results_json={m: json.load(open(p + "/results.json"))["ours_30000"] for m, p in MODELS.items()}, views={})
    for i, c in enumerate(test):
        r = {m: np.asarray(Image.open(f"{p}/test/ours_30000/renders/{i:05d}.png").convert("RGB")) for m, p in MODELS.items()}
        H, W = r["hac"].shape[:2]
        bb = wheel_bbox(pts, c, W, H)
        v = dict(hac_vs_scaffold_full=psnr(r["hac"], r["scaffold"]))
        gt = np.asarray(Image.open(f"data/mipnerf360/bicycle/images_4/{c['img_name']}.JPG").convert("RGB"))
        h4, w4 = gt.shape[:2]
        bb4 = wheel_bbox(pts, c, w4, h4)
        if bb is not None and bb4 is not None:
            x0, x1, y0, y1 = bb
            v["hac_vs_scaffold_wheel"] = psnr(r["hac"][y0:y1, x0:x1], r["scaffold"][y0:y1, x0:x1])
            x0, x1, y0, y1 = bb4
            for m in MODELS:
                rs = np.asarray(Image.fromarray(r[m]).resize((w4, h4), Image.BICUBIC))
                v[f"{m}_vs_gt4_wheel"] = psnr(rs[y0:y1, x0:x1], gt[y0:y1, x0:x1])
                v[f"{m}_vs_gt4_full"] = psnr(rs, gt)
        out["views"][c["img_name"]] = {k: round(x, 3) for k, x in v.items()}
    V = [v for v in out["views"].values() if "hac_vs_scaffold_wheel" in v]
    mean = lambda k: round(float(np.mean([v[k] for v in V])), 3)
    out["summary"] = dict(n_views_with_wheel=len(V), hac_vs_scaffold_full=round(float(np.mean([v["hac_vs_scaffold_full"] for v in out["views"].values()])), 3),
                          hac_vs_scaffold_wheel=mean("hac_vs_scaffold_wheel"),
                          gt4_wheel=dict(scaffold=mean("scaffold_vs_gt4_wheel"), hac=mean("hac_vs_gt4_wheel"),
                                         delta_hac_minus_scaffold=round(mean("hac_vs_gt4_wheel") - mean("scaffold_vs_gt4_wheel"), 3)),
                          gt4_full=dict(scaffold=mean("scaffold_vs_gt4_full"), hac=mean("hac_vs_gt4_full"),
                                        delta_hac_minus_scaffold=round(mean("hac_vs_gt4_full") - mean("scaffold_vs_gt4_full"), 3)),
                          test_psnr_delta_results_json=round(out["results_json"]["hac"]["PSNR"] - out["results_json"]["scaffold"]["PSNR"], 4))
    pathlib.Path("results/hac/image_side.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=1)); print(json.dumps(out["results_json"]))


if __name__ == "__main__":
    main()
