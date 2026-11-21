"""Render sheets from the saved test-view renders (ada, scripts/render_eval.py, 8-bit PNG).

 1. Render sheet: wheel crop of one test view for the ground truth and every rendered variant, each
    titled with that view's wheel-region PSNR (quant8 was not rendered on the wheel: degenerate).
 2. Headline sheet: ground truth, original, opacity prune 30%, FCGS lambda=16e-4, zoomed on the
    weakest link of results/explain/weak_link_r384.json projected into the view, plus |render -
    original render| x4 so you can see WHERE each method changes the image.

The weak link's world position is box-frame (u,v,w) -> world = centre + uvw @ R_box (rows of R_box are
the box axes in world coordinates), projected with the INRIA camera model (same as render_eval.py).

    python scripts/render_sheet.py [VIEW]        # default _DSC8679
"""
import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

R = pathlib.Path("results/render_wheel_png")
GT = pathlib.Path("data/mipnerf360/bicycle/images_4")
SHEET = ["original", "fcgs_1e-4", "fcgs_4e-4", "fcgs_16e-4", "quant16",
         "opac30", "opac50", "opac70", "rand30", "rand30_s1", "rand30_s2", "rand50", "rand70"]
LABEL = {"original": "original (uncompressed)", "fcgs_1e-4": "FCGS λ=1e-4", "fcgs_4e-4": "FCGS λ=4e-4",
         "fcgs_16e-4": "FCGS λ=16e-4", "quant16": "quant 16-bit", "opac30": "opacity prune 30%",
         "opac50": "opacity prune 50%", "opac70": "opacity prune 70%", "rand30": "random prune 30% (seed 0)",
         "rand30_s1": "random prune 30% (seed 1)", "rand30_s2": "random prune 30% (seed 2)",
         "rand50": "random prune 50%", "rand70": "random prune 70%"}


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def view_row(name, view):
    m = json.load(open(R / f"{name}_metrics.json"))
    return next(r for r in m["per_view"] if r["view"] == view)


def project(world, cam, W, H):
    c2w = np.eye(4); c2w[:3, :3] = np.array(cam["rotation"]); c2w[:3, 3] = np.array(cam["position"])
    w2c = np.linalg.inv(c2w)
    fx = 2 * math.atan(cam["width"] / (2 * cam["fx"])); fy = 2 * math.atan(cam["height"] / (2 * cam["fy"]))
    zn, zf = 0.01, 100.0
    t, r = math.tan(fy / 2) * zn, math.tan(fx / 2) * zn
    P = np.zeros((4, 4)); P[0, 0] = zn / r; P[1, 1] = zn / t; P[3, 2] = 1.0
    P[2, 2] = zf / (zf - zn); P[2, 3] = -(zf * zn) / (zf - zn)
    ph = P @ w2c @ np.append(world, 1.0); ndc = ph[:3] / ph[3]
    return ((ndc[0] + 1) * W - 1) * 0.5, ((ndc[1] + 1) * H - 1) * 0.5


def main(view="_DSC8679"):
    gt = load(GT / f"{view}.JPG")
    H, W = gt.shape[:2]
    x0, x1, y0, y1 = view_row("original", view)["wheel"]["bbox"]
    out = pathlib.Path("results/render_wheel_png")

    # 1. render sheet
    names = ["GT"] + SHEET
    nc = 5; nr = math.ceil(len(names) / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(3.3 * nc, 3.3 * nr * (y1 - y0) / (x1 - x0) + 0.6 * nr))
    for ax in axs.ravel():
        ax.axis("off")
    for ax, n in zip(axs.ravel(), names):
        img = gt if n == "GT" else load(R / f"{n}_{view}.png")
        ax.imshow(img[y0:y1, x0:x1])
        if n == "GT":
            ax.set_title("ground-truth photo", fontsize=9.5)
        else:
            w = view_row(n, view)["wheel"]
            ax.set_title(f"{LABEL[n]}\nwheel PSNR {w['psnr']:.2f} dB", fontsize=9)
    fig.suptitle(f"Test view {view}, wheel crop (same box for all; renders 1237×822, 8-bit)", fontsize=11)
    fig.tight_layout(); fig.savefig(out / f"c6_sheet{view}.png", dpi=110); plt.close(fig)

    # 2. headline zoom at the projected weak link
    wl = json.load(open("results/explain/weak_link_r384.json"))
    box = json.load(open("configs/crop_box.json"))
    world = np.array(box["centre_world"]) + np.array(wl["uvw"]) @ np.array(box["R_box"])
    cams = {c["img_name"]: c for c in json.load(open("data/pretrained/bicycle/cameras.json"))}
    px, py = project(world, cams[view], W, H)
    # projection check: the 3D spoke through the link (+-0.12 units along the Hessian direction, isotropic
    # voxels so index direction = box-frame direction) projected into the photo; it lies on an image spoke
    t = np.array(wl["spoke_dir_index"]); t = t / np.linalg.norm(t)
    seg = np.array([project(world + k * (t @ np.array(box["R_box"])), cams[view], W, H) for k in np.linspace(-0.12, 0.12, 25)])
    print(f"weak link world {world.round(4)} -> pixel ({px:.1f}, {py:.1f}) in {view}")
    Z = 70
    zx0, zx1 = int(max(0, px - Z)), int(min(W, px + Z)); zy0, zy1 = int(max(0, py - Z)), int(min(H, py + Z))
    imgs = {"GT": gt, **{n: load(R / f"{n}_{view}.png") for n in ("original", "opac30", "fcgs_16e-4")}}
    fig, axs = plt.subplots(2, 4, figsize=(13.5, 7.4))
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    for j, (n, ttl) in enumerate([("GT", "ground-truth photo"), ("original", "original render"),
                                  ("opac30", "opacity prune 30%"), ("fcgs_16e-4", "FCGS λ=16e-4")]):
        ax = axs[0, j]; ax.imshow(imgs[n][zy0:zy1, zx0:zx1], interpolation="nearest")
        if n == "GT":
            ax.plot(seg[:, 0] - zx0, seg[:, 1] - zy0, ".", color="#eda100", ms=2.5)
        ax.scatter([px - zx0], [py - zy0], s=160, facecolors="none", edgecolors="#eda100", linewidths=1.8)
        ax.set_title(ttl + (" (dots: the 3D spoke, projected)" if n == "GT" else ""), fontsize=10)
    ax = axs[1, 0]; ax.imshow(gt[y0:y1, x0:x1]); ax.add_patch(plt.Rectangle((zx0 - x0, zy0 - y0), zx1 - zx0, zy1 - zy0,
                                                                         fill=False, ec="#eda100", lw=1.5))
    ax.set_title("where: wheel crop, zoom window", fontsize=10)
    axs[1, 1].axis("off")
    axs[1, 1].text(0.0, 0.9, "Bottom right: |render − original render|\n× 4, zoom window.\n\n"
                   f"Density at this weak link\n(384³, ratio to original):\n  opacity 30%   {wl['density_at_link']['opac30'] / wl['density_at_link']['original']:.2f}×\n"
                   f"  FCGS 16e-4    {wl['density_at_link']['fcgs_16e-4'] / wl['density_at_link']['original']:.2f}×\n\n"
                   "Wheel ΔPSNR (25 views):\n  opacity 30%   0.50 dB\n  FCGS 16e-4    0.49 dB",
                   va="top", fontsize=9.5, family="monospace", color="#333333", transform=axs[1, 1].transAxes)
    for j, n in ((2, "opac30"), (3, "fcgs_16e-4")):
        d = np.clip(np.abs(imgs[n] - imgs["original"]).mean(axis=2) * 4, 0, 1)
        ax = axs[1, j]; ax.imshow(d[zy0:zy1, zx0:zx1], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.scatter([px - zx0], [py - zy0], s=160, facecolors="none", edgecolors="#eda100", linewidths=1.8)
        ax.set_title(f"|{LABEL[n]} − original| × 4", fontsize=10)
    fig.suptitle(f"Weakest link (orange circle) in test view {view}: images vs density", fontsize=11)
    fig.tight_layout(); fig.savefig(out / f"weak_link_render{view}.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main(*sys.argv[1:])
