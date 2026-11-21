"""Inspect a 3DGS scene and produce the material used to choose the crop box.

1. Header check: property names, Gaussian count, raw and activated value ranges.
2. A scene-aligned frame from the cameras:
     centre = least-squares point closest to all camera viewing axes (what they orbit),
     up     = mean camera up vector (COLMAP cameras have +y pointing down in the image),
     u, v   = horizontal axes, u along the main horizontal spread of the object.
3. Scatter plots of Gaussian centres (top / front / side) in that frame, coloured by
   opacity, with candidate crop boxes from the config drawn on top.
4. For each candidate box: number of Gaussians, smallest-scale distribution vs. voxel
   spacing at 128/256/384.

    python scripts/inspect_scene.py configs/crop.yaml
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from topogs.io_ply import load_ply  # noqa: E402


def camera_frame(cameras_json):
    cams = json.loads(pathlib.Path(cameras_json).read_text())
    C = np.array([c["position"] for c in cams])
    Rs = np.array([c["rotation"] for c in cams])  # camera-to-world rotation; columns = camera axes
    d = Rs[:, :, 2]                                # viewing direction (+z camera)
    up = -Rs[:, :, 1].mean(0)                      # COLMAP: +y is down in the image
    up /= np.linalg.norm(up)
    # least-squares point closest to all viewing rays
    A = np.zeros((3, 3)); b = np.zeros(3)
    for c, di in zip(C, d):
        P = np.eye(3) - np.outer(di, di)
        A += P; b += P @ c
    centre = np.linalg.solve(A, b)
    return centre, up, C


def horizontal_axes(up, pts_h):
    """u = main horizontal direction of the given (centred) points, v = up x u."""
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    xy = np.stack([pts_h @ e1, pts_h @ e2], 1)
    w, V = np.linalg.eigh(np.cov(xy.T))
    main = V[:, np.argmax(w)]
    u = main[0] * e1 + main[1] * e2
    v = np.cross(up, u)
    return u, v


def draw_box(ax, box, i, j, color, label):
    c, h = np.array(box["center"]), np.array(box["half"])
    ax.add_patch(plt.Rectangle((c[i] - h[i], c[j] - h[j]), 2 * h[i], 2 * h[j],
                               fill=False, ec=color, lw=1.5))
    ax.text(c[i] - h[i], c[j] + h[j], label, color=color, fontsize=9, va="bottom")


def main(cfg_path):
    cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
    out = pathlib.Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    scene = load_ply(cfg["ply"])

    # ---------------- 1. header check
    raw = scene.raw
    report = {"n_gaussians": scene.n, "properties": list(raw.dtype.names)}
    for k in ["opacity", "scale_0", "rot_0"]:
        report[f"raw_{k}_min_max"] = [float(raw[k].min()), float(raw[k].max())]
    o, s = scene.opacities, scene.scales
    report["opacity_sigmoid_quantiles_1_50_99"] = np.quantile(o, [.01, .5, .99]).round(4).tolist()
    report["scale_exp_quantiles_1_50_99"] = np.quantile(s, [.01, .5, .99]).round(5).tolist()
    qn = np.linalg.norm(np.stack([raw[f"rot_{i}"] for i in range(4)], 1), axis=1)
    report["raw_quat_norm_quantiles_1_50_99"] = np.quantile(qn, [.01, .5, .99]).round(4).tolist()

    # ---------------- 2. frame
    centre, up, C = camera_frame(cfg["cameras"])
    mu = scene.means
    near = np.linalg.norm(mu - centre, axis=1) < cfg.get("object_radius", 1.5)
    sel = near & (o > 0.5)
    u, v = horizontal_axes(up, mu[sel] - centre)
    if "frame" in cfg:  # a frame fixed in the config overrides the automatic one
        centre = np.array(cfg["frame"]["centre"]); R_box = np.array(cfg["frame"]["R_box"])
    else:
        R_box = np.stack([u, v, up])                # rows = box axes in world coords
    report["frame"] = {"centre": centre.round(5).tolist(), "R_box": R_box.round(6).tolist()}
    P = (mu - centre) @ R_box.T                    # box-frame coordinates (u, v, w=up)

    # ---------------- 3. scatter plots
    lim = cfg.get("view_half", 2.0)
    m = np.all(np.abs(P) < lim, axis=1) & (o > cfg.get("plot_min_opacity", 0.3))
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(m)
    if len(idx) > 400_000:
        idx = rng.choice(idx, 400_000, replace=False)
    views = [("top (u-v)", 0, 1), ("front (u-w)", 0, 2), ("side (v-w)", 1, 2)]
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    for ax, (title, i, j) in zip(axs, views):
        # draw far-to-near along the hidden axis so the front-most points end up on top
        k = 3 - i - j
        order = idx[np.argsort(P[idx, k])]
        ax.scatter(P[order, i], P[order, j], s=0.2, c=o[order], cmap="viridis", alpha=0.5, lw=0)
        for b, (name, box) in enumerate(cfg.get("boxes", {}).items()):
            draw_box(ax, box, i, j, colors[b % len(colors)], name)
        ax.set_title(title); ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("uvw"[i]); ax.set_ylabel("uvw"[j]); ax.grid(alpha=0.2)
    fig.suptitle(f"Gaussian centres with opacity > {cfg.get('plot_min_opacity', 0.3)} "
                 f"(scene-aligned frame, units = scene units)")
    fig.tight_layout(); fig.savefig(out / "scatter_views.png", dpi=110); plt.close(fig)

    # ---------------- 3b. front view of only the bike's depth slab (crop-choice figure)
    if "slab_v" in cfg:
        v0, v1 = cfg["slab_v"]
        m = (P[:, 1] > v0) & (P[:, 1] < v1) & (np.abs(P[:, 0]) < 1.4) & (P[:, 2] > -1.2) \
            & (P[:, 2] < 0.35) & (o > cfg.get("plot_min_opacity", 0.02))
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.scatter(P[m, 0], P[m, 2], s=0.3, c=o[m], cmap="viridis", alpha=0.6, lw=0)
        for b, (name, box) in enumerate(cfg.get("boxes", {}).items()):
            draw_box(ax, box, 0, 2, colors[b % len(colors)], name)
        ax.set_aspect("equal"); ax.grid(alpha=0.2); ax.set_xlabel("u"); ax.set_ylabel("w (up)")
        ax.set_title(f"Front view of the depth slab v in [{v0}, {v1}] only, Gaussian centres with "
                     f"opacity > {cfg.get('plot_min_opacity', 0.02)} (dark = low opacity)")
        fig.tight_layout(); fig.savefig(out / "c2_boxes_bike_slab.png", dpi=100); plt.close(fig)

    # ---------------- 4. per-box statistics
    report["boxes"] = {}
    for name, box in cfg.get("boxes", {}).items():
        c, h = np.array(box["center"]), np.array(box["half"])
        inb = np.all(np.abs(P - c) <= h, axis=1)
        smin = s[inb].min(axis=1)
        st = {"n_gaussians": int(inb.sum()), "n_opacity_gt_0.5": int((inb & (o > 0.5)).sum()),
              "smallest_scale_quantiles_10_50_90": np.quantile(smin, [.1, .5, .9]).round(5).tolist()}
        for res in (128, 256, 384):
            hv = 2 * h.max() / res
            st[f"voxel_{res}"] = round(float(hv), 5)
            st[f"frac_thinner_than_voxel_{res}"] = round(float(np.mean(smin < hv)), 3)
        report["boxes"][name] = st

    (out / "inspect_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "properties"}, indent=2))
    print("properties:", len(report["properties"]), report["properties"][:12], "...")


if __name__ == "__main__":
    main(sys.argv[1])
