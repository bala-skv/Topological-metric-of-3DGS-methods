"""The simplification figure (results/simplify/simplify.png) as separate, larger images.

Reads results/simplify/simplify.json (256^3 sweep) and recomputes only the two simplified fields shown.
Writes to results/simplify/panels/:
  wheel_original.png, wheel_h4.png, wheel_h16.png   the three wheel slabs, one image each
  wheels_row.png                                    the three wheels side by side, shared colour bar
  features_kept.png                                 peaks / cavities / air loops vs h
  voxels_changed.png                                share of voxels changed vs h

    python scripts/plot_simplify_panels.py
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.explain_change import pairs_with_cells  # noqa: E402
from scripts.simplify_field import simplify  # noqa: E402

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 12})
VMAX = 6


def style(ax, xs, ks, tau):
    ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels(["0"] + [str(k) for k in ks[1:]])
    ax.set_xlabel(f"simplification level h (multiples of the cutoff τ = {tau:.3f})")
    ax.grid(True, color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    R = json.loads(pathlib.Path("results/simplify/simplify.json").read_text())["256"]
    rows, tau = R["rows"], R["cutoff"]["H1"]
    an = yaml.safe_load(open("configs/analysis.yaml"))
    out = pathlib.Path("results/simplify/panels"); out.mkdir(exist_ok=True)

    # wheels
    f = np.load("results/fields/original/original_r256_off0.0.npy").astype(np.float64)
    D, B, _ = pairs_with_cells(f); b, d = -D[:, 0], -D[:, 1]
    Bc = B[((b - d) > an["persistence_cutoff"][256]["H1"]) & (d < an["air_loop_death_max"])]
    v0 = int(np.median(Bc[:, 1])); sl = slice(max(v0 - 3, 0), v0 + 4)
    fields = [("wheel_original", "Original field", f),
              ("wheel_h4", f"Simplified, h = 4τ = {rows[3]['h']:.2f}", simplify(f, rows[3]["h"])),
              ("wheel_h16", f"Simplified, h = 16τ = {rows[5]['h']:.2f}", simplify(f, rows[5]["h"]))]
    imgs = [(n, t, np.clip(g[:, sl, :].max(axis=1).T, 0, VMAX)) for n, t, g in fields]
    note = "max over a slab of ±3 voxels through the spoke plane, 256³; density clipped at 6"
    for name, title, img in imgs:
        fig, ax = plt.subplots(figsize=(7.5, 7))
        im = ax.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=VMAX)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8, label="density")
        ax.set_xlabel(note, fontsize=10, color="#555555")
        fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.6), layout="constrained")
    for ax, (_, title, img) in zip(axs, imgs):
        im = ax.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=VMAX)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axs, shrink=0.8, label="density", pad=0.01)
    fig.suptitle("Persistence-based simplification of the original field: " + note, fontsize=13)
    fig.savefig(out / "wheels_row.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # graphs
    ks = [r["k"] for r in rows]; xs = [0.5] + ks[1:]          # h = 0 (original) at the left edge of the log axis
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.plot(xs, [r["n"][0] for r in rows], color="#eb6834", marker="s", lw=2.5, ms=9, label="H₀ peaks (all)")
    ax.plot(xs, [max(r["n"][2], 0.8) for r in rows], color="#2a78d6", marker="o", lw=2.5, ms=9,
            label="H₂ cavities (all; 0 drawn at 0.8)")
    ax.plot(xs, [r["air"] for r in rows], color="#3a3a3a", marker="D", lw=2.5, ms=9, label="air loops (spoke gaps)")
    ax.set_yscale("log"); ax.set_ylabel("number of features (log scale)")
    ax.set_title("Features kept after simplification (256³)")
    ax.legend(frameon=False, loc="upper right"); style(ax, xs, ks, tau)
    fig.tight_layout(); fig.savefig(out / "features_kept.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.plot(xs, [100 * r["changed"] for r in rows], color="#777777", marker="o", lw=2.5, ms=9)
    for x, r in zip(xs[1:], rows[1:]):
        ax.annotate(f"{100 * r['changed']:.1f}%", (x, 100 * r["changed"]), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=11, color="#333333")
    ax.set_ylim(0, 5); ax.set_ylabel("voxels changed (%)")
    ax.set_title("Cost: share of voxels changed (256³)"); style(ax, xs, ks, tau)
    fig.tight_layout(); fig.savefig(out / "voxels_changed.png", dpi=200); plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
