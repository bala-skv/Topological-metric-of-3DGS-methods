"""Figure 1 as three separate, larger images (same data as the 3-panel *_diagram.png).

Reads the saved diagrams results/persistence/original/original_r{res}_off0.0_dgms.npz (GUDHI convention
(-birth density, -death density)) and the cutoff from configs/analysis.yaml. Writes to
results/persistence/original/panels/:
  original_r{res}_diagram_all.png   all three dimensions
  original_r{res}_diagram_H1.png    H1 only, air-enclosing loops (spoke gaps) highlighted
  original_r{res}_barcode.png       the longest bars per dimension

    python scripts/plot_diagram_panels.py [res]      # default 256
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

COL = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
NAME = {0: "H₀ peaks (components)", 1: "H₁ loops", 2: "H₂ cavities"}
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 11})


def style(ax):
    ax.grid(True, color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main(res=256):
    res = int(res)
    an = yaml.safe_load(open("configs/analysis.yaml"))
    cut, dmax = an["persistence_cutoff"][res]["H1"], an["air_loop_death_max"]
    src = pathlib.Path("results/persistence/original")
    z = np.load(src / f"original_r{res}_off0.0_dgms.npz")
    P = {d: -z[f"H{d}"] for d in range(3)}          # (birth density, death density), birth >= death
    out = src / "panels"; out.mkdir(exist_ok=True)

    # 1. all dimensions
    fig, ax = plt.subplots(figsize=(8, 7.5))
    hi = max(p.max() for p in P.values())
    for d in (0, 1, 2):                                   # cavities last so they are not hidden
        ax.scatter(P[d][:, 0], P[d][:, 1], s=10, c=COL[d], alpha=0.55, lw=0, label=f"{NAME[d]} ({len(P[d])})")
    ax.plot([0, hi], [0, hi], "k-", lw=0.8, label="diagonal: birth = death")
    ax.plot([cut, hi], [0, hi - cut], "k--", lw=1, label=f"cutoff: persistence = {cut:.3f}")
    ax.set_xlim(0, hi * 1.03); ax.set_ylim(0, hi * 1.03); ax.set_aspect("equal")
    ax.set_xlabel("birth density (feature appears)"); ax.set_ylabel("death density (feature disappears)")
    ax.set_title(f"Persistence diagram, original field {res}³ (superlevel sets)")
    ax.text(0.55 * hi, 0.80 * hi, "upper-left half empty by construction:\ndeath ≤ birth for every feature", fontsize=11,
            color="#666666", va="top", ha="right")
    ax.legend(loc="upper left", frameon=False); style(ax)
    fig.tight_layout(); fig.savefig(out / f"original_r{res}_diagram_all.png", dpi=200); plt.close(fig)

    # 2. H1 only, air loops highlighted
    p = P[1]; pers = p[:, 0] - p[:, 1]
    air = (pers > cut) & (p[:, 1] < dmax)
    long_ = (pers > cut) & ~air
    short = pers <= cut
    top = p[:, 0].max()
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.scatter(p[short, 0], p[short, 1], s=8, c="#c9c9c9", lw=0, label=f"below cutoff, noise ({short.sum()})")
    ax.scatter(p[long_, 0], p[long_, 1], s=22, c=COL[1], alpha=0.8, lw=0, label=f"above cutoff, other loops ({long_.sum()})")
    ax.scatter(p[air, 0], p[air, 1], s=40, c="#b3471d", marker="D", edgecolors="white", lw=0.8,
               label=f"air-enclosing loops: spoke gaps etc. ({air.sum()})")
    ax.plot([0, top], [0, top], "k-", lw=0.8)
    ax.plot([cut, top], [0, top - cut], "k--", lw=1, label=f"cutoff: persistence = {cut:.3f}")
    ax.set_xlim(0, top * 1.03); ax.set_ylim(-0.4, top * 1.03); ax.set_aspect("equal")
    ax.set_xlabel("birth density = weakest link of the loop"); ax.set_ylabel("death density (hole fills in)")
    ax.set_title(f"H₁ loops only, original {res}³: {int((pers > cut).sum())} of {len(p)} above cutoff")
    ax.annotate("death ≈ 0: the hole is empty air,\nit never fills in", xy=(4.5, 0.1), xytext=(8, 3.5), fontsize=11,
                color="#444444", arrowprops=dict(arrowstyle="->", color="#777777"))
    ax.legend(loc="upper left", frameon=False); style(ax)
    fig.tight_layout(); fig.savefig(out / f"original_r{res}_diagram_H1.png", dpi=200); plt.close(fig)

    # 3. barcode: one panel per dimension, each on its own density range (H1/H2 bars are tiny on the H0 scale)
    n = 20
    fig, axs = plt.subplots(3, 1, figsize=(10, 11), gridspec_kw=dict(hspace=0.55))
    for ax, d in zip(axs, (0, 1, 2)):
        q = P[d]; pers = q[:, 0] - q[:, 1]
        order = np.argsort(-pers)[:n]
        for y, i in enumerate(order):
            ax.plot([q[i, 1], q[i, 0]], [y, y], c=COL[d], lw=4, solid_capstyle="butt")
        ax.axvline(0, color="#999999", lw=0.8)
        ax.invert_yaxis(); ax.set_yticks([]); ax.set_xlim(left=min(-0.02 * q[order, 0].max(), 0))
        ax.set_title(f"{NAME[d]}: {n} longest of {len(q)} (longest persistence {pers.max():.2f})", loc="left")
        ax.grid(True, axis="x", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        ax.set_xlabel("density")
    fig.suptitle(f"Barcode, original {res}³ (each bar runs from death on the left to birth on the right;\n"
                 "note the different density range in each panel)", fontsize=13)
    fig.savefig(out / f"original_r{res}_barcode.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print(out, "air loops", int(air.sum()), "long non-air", int(long_.sum()))


if __name__ == "__main__":
    main(*sys.argv[1:])
