"""The disagreement scatter (wheel PSNR drop vs H1 bottleneck / noise floor) as one larger image per
resolution. Same data as results/combined/disagreement_scatter.png; reads results/combined/combined_table.csv
(run scripts/combined_table.py first). Writes results/combined/panels/disagreement_scatter_r{256,384,512}.png.

    python scripts/plot_scatter_panels.py
"""
import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.combined_table import FAMILY, LABEL, NO_LABEL, family  # noqa: E402

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 12})
# label offsets (points) for the crowded low-PSNR cluster, drawn with leader lines
OFFSET = {"fcgs_1e-4": (-40, -34), "fcgs_4e-4": (-60, 2), "fcgs_16e-4": (-50, 34),
          "quant16": (24, -36), "opac30": (30, 28), "rand30": (18, -26)}


def main():
    rows = [r for r in csv.DictReader(open("results/combined/combined_table.csv"))
            if r["res"] in ("256", "384", "512") and r["wheel_d_psnr"]]
    out = pathlib.Path("results/combined/panels"); out.mkdir(exist_ok=True)
    for res in ("256", "384", "512"):
        fig, ax = plt.subplots(figsize=(10, 7))
        for r in (r for r in rows if r["res"] == res):
            x, y = float(r["bottleneck_H1_over_floor"]), float(r["wheel_d_psnr"])
            c, m = FAMILY[family(r["variant"])]
            ax.scatter(x, y, s=150 if m == "^" else 110, c=c, marker=m, edgecolors="white", linewidths=2, zorder=3)
            if r["variant"] in NO_LABEL:
                continue
            off = OFFSET.get(r["variant"], (9, 5))
            lab = LABEL[r["variant"]] + (" (3 seeds)" if r["variant"] == "rand30" else "")
            ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=12, color="#333333",
                        ha="right" if off[0] < 0 else "left", va="center", bbox=dict(fc="white", ec="none", pad=1, alpha=0.85),
                        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8, shrinkB=6) if r["variant"] in OFFSET else None)
        ax.axvline(1.0, color="#777777", lw=1.2, ls="--", zorder=1)
        ax.text(1.04, 0.97, "noise floor", transform=ax.get_xaxis_transform(), fontsize=12, color="#555555")
        ax.set_xscale("log"); ax.set_xlim(0.3, 40); ax.set_ylim(-0.6, 6.8)
        ax.set_xticks([0.3, 0.5, 1, 2, 5, 10, 20]); ax.set_xticklabels(["0.3", "0.5", "1", "2", "5", "10", "20"])
        ax.set_xlabel("H₁ bottleneck distance to original ÷ half-voxel noise floor (log)")
        ax.set_ylabel("wheel-region PSNR drop vs original render (dB)")
        ax.set_title(f"Image error vs topological change, {res}³ grid")
        ax.grid(True, which="major", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        handles = [plt.Line2D([], [], ls="", marker=m, color=c, markersize=10, label=lab)
                   for lab, (c, m) in zip(["FCGS", "quantization 16-bit", "opacity pruning", "random pruning"], FAMILY.values())]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
        fig.text(0.5, -0.01, "quant 8-bit omitted (degenerate, not rendered on the wheel).", ha="center", fontsize=11, color="#555555")
        fig.tight_layout()
        fig.savefig(out / f"disagreement_scatter_r{res}.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
