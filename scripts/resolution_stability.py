"""Resolution stability: how the audit changes from 128^3 to 512^3.

Reads results/combined/combined_table.csv (run scripts/combined_table.py first) and the frozen noise
floors in configs/analysis.yaml. Writes results/combined/resolution_stability.{png,md,json}:

 (a) H1 bottleneck distance to the original (absolute) and the half-voxel noise floor eta(H1)
 (b) the same distance divided by eta: every ratio grows because eta shrinks, not the distance
 (c) air-enclosing H1 loops (the original's spoke gaps) per variant
 (d) weakest-link density ratio (variant / original at the original's air-loop birth cells)
 + Spearman rank correlation of the variants' H1 ratios between every pair of resolutions.

    python scripts/resolution_stability.py
"""
import csv
import itertools
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

RES = (128, 256, 384, 512)
# shown in the figure: family colour, strength by line style (solid = the headline pair's members)
SHOW = [("fcgs_1e-4", "FCGS λ=1e-4", "#2a78d6", "o", "--"),
        ("fcgs_16e-4", "FCGS λ=16e-4", "#2a78d6", "o", "-"),
        ("quant16", "quant 16-bit", "#eb6834", "s", "-"),
        ("opac30", "opacity 30%", "#1baf7a", "^", "-"),
        ("opac70", "opacity 70%", "#1baf7a", "^", "--"),
        ("rand30", "random 30%", "#eda100", "D", "-"),
        ("rand70", "random 70%", "#eda100", "D", "--")]
SEEDS = ("rand30_s1", "rand30_s2")
ORIG_AIR = {128: 31, 256: 35, 384: 37, 512: 37}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    rows = list(csv.DictReader(open("results/combined/combined_table.csv")))
    T = {}
    for r in rows:
        T.setdefault(r["variant"], {})[int(r["res"])] = r
    eta = {int(k): v["H1"] for k, v in yaml.safe_load(open("configs/analysis.yaml"))["noise_floor"].items()}

    def series(v, key, res=RES):
        return [fnum(T[v][x].get(key)) if x in T[v] else None for x in res]

    # Spearman of B(H1)/eta over all variants present at both resolutions
    variants = sorted(T)
    sp = {}
    for a, b in itertools.combinations(RES, 2):
        va = [v for v in variants if a in T[v] and b in T[v]]
        rho = spearmanr([fnum(T[v][a]["bottleneck_H1_over_floor"]) for v in va],
                        [fnum(T[v][b]["bottleneck_H1_over_floor"]) for v in va]).correlation
        sp[f"{a}-{b}"] = dict(rho=round(float(rho), 3), n=len(va))

    fig, axs = plt.subplots(2, 2, figsize=(13, 9.2))
    kw = dict(lw=2, ms=7, markeredgecolor="white", markeredgewidth=1.5)

    def finish(ax, title, ylabel, logy=False):
        ax.set_xticks(RES); ax.set_xticklabels([f"{x}³" for x in RES]); ax.set_xlabel("grid resolution")
        ax.set_title(title, fontsize=10.5, loc="left"); ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # (a) absolute distance + eta
    ax = axs[0, 0]
    for v, lab, c, m, ls in SHOW:
        ax.plot(RES, series(v, "bottleneck_H1"), color=c, marker=m, ls=ls, label=lab, **kw)
    for v in SEEDS:
        ax.plot(RES[1:], series(v, "bottleneck_H1", RES[1:]), color="#eda100", lw=0.8, alpha=0.6)
    ax.plot(RES, [eta[x] for x in RES], color="#3a3a3a", lw=2, ls=":", marker="x", label="noise floor η(H₁)")
    finish(ax, "(a) H₁ bottleneck distance to the original (density units)", "distance", logy=True)

    # (b) ratio
    ax = axs[0, 1]
    for v, lab, c, m, ls in SHOW:
        ax.plot(RES, series(v, "bottleneck_H1_over_floor"), color=c, marker=m, ls=ls, **kw)
    for v in SEEDS:
        ax.plot(RES[1:], series(v, "bottleneck_H1_over_floor", RES[1:]), color="#eda100", lw=0.8, alpha=0.6)
    ax.axhline(1, color="#777777", ls="--", lw=1)
    ax.text(RES[0], 1.08, "noise floor", fontsize=8.5, color="#555555")
    ax.axvspan(200, 440, color="#f2f2f2", zorder=0)
    ax.text(320, 0.97, "pre-registered rule:\nabove floor at 256³ and 384³", ha="center", va="top", fontsize=8.5,
            color="#555555", transform=ax.get_xaxis_transform())
    finish(ax, "(b) the same distance ÷ η: ratios grow because η shrinks", "distance ÷ noise floor", logy=True)

    # (c) air loops
    ax = axs[1, 0]
    ax.plot(RES, [ORIG_AIR[x] for x in RES], color="#3a3a3a", lw=2.5, marker="o", label="original", **{k: kw[k] for k in ("ms",)})
    for v, lab, c, m, ls in SHOW:
        ax.plot(RES, series(v, "H1_air_loops"), color=c, marker=m, ls=ls, **kw)
    finish(ax, "(c) air-enclosing H₁ loops (spoke gaps) above the cutoff", "count")

    # (d) weak links
    ax = axs[1, 1]
    wr = (256, 384, 512)
    for v, lab, c, m, ls in SHOW:
        ax.plot(wr, series(v, "weak_link_median", wr), color=c, marker=m, ls=ls, **kw)
    for v in SEEDS:
        ax.plot(wr, series(v, "weak_link_median", wr), color="#eda100", lw=0.8, alpha=0.6)
    ax.axhline(1, color="#777777", ls="--", lw=1)
    ax.set_ylim(0, 1.1); ax.set_xlim(220, 548)
    finish(ax, "(d) weakest links: median density ratio variant ÷ original", "density ratio")
    ax.set_xticks(wr); ax.set_xticklabels([f"{x}³" for x in wr])

    h, l = axs[0, 0].get_legend_handles_labels()
    h.insert(0, plt.Line2D([], [], color="#3a3a3a", lw=2.5, marker="o", label="original (c)")); l.insert(0, "original (panel c)")
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=9.5)
    sp_txt = ", ".join(f"{k.replace('-', '/')}: {d['rho']:.2f}" for k, d in sp.items())
    fig.text(0.5, -0.005, f"Thin gold lines: random 30% seeds 1 and 2. Quant 8-bit (degenerate) omitted. Ranking of all variants by "
             f"H₁ distance ÷ floor, Spearman ρ between resolutions: {sp_txt}.", ha="center", fontsize=8.5, color="#555555", wrap=True)
    fig.tight_layout(rect=(0, 0.01, 1, 0.94))
    out = pathlib.Path("results/combined")
    fig.savefig(out / "resolution_stability.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    md = ["| variant | B(H₁) abs 128/256/384/512 | B(H₁)/η | air loops | weak links 256/384/512 |", "|---|---|---|---|---|",
          "| noise floor η(H₁) | " + " / ".join(f"{eta[x]:.3f}" for x in RES) + " | 1 | original " +
          " / ".join(str(ORIG_AIR[x]) for x in RES) + " | 1 |"]
    fmt = lambda xs, f="{:.3f}": " / ".join("–" if x is None else f.format(x) for x in xs)  # noqa: E731
    for v in [s[0] for s in SHOW] + list(SEEDS) + ["fcgs_4e-4", "opac50", "rand50", "quant8"]:
        md.append(f"| {v} | {fmt(series(v, 'bottleneck_H1'))} | {fmt(series(v, 'bottleneck_H1_over_floor'), '{:.2f}')} | "
                  f"{fmt(series(v, 'H1_air_loops'), '{:.0f}')} | {fmt(series(v, 'weak_link_median', wr), '{:.2f}')} |")
    md += ["", "Spearman ρ of B(H₁)/η across all variants: " + sp_txt]
    (out / "resolution_stability.md").write_text("\n".join(md) + "\n")
    (out / "resolution_stability.json").write_text(json.dumps(dict(spearman=sp, eta_H1=eta), indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
