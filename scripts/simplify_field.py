"""Objective 5 (exploratory): persistence-based simplification of the original density field.

Two exact operators from mathematical morphology, each a persistence simplification in 3D:
  S0(h)  h-maxima transform  g = R^dilation_f(f - h): removes every regional maximum whose dynamic is
         < h, i.e. every H0 feature of the superlevel filtration with persistence < h (26-connectivity,
         same as our cubical complex), and lowers the surviving peaks by at most h.
  S2(h)  h-minima transform  g = R^erosion_f(f + h): fills every basin shallower than h, i.e. (Alexander
         duality in the box) every H2 cavity with persistence < h.
Both change f by at most h in sup norm, so by the stability theorem every diagram moves by <= h.
There is no such operator for H1 alone in 3D, so H1 is what we *observe*: do the spoke-gap loops and
their weakest links survive simplification of the other two dimensions?

Sweep h = k * tau_H1(res), k = 1, 2, 4, 8, 16, applied as S2(S0(f)). Measures per (res, h): sup and mean
change, voxels changed, diagram sizes, features above the cutoff, bottleneck to the original per
dimension (in units of the half-voxel floor eta), air-enclosing loops, and the density ratio g / f at the
original's air-loop birth cells (weakest links).

    python scripts/simplify_field.py            # 256 and 384
Writes results/simplify/{simplify.json, simplify.md, simplify.png}.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from skimage.morphology import reconstruction  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.explain_change import pairs_with_cells  # noqa: E402
from topogs.topology import bottleneck, count_air_loops, count_long, persistence  # noqa: E402

KS = (1, 2, 4, 8, 16)
FOOT = np.ones((3, 3, 3), dtype=bool)          # 26-connectivity for the superlevel sets, as in the cubical complex
# The complement of a 26-connected set is read with the dual 6-connectivity (face neighbours), so cavities
# (bounded basins) are found with a 6-connected footprint.
FOOT6 = np.zeros((3, 3, 3), dtype=bool); FOOT6[1, 1, :] = FOOT6[1, :, 1] = FOOT6[:, 1, 1] = True


def simplify(f, h):
    g = reconstruction(f - h, f, method="dilation", footprint=FOOT)     # S0: H0 features < h
    # S2: H2 cavities < h. The seed is anchored to the field on the box boundary: air that reaches the
    # boundary is not a cavity (H2 = bounded components of the complement) and must stay at its density.
    # Without the anchor the global minimum (open air, ~0) is raised by h too, which makes every spoke gap
    # stop counting as "air" (death density < 0.05) although no loop is removed.
    seed = g + h
    for ax in range(3):
        idx = [slice(None)] * 3
        for end in (0, -1):
            idx[ax] = end
            seed[tuple(idx)] = g[tuple(idx)]
    g = reconstruction(seed, g, method="erosion", footprint=FOOT6)
    return g


def summary(res_dgms, cut, death_max):
    d = res_dgms["dgms"]
    return dict(n=[int(len(d[k])) for k in range(3)], above=[count_long(d[k], cut[f"H{k}"]) for k in range(3)],
                air=count_air_loops(d[1], cut["H1"], death_max))


def main(res_list=(256, 384)):
    an = yaml.safe_load(open("configs/analysis.yaml"))
    out = pathlib.Path("results/simplify"); out.mkdir(parents=True, exist_ok=True)
    js = pathlib.Path(out / "simplify.json")
    allres = json.loads(js.read_text()) if js.exists() else {}
    for res in res_list:
        res = int(res)
        f = np.load(f"results/fields/original/original_r{res}_off0.0.npy").astype(np.float64)
        cut, eta, dmax = an["persistence_cutoff"][res], an["noise_floor"][res], an["air_loop_death_max"]
        # the original's weakest links (birth cells of its air loops), same definition as structure_levels.py
        D, B, _ = pairs_with_cells(f)
        b, d = -D[:, 0], -D[:, 1]
        Bc = B[((b - d) > cut["H1"]) & (d < dmax)]
        f_links = f[tuple(Bc.T)]
        p0 = persistence(f)
        rows = [dict(k=0, h=0.0, **summary(p0, cut, dmax), sup=0.0, mean=0.0, changed=0.0,
                     bottleneck_over_eta=[0.0, 0.0, 0.0], weak_median=1.0, weak_min=1.0, seconds=round(p0["seconds"], 1))]
        print(res, "original", rows[0], flush=True)
        keep = {}
        for k in KS:
            h = k * cut["H1"]
            g = simplify(f, h)
            p = persistence(g)
            dg = np.abs(g - f)
            ratio = g[tuple(Bc.T)] / f_links
            row = dict(k=k, h=round(h, 4), **summary(p, cut, dmax), sup=round(float(dg.max()), 4),
                       mean=round(float(dg.mean()), 5), changed=round(float((dg > 1e-9).mean()), 4),
                       bottleneck_over_eta=[round(bottleneck(p0["dgms"][i], p["dgms"][i]) / eta[f"H{i}"], 3) for i in range(3)],
                       weak_median=round(float(np.median(ratio)), 3), weak_min=round(float(ratio.min()), 3),
                       seconds=round(p["seconds"], 1))
            rows.append(row)
            print(res, row, flush=True)
            if k in (4, 16):
                keep[k] = g
        allres[str(res)] = dict(eta=eta, cutoff=cut, n_links=int(len(Bc)), rows=rows)
        js.write_text(json.dumps(allres, indent=2))
        if res == 256:
            fig_slices(f, keep, Bc, rows, out)

    md = []
    for res, R in allres.items():
        md += [f"\n**{res}³** (cutoff τ(H₁) = {R['cutoff']['H1']:.3f}; {R['n_links']} weakest links)\n",
               "| h (× τ_H1) | h | max \\|f−g\\| | voxels changed | peaks H₀ (all / >τ) | loops H₁ (all / >τ) | cavities H₂ (all / >τ) | "
               "air loops | weak links median (min) | B/η H₀ / H₁ / H₂ |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in R["rows"]:
            md.append(f"| {r['k']} | {r['h']:.3f} | {r['sup']:.3f} | {100 * r['changed']:.1f}% | {r['n'][0]} / {r['above'][0]} | "
                      f"{r['n'][1]} / {r['above'][1]} | {r['n'][2]} / {r['above'][2]} | {r['air']} | "
                      f"{r['weak_median']:.2f} ({r['weak_min']:.2f}) | " + " / ".join(f"{x:.2f}" for x in r["bottleneck_over_eta"]) + " |")
    (out / "simplify.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def fig_slices(f, keep, Bc, rows, out):
    """Wheel max-projection (thin slab through the median link depth) of f and the simplified fields,
    plus the sweep curves."""
    v0 = int(np.median(Bc[:, 1])); sl = slice(max(v0 - 3, 0), v0 + 4)
    fig, axs = plt.subplots(1, 5, figsize=(21, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.1, 1.1]))
    for ax, (ttl, g) in zip(axs[:3], [("original", f), (f"simplified, h = 4τ = {rows[3]['h']:.2f}", keep[4]),
                                      (f"simplified, h = 16τ = {rows[5]['h']:.2f}", keep[16])]):
        ax.imshow(np.clip(g[:, sl, :].max(axis=1).T, 0, 6), origin="lower", cmap="gray", vmin=0, vmax=6)
        ax.set_title(ttl, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    ks = [r["k"] for r in rows]
    xs = [0.5] + ks[1:]                              # h = 0 (original) drawn at the left edge of the log axis
    ax = axs[3]
    ax.plot(xs, [r["n"][0] for r in rows], color="#eb6834", marker="s", lw=2, label="H₀ peaks (all)")
    ax.plot(xs, [max(r["n"][2], 0.8) for r in rows], color="#2a78d6", marker="o", lw=2, label="H₂ cavities (all; 0 drawn at 0.8)")
    ax.plot(xs, [r["air"] for r in rows], color="#3a3a3a", marker="D", lw=2, label="air loops (spoke gaps)")
    ax.set_yscale("log"); ax.set_ylabel("number of features (log)")
    ax.set_title("features kept (256³)", fontsize=10); ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax = axs[4]
    ax.plot(xs, [100 * r["changed"] for r in rows], color="#777777", marker="o", lw=2)
    ax.set_ylim(0, 5); ax.set_ylabel("voxels changed (%)")
    ax.set_title("cost: share of voxels changed (256³)", fontsize=10)
    for ax in axs[3:]:
        ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels(["0"] + [str(k) for k in ks[1:]])
        ax.set_xlabel("h (× τ_H1 = 0.617)"); ax.grid(True, color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Persistence-based simplification of the original field (256³): h-maxima (26-connected) removes peaks, h-minima "
                 "(6-connected, box boundary fixed) fills cavities. All 35 spoke-gap loops and their weakest links are unchanged at every h. "
                 "Images: slab ±3 voxels, clipped at 6", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "simplify.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def replot():
    """Redraw the figure from simplify.json (256^3), recomputing only the two simplified fields shown."""
    R = json.loads(pathlib.Path("results/simplify/simplify.json").read_text())["256"]
    an = yaml.safe_load(open("configs/analysis.yaml"))
    f = np.load("results/fields/original/original_r256_off0.0.npy").astype(np.float64)
    D, B, _ = pairs_with_cells(f); b, d = -D[:, 0], -D[:, 1]
    Bc = B[((b - d) > an["persistence_cutoff"][256]["H1"]) & (d < an["air_loop_death_max"])]
    rows = R["rows"]
    fig_slices(f, {4: simplify(f, rows[3]["h"]), 16: simplify(f, rows[5]["h"])}, Bc, rows, pathlib.Path("results/simplify"))


if __name__ == "__main__":
    if sys.argv[1:] == ["--plot"]:
        replot()
    else:
        main(*(sys.argv[1:] and [sys.argv[1:]]))
