"""Find the cause: WHERE does a variant's topology differ from the original's?

For homology dimension 1 (loops), compute both cubical complexes with cell locations, find the
optimal 2-Wasserstein matching (Hera, internal L2) between the diagrams, and rank features by
their matching cost: a feature matched to the diagonal was lost (original side) or created
(variant side); a matched pair that moved far changed its birth/death densities. Each costly
feature is placed at its death cell (for an air loop: inside the hole) and its birth cell, and
classified by region of the wheel box. Output: JSON + figure on the original's max-projection.

    python scripts/explain_change.py VARIANT RES [--top 25]
"""
import argparse
import json
import pathlib
import sys

import gudhi
import gudhi.hera
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from topogs.grids import crop_grid  # noqa: E402

AN = "configs/analysis.yaml"


def pairs_with_cells(field, dim=1):
    cc = gudhi.CubicalComplex(top_dimensional_cells=-np.asarray(field, dtype=np.float64))
    cc.persistence(homology_coeff_field=2)
    reg, _ = cc.cofaces_of_persistence_pairs()
    p = reg[dim] if len(reg) > dim else np.zeros((0, 2), int)
    flat = np.asarray(field, dtype=np.float64).ravel(order="F")
    b, d = flat[p[:, 0]], flat[p[:, 1]]
    dg = np.stack([-b, -d], 1)                         # GUDHI convention (sublevel of -density)
    bcell = np.stack(np.unravel_index(p[:, 0], field.shape, order="F"), 1)
    dcell = np.stack(np.unravel_index(p[:, 1], field.shape, order="F"), 1)
    return dg, bcell, dcell


def region(u, w, r):
    if w < -0.38:
        return "ground/bottom"
    if r < 0.05:
        return "hub"
    if r < 0.36:
        return "spoke band"
    if r < 0.49:
        return "rim/tire"
    return "outside wheel (frame, corners)"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("variant"); ap.add_argument("res", type=int)
    ap.add_argument("--top", type=int, default=25); a = ap.parse_args()
    an = yaml.safe_load(open(AN)); res = a.res
    f0 = np.load(f"results/fields/original/original_r{res}_off0.0.npy")
    f1 = np.load(f"results/fields/{a.variant}/{a.variant}_r{res}_off0.0.npy")
    g = crop_grid(an["crop_config"], an["box"], res); ax_u, ax_v, ax_w = g.axes()
    eta = an["noise_floor"][res]["H1"]
    D0, B0, C0 = pairs_with_cells(f0); D1, B1, C1 = pairs_with_cells(f1)
    w2, match = gudhi.hera.wasserstein_distance(D0, D1, order=2, internal_p=2, delta=0.01, matching=True)
    feats = []
    for i, j in np.asarray(match):
        if i >= 0 and j >= 0:
            cost = float(np.linalg.norm(D0[i] - D1[j])); kind = "moved"; src, k = 0, i
        elif i >= 0:
            cost = float((D0[i, 1] - D0[i, 0]) / np.sqrt(2)); kind = "lost"; src, k = 0, i
        else:
            cost = float((D1[j, 1] - D1[j, 0]) / np.sqrt(2)); kind = "new"; src, k = 1, j
        D, C, B = (D0, C0, B0) if src == 0 else (D1, C1, B1)
        ci = C[k]; u, w = float(ax_u[ci[0]]), float(ax_w[ci[2]]); r = float(np.hypot(u, w))
        bi = B[k]
        feats.append(dict(kind=kind, cost=round(cost, 4), side="original" if src == 0 else "variant",
                          birth_density=round(float(-D[k, 0]), 4), death_density=round(float(-D[k, 1]), 5),
                          persistence=round(float(D[k, 1] - D[k, 0]), 4),
                          partner=(dict(birth_density=round(float(-D1[j, 0]), 4), death_density=round(float(-D1[j, 1]), 5))
                                   if kind == "moved" else None),
                          death_uvw=[round(u, 4), round(float(ax_v[ci[1]]), 4), round(w, 4)],
                          birth_uvw=[round(float(ax_u[bi[0]]), 4), round(float(ax_v[bi[1]]), 4), round(float(ax_w[bi[2]]), 4)],
                          r=round(r, 4), region=region(u, w, r)))
    feats.sort(key=lambda x: -x["cost"])
    tot2 = sum(x["cost"] ** 2 for x in feats)
    big = [x for x in feats if x["cost"] > eta]            # each alone exceeds the noise floor
    by_region = {}
    for x in feats:
        by_region.setdefault(x["region"], 0.0); by_region[x["region"]] += x["cost"] ** 2
    summary = dict(variant=a.variant, res=res, w2_H1=round(w2, 4), eta_H1=eta,
                   n_features_cost_above_eta=len(big),
                   kinds_above_eta={k: sum(1 for x in big if x["kind"] == k) for k in ("lost", "new", "moved")},
                   regions_above_eta={rg: sum(1 for x in big if x["region"] == rg) for rg in sorted({x["region"] for x in big})},
                   share_of_W2squared_by_region={k: round(v / tot2, 3) for k, v in sorted(by_region.items(), key=lambda t: -t[1])},
                   top=feats[:a.top])
    out = pathlib.Path("results/explain"); out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.variant}_r{res}_H1.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "top"}, indent=1))
    for x in feats[:12]:
        print(f"  {x['kind']:5s} cost {x['cost']:.3f} pers {x['persistence']:.3f} birth {x['birth_density']:.3f} "
              f"death {x['death_density']:.4f} {x['region']:14s} at u,w=({x['death_uvw'][0]:+.3f},{x['death_uvw'][2]:+.3f})"
              + (f" -> partner b {x['partner']['birth_density']:.3f} d {x['partner']['death_density']:.4f}" if x['partner'] else ""))

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    for ax, f, ttl in zip(axs, (f0, f1), ("original", a.variant)):
        mp = f.max(axis=1)
        ax.imshow(np.log10(mp.T + 1e-3), origin="lower", extent=[ax_u[0], ax_u[-1], ax_w[0], ax_w[-1]],
                  cmap="gray", vmin=-3, vmax=np.log10(f0.max()))
        ax.set_aspect("equal"); ax.set_xlabel("u (box frame)"); ax.set_ylabel("w")
        ax.set_title(f"{ttl} {res}³ (max projection)")
    col = {"lost": "#e34948", "new": "#2a78d6", "moved": "#eda100"}
    mk = {"lost": "v", "new": "^", "moved": "o"}
    for x in big:
        axs[0].scatter(x["death_uvw"][0], x["death_uvw"][2], s=30 + 120 * x["cost"] / max(big[0]["cost"], 1e-9),
                       c=col[x["kind"]], marker=mk[x["kind"]], edgecolors="white", linewidths=1.2, zorder=3)
    h = [plt.Line2D([], [], ls="", marker=mk[k], color=col[k], markersize=8, label=f"{k} ({summary['kinds_above_eta'][k]})")
         for k in ("lost", "new", "moved")]
    axs[0].legend(handles=h, loc="upper right", fontsize=8.5, title=f"H₁ features with cost > η={eta:.3g}", title_fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / f"{a.variant}_r{res}_H1.png", dpi=90); plt.close(fig)


if __name__ == "__main__":
    main()
