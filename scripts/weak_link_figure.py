"""Before/after figure for the headline case: one of the spokes' weakest links in the original,
opacity pruning 30% and FCGS lambda=16e-4.

Which link: the original's air-enclosing H1 loops each close at a birth cell (the lowest-density
point of the ring around a spoke gap; see scripts/structure_levels.py). We take the loop in the
spoke band whose opacity-30% density ratio at that cell is closest to the median ratio, i.e. a
TYPICAL weakened link, not the worst one.

Panels: (a) whole wheel (thin slab max-projection through the link's depth) with the zoom window,
(b-d) the zoom for the three scenes on one colour scale, (e) density profile along the spoke
through the link. Spoke direction = Hessian eigenvector of the original field with the largest
eigenvalue at the link: across a spoke density falls off fast (two strongly negative eigenvalues),
along it density changes slowly, and at the link (a minimum along the spoke) that eigenvalue is
positive.

    python scripts/weak_link_figure.py [RES]          # default 384
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.ndimage import gaussian_filter, map_coordinates  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.explain_change import pairs_with_cells  # noqa: E402
from topogs.grids import crop_grid  # noqa: E402

SCENES = [("original", "original", "#3a3a3a"),
          ("opac30", "opacity prune 30%", "#1baf7a"),
          ("fcgs_16e-4", "FCGS λ=16e-4", "#2a78d6")]


def field(v, res):
    return np.load(f"results/fields/{v}/{v}_r{res}_off0.0.npy").astype(np.float64)


def spoke_direction(f0, c, sigma=1.0):
    """Unit vector (index space) of the Hessian eigenvector with the largest eigenvalue at cell c."""
    s = gaussian_filter(f0, sigma)
    lo = np.maximum(np.array(c) - 2, 0); w = s[lo[0]:lo[0] + 5, lo[1]:lo[1] + 5, lo[2]:lo[2] + 5]
    g = np.gradient(w); H = np.empty((3, 3))
    for i in range(3):
        gi = np.gradient(g[i])
        for j in range(3):
            H[i, j] = gi[j][tuple(np.array(c) - lo)]
    ev, V = np.linalg.eigh((H + H.T) / 2)
    return V[:, -1], ev


def main(res=384):
    res = int(res)
    an = yaml.safe_load(open("configs/analysis.yaml"))
    g = crop_grid(an["crop_config"], an["box"], res); au, av, aw = g.axes()
    vox = au[1] - au[0]
    F = {v: field(v, res) for v, _, _ in SCENES}
    f0 = F["original"]
    D, B, C = pairs_with_cells(f0)
    b, d = -D[:, 0], -D[:, 1]
    air = ((b - d) > an["persistence_cutoff"][res]["H1"]) & (d < an["air_loop_death_max"])
    Bc = B[air]
    r_link = np.hypot(au[Bc[:, 0]], aw[Bc[:, 2]])
    spoke = (r_link > an["spoke_r"][0]) & (r_link < an["spoke_r"][1])
    b0 = f0[tuple(Bc.T)]
    ratio = F["opac30"][tuple(Bc.T)] / b0
    med = np.median(ratio)
    cand = np.where(spoke)[0]
    k = cand[np.argmin(np.abs(ratio[cand] - med))]
    c = tuple(int(x) for x in Bc[k])
    vals = {v: float(F[v][c]) for v, _, _ in SCENES}
    print(f"{res}^3: {air.sum()} air loops, {spoke.sum()} links in the spoke band; opac30 ratio median {med:.3f}")
    print(f"chosen link cell {c} at u,v,w = ({au[c[0]]:+.3f},{av[c[1]]:+.3f},{aw[c[2]]:+.3f}), r = {r_link[k]:.3f}")
    print("density at link:", {v: round(x, 3) for v, x in vals.items()},
          "ratios:", {v: round(x / vals['original'], 3) for v, x in vals.items()})

    t, ev = spoke_direction(f0, c)
    print("Hessian eigenvalues", ev.round(3), "spoke direction (index space)", t.round(3))
    L = int(round(0.06 / vox))                       # +- 0.06 scene units along the spoke
    s = np.arange(-L, L + 1)
    pts = np.array(c)[:, None] + t[:, None] * s[None, :]
    prof = {v: map_coordinates(F[v], pts, order=1) for v, _, _ in SCENES}

    # thin slab (+-3 voxels in v, about one filter width) through the link
    sl = slice(max(c[1] - 3, 0), c[1] + 4)
    mp = {v: F[v][:, sl, :].max(axis=1) for v, _, _ in SCENES}
    Z = int(round(0.07 / vox))
    zu = slice(max(c[0] - Z, 0), c[0] + Z + 1); zw = slice(max(c[2] - Z, 0), c[2] + Z + 1)
    vmax = 6.0                                       # whole-wheel panel
    zmax = 2.5                                       # zoom panels: spoke range, so the drop is visible

    fig = plt.figure(figsize=(15, 8.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.8], hspace=0.22, wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ext = [au[0], au[-1], aw[0], aw[-1]]
    ax.imshow(np.clip(mp["original"].T, 0, vmax), origin="lower", extent=ext, cmap="gray", vmin=0, vmax=vmax)
    u0, u1, w0, w1 = au[zu][0], au[zu][-1], aw[zw][0], aw[zw][-1]
    ax.add_patch(plt.Rectangle((u0, w0), u1 - u0, w1 - w0, fill=False, ec="#eda100", lw=1.8))
    ax.set_title(f"(a) original, whole wheel (clipped at {vmax:g})\nslab ±3 voxels at the link's depth, {res}³", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    zext = [u0, u1, w0, w1]
    for i, (v, lab, col) in enumerate(SCENES):
        ax = fig.add_subplot(gs[0, i + 1])
        im = ax.imshow(np.clip(mp[v][zu, zw].T, 0, zmax), origin="lower", extent=zext, cmap="gray", vmin=0, vmax=zmax)
        cu, cw = au[c[0]], aw[c[2]]
        du, dw = t[0] * L * vox, t[2] * L * vox
        off = 0.012                                  # draw the profile line beside the spoke, not on it
        nu, nw = -dw / np.hypot(du, dw) * off, du / np.hypot(du, dw) * off
        ax.plot([cu - du + nu, cu + du + nu], [cw - dw + nw, cw + dw + nw], color="#eda100", lw=1.0, ls="--", alpha=0.8)
        ax.scatter([cu], [cw], s=70, facecolors="none", edgecolors="#eda100", linewidths=1.8, zorder=3)
        ax.set_title(f"({'bcd'[i]}) {lab}\ndensity at the link {vals[v]:.2f} ({vals[v] / vals['original']:.2f}×)",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    cax = fig.add_axes([0.915, 0.555, 0.008, 0.3])
    fig.colorbar(im, cax=cax).set_label(f"density (b–d, clipped at {zmax:g})", fontsize=9)

    ax = fig.add_subplot(gs[1, :3])
    x = s * vox
    for v, lab, col in SCENES:
        ax.plot(x, prof[v], color=col, lw=2, label=lab, zorder=3 if v == "opac30" else 2)
    ax.axvline(0, color="#999999", lw=0.8, ls=":")
    ax.text(0.002, ax.get_ylim()[1] * 0.97, "weakest link", fontsize=8.5, color="#555555", va="top")
    ax.set_xlabel("position along the spoke through the link (scene units; 1 voxel = %.4f)" % vox)
    ax.set_ylabel("density")
    ax.set_title("(e) density profile along the spoke through the link (direction marked by the dashed line in b–d)", fontsize=10)
    ax.grid(True, color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc="center right", bbox_to_anchor=(1.0, 0.5))

    ax = fig.add_subplot(gs[1, 3]); ax.axis("off")
    sl_json = json.load(open("results/explain/structure_levels.json")).get(str(res), {}).get("variants", {})
    lines = [f"All {int(air.sum())} weakest links ({res}³),",
             "density ratio to the original:", ""]
    for v, lab, _ in SCENES[1:]:
        r = sl_json.get(v)
        if r:
            lines.append(f"{lab}:\n  median {r['weak_median']:.2f}×, p10 {r['weak_p10']:.2f}×, min {r['weak_min']:.2f}×")
    lines += ["", "Wheel-region image metrics", "(25 test views, ΔPSNR vs original):",
              "  opacity prune 30%  0.50 dB", "  FCGS λ=16e-4      0.49 dB"]
    ax.text(0.0, 0.95, "\n".join(lines), va="top", fontsize=9.5, family="monospace", color="#333333")

    out = pathlib.Path("results/explain"); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"weak_link_r{res}.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    meta = dict(res=res, cell=list(c), uvw=[float(au[c[0]]), float(av[c[1]]), float(aw[c[2]])], r=float(r_link[k]),
                opac30_ratio_median_all_links=float(med), density_at_link=vals,
                hessian_eigenvalues=ev.tolist(), spoke_dir_index=t.tolist())
    (out / f"weak_link_r{res}.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
