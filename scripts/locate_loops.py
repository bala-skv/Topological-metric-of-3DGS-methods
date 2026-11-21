"""Where are the long H1 loops? For each H1 pair above the frozen cutoff, GUDHI gives the voxel
whose arrival completes the loop (birth cell) and the voxel whose arrival fills its hole
(death cell). For an "air-enclosing" loop the death cell sits inside the empty hole, so it marks
WHICH gap the loop encloses. Positions are mapped to the box frame and drawn on the front
max-projection; each loop is also classified by its distance from the box centre (= hub).

Index convention checked on a synthetic ring: cofaces_of_persistence_pairs() returns indices
into the Fortran-order flattening of top_dimensional_cells.

    python scripts/locate_loops.py NAME FIELD.npy RES OUT_DIR
"""
import json
import pathlib
import sys

import gudhi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from topogs.grids import crop_grid  # noqa: E402

AN = "configs/analysis.yaml"


def loops(field, tau, death_max):
    cc = gudhi.CubicalComplex(top_dimensional_cells=-np.asarray(field, dtype=np.float64))
    cc.persistence(homology_coeff_field=2)
    reg, _ = cc.cofaces_of_persistence_pairs()
    pairs = reg[1] if len(reg) > 1 else np.zeros((0, 2), int)
    flat = np.asarray(field, dtype=np.float64).ravel(order="F")
    b_den, d_den = flat[pairs[:, 0]], flat[pairs[:, 1]]          # birth density > death density
    keep = (b_den - d_den) > tau
    out = []
    for (bc, dc), bd, dd in zip(pairs[keep], b_den[keep], d_den[keep]):
        out.append(dict(birth_cell=np.unravel_index(bc, field.shape, order="F"),
                        death_cell=np.unravel_index(dc, field.shape, order="F"),
                        birth_density=float(bd), death_density=float(dd), air=bool(dd < death_max)))
    return out


def main(name, field_path, res, out_dir):
    res = int(res)
    an = yaml.safe_load(open(AN))
    field = np.load(field_path)
    g = crop_grid(an["crop_config"], an["box"], res)
    ax_u, ax_v, ax_w = g.axes()
    tau = an["persistence_cutoff"][res]["H1"]
    L = loops(field, tau, an["air_loop_death_max"])
    rows = []
    for l in L:
        i, j, k = l["death_cell"]
        u, v, w = float(ax_u[i]), float(ax_v[j]), float(ax_w[k])
        rows.append(dict(u=round(u, 4), v=round(v, 4), w=round(w, 4), r=round(float(np.hypot(u, w)), 4),
                         angle_deg=round(float(np.degrees(np.arctan2(w, u))), 1),
                         birth_density=round(l["birth_density"], 4), death_density=round(l["death_density"], 5),
                         persistence=round(l["birth_density"] - l["death_density"], 4), air=l["air"]))
    air = [r for r in rows if r["air"]]
    lo, hi = an["spoke_r"][0] * 0.5, 0.40        # hub radius ~0.04 .. rim inner radius ~0.38-0.40
    inside = [r for r in air if lo < r["r"] < hi]
    summary = dict(name=name, res=res, tau_H1=tau, n_long=len(rows), n_air=len(air),
                   n_air_inside_wheel=len(inside), n_air_outside_wheel=len(air) - len(inside),
                   wheel_band=[lo, hi], loops=rows)
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}_r{res}_loops.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "loops"}))

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    mp = field.max(axis=1)
    for ax in axs:
        ax.imshow(np.log10(mp.T + 1e-3), origin="lower", extent=[ax_u[0], ax_u[-1], ax_w[0], ax_w[-1]],
                  cmap="gray", vmin=-3, vmax=np.log10(mp.max()))
        ax.set_aspect("equal"); ax.set_xlabel("u (box frame)"); ax.set_ylabel("w")
    a = axs[0]
    for r in rows:
        a.plot(r["u"], r["w"], "o" if r["air"] else "x", ms=7 if r["air"] else 5,
               mfc="none" if r["air"] else None, mec="tab:orange" if r["air"] else None,
               color="tab:orange" if r["air"] else "tab:cyan", mew=1.5)
    for rad in (lo, hi):
        a.add_patch(plt.Circle((0, 0), rad, fill=False, ls="--", color="w", lw=0.7))
    a.set_title(f"{name} {res}^3: hole centres of long H1 loops\n"
                f"orange o = air-enclosing ({len(air)}; {len(inside)} inside the wheel band), cyan x = other ({len(rows) - len(air)})")
    a = axs[1]
    if air:
        sc = a.scatter([r["u"] for r in air], [r["w"] for r in air], c=[r["persistence"] for r in air],
                       cmap="viridis", s=40)
        fig.colorbar(sc, ax=a, shrink=0.7, label="persistence (density)")
    a.set_title("air-enclosing loops coloured by persistence")
    fig.tight_layout(); fig.savefig(out / f"{name}_r{res}_loops.png", dpi=90); plt.close(fig)


if __name__ == "__main__":
    main(*sys.argv[1:5])
