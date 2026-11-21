"""HAC step 6: audit HAC (lambda 0.004) against its own uncompressed baseline Scaffold-GS.
All floors, cutoffs and the decision rule come from configs/analysis_hac.yaml, frozen (committed) from
Scaffold-GS alone before this script was first run.

Per resolution:
  - bottleneck and W2 per H0/H1/H2, HAC mean field vs Scaffold-GS mean field, on full diagrams and with
    Scaffold-GS's tau applied to both; ratio to the effective floor (and to each component floor)
  - air-enclosing loops (persistence > tau_H1, death density < 0.05) of Scaffold-GS, HAC and, for
    reference, the original 3DGS model; primary cutoff = Scaffold-GS's tau, secondary = the original's tau.
    Each air loop is placed at its death cell (inside its hole) and counted as inside the wheel if its radius
    from the box centre is in (spoke_r[0]/2, 0.40), as scripts/locate_loops.py does
  - weak links: HAC density / Scaffold-GS density at the birth cells of Scaffold-GS's air loops
  - identity: camera 0 re-exported in a separate process; field max |diff| and diagram distances
  - decision rule: H1 bottleneck (full) / effective floor > 1 at 256^3 AND 384^3

    systemd-run --user --scope -p MemoryMax=14G -p MemorySwapMax=4G .venv/bin/python scripts/hac_audit.py 256 384
        -> results/hac/audit.json, audit_table.md, figures (results for other resolutions are kept)
    --model hacdrop: the not-validated drop-rule sensitivity, written to results/hac/audit_hacdrop.json
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
from topogs.grids import crop_grid  # noqa: E402
from topogs.topology import count_air_loops, distances, load_dgms  # noqa: E402
from topogs.viz import density_panels  # noqa: E402

F = pathlib.Path("results/fields/hac")
DG = pathlib.Path("results/hac/dgms")
OUT = pathlib.Path("results/hac")
MEAS = [f"{m}_H{d}" for m in ("bottleneck", "wasserstein2") for d in range(3)]


def air_cells(pairs, tau, death_max, g, band):
    """Air-loop birth cells and death-cell positions (box frame), from cached pairs_with_cells(field)."""
    D, B, Dc = pairs
    b, d = -D[:, 0], -D[:, 1]
    air = ((b - d) > tau) & (d < death_max)
    au, av, aw = g.axes()
    du, dw = au[Dc[air, 0]], aw[Dc[air, 2]]
    r = np.hypot(du, dw)
    inside = (r > band[0]) & (r < band[1])
    return B[air], dict(n_air=int(air.sum()), n_air_inside_wheel=int(inside.sum()),
                        birth_density_median=round(float(np.median(b[air])), 3) if air.any() else None), (du, dw, inside)


def main(*resolutions, model="hac"):
    sfx = "" if model == "hac" else "_" + model
    an = yaml.safe_load(open("configs/analysis.yaml"))
    ah = yaml.safe_load(open("configs/analysis_hac.yaml"))
    cam0 = yaml.safe_load(open("configs/hac_export.yaml"))["cameras"][0]
    band = (an["spoke_r"][0] * 0.5, 0.40)
    dmax = ah["air_loop_death_max"]
    out = dict(config="configs/analysis_hac.yaml", compared_model=model, res={})
    for res in map(int, resolutions):
        tau = ah["persistence_cutoff"][res]; tau_o = an["persistence_cutoff"][res]
        fl = ah["floors"][res]
        S = load_dgms(DG / f"scaffold_mean_r{res}_off0.0_dgms.npz")
        Hc = load_dgms(DG / f"{model}_mean_r{res}_off0.0_dgms.npz")
        O = load_dgms(f"results/persistence/original/original_r{res}_off0.0_dgms.npz")
        full = distances(S, Hc)
        cut = {}
        for d in range(3):
            cut.update(distances(dict(dgms={d: S["dgms"][d]}), dict(dgms={d: Hc["dgms"][d]}), min_pers=tau[f"H{d}"]))
        ratio = {k: {m: round(x[m] / fl[k]["effective"][m], 3) for m in MEAS} for k, x in (("full", full), ("cut", cut))}
        ratio_hv = {m: round(full[m] / fl["full"]["half_voxel"][m], 3) for m in MEAS}
        ratio_cs = {m: round(full[m] / fl["full"]["camera_spread"][m], 3) for m in MEAS}
        rr = dict(dist_full={k: round(v, 6) for k, v in full.items()}, dist_cut={k: round(v, 6) for k, v in cut.items()},
                  ratio_effective=ratio, ratio_half_voxel_full=ratio_hv, ratio_camera_spread_full=ratio_cs)
        # air loops (counts from diagrams; positions from cell pairs)
        g = crop_grid(an["crop_config"], an["box"], res)
        fS = np.load(F / f"scaffold_mean_r{res}_off0.0.npy"); fH = np.load(F / f"{model}_mean_r{res}_off0.0.npy")
        loops = {}
        for nm, dg in (("scaffold", S), ("hac", Hc), ("original", O)):
            loops[nm] = dict(tau_scaffold=count_air_loops(dg["dgms"][1], tau["H1"], dmax),
                             tau_original=count_air_loops(dg["dgms"][1], tau_o["H1"], dmax))
        pS, pH = pairs_with_cells(fS), pairs_with_cells(fH)          # one persistence run per field
        BcS, locS, ptsS = air_cells(pS, tau["H1"], dmax, g, band)
        _, locH, ptsH = air_cells(pH, tau["H1"], dmax, g, band)
        _, locSo, _ = air_cells(pS, tau_o["H1"], dmax, g, band)
        _, locHo, _ = air_cells(pH, tau_o["H1"], dmax, g, band)
        del pS, pH
        loops["scaffold_located_tau_scaffold"] = locS; loops["hac_located_tau_scaffold"] = locH
        loops["scaffold_located_tau_original"] = locSo; loops["hac_located_tau_original"] = locHo
        assert locS["n_air"] == loops["scaffold"]["tau_scaffold"] and locH["n_air"] == loops["hac"]["tau_scaffold"]
        rr["air_loops"] = loops
        # weak links
        b0 = fS[BcS[:, 0], BcS[:, 1], BcS[:, 2]].astype(np.float64)
        w = fH[BcS[:, 0], BcS[:, 1], BcS[:, 2]] / b0
        rr["weak_links"] = dict(n=int(len(w)), scaffold_density_median=round(float(np.median(b0)), 3),
                                ratio_median=round(float(np.median(w)), 3), ratio_p10=round(float(np.percentile(w, 10)), 3),
                                ratio_min=round(float(w.min()), 3), ratio_max=round(float(w.max()), 3)) if len(w) else dict(n=0)
        # identity (camera 0 re-exported in a separate process)
        idn = {}
        for m in ("scaffold", model):
            if not (F / f"{m}_{cam0}_rep_r{res}_off0.0.npy").exists():
                continue
            fa = np.load(F / f"{m}_{cam0}_r{res}_off0.0.npy"); fb = np.load(F / f"{m}_{cam0}_rep_r{res}_off0.0.npy")
            e = dict(field_max_abs_diff=float(np.abs(fa.astype(np.float64) - fb).max()))
            pa, pb = DG / f"{m}_{cam0}_r{res}_off0.0_dgms.npz", DG / f"{m}_{cam0}_rep_r{res}_off0.0_dgms.npz"
            if pa.exists() and pb.exists():
                e["dist"] = {k: v for k, v in distances(load_dgms(pa), load_dgms(pb)).items()}
            idn["scaffold" if m == "scaffold" else "hac"] = e
        rr["identity_cam0_rep"] = idn
        rr["field_max"] = dict(scaffold=float(fS.max()), hac=float(fH.max()))
        rr["above_effective_floor_H1_bottleneck"] = bool(ratio["full"]["bottleneck_H1"] > 1)
        out["res"][res] = rr
        print(res, json.dumps(rr, indent=1), flush=True)
        if res == 256:
            for nm, f in (("scaffold", fS), (model, fH)):
                density_panels(f, g, OUT / f"{nm}_mean_r{res}_slices.png",
                               title=f"{nm} mean of 8 camera fields, box B, {res}^3")
        # loop positions figure
        au, av, aw = g.axes()
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))
        for ax, (nm, f, (du, dw, ins)) in zip(axs, (("Scaffold-GS", fS, ptsS), (model, fH, ptsH))):
            mp = f.max(axis=1)
            ax.imshow(np.log10(mp.T + 1e-3), origin="lower", extent=[au[0], au[-1], aw[0], aw[-1]], cmap="gray",
                      vmin=-3, vmax=np.log10(mp.max()))
            ax.plot(du[ins], dw[ins], "o", mfc="none", mec="tab:orange", ms=7, mew=1.5)
            ax.plot(du[~ins], dw[~ins], "x", color="tab:cyan", ms=6)
            for rad in band:
                ax.add_patch(plt.Circle((0, 0), rad, fill=False, ls="--", color="w", lw=0.7))
            ax.set_title(f"{nm} {res}^3: air-loop hole centres (tau = Scaffold-GS {tau['H1']:.3g})\n"
                         f"{int(ins.sum())} inside the wheel band, {int((~ins).sum())} outside")
        fig.tight_layout(); fig.savefig(OUT / f"air_loops_r{res}{sfx}.png", dpi=80); plt.close(fig)
    old = json.loads((OUT / f"audit{sfx}.json").read_text()) if (OUT / f"audit{sfx}.json").exists() else dict(res={})
    old["res"].update({str(k): v for k, v in out["res"].items()}); old["config"] = out["config"]; old["compared_model"] = model
    allres = sorted(int(k) for k in old["res"])
    rule = [r for r in ah["decision_rule"]["resolutions"] if str(r) in old["res"]]
    old["decision_rule"] = dict(statement="H1 bottleneck (full) / effective floor > 1 at 256^3 AND 384^3", resolutions=rule,
                                passes=bool(len(rule) == 2 and all(old["res"][str(r)]["above_effective_floor_H1_bottleneck"] for r in rule)))
    (OUT / f"audit{sfx}.json").write_text(json.dumps(old, indent=2))
    # markdown table
    L = ["| res | H1 bottleneck | eff. floor | ratio | H1 W2 | ratio | H0 b ratio | H2 b ratio | air loops S / H / orig (tau_S) | (tau_orig) | weak links median (p10, min) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in allres:
        x = old["res"][str(r)]; fl = ah["floors"][r]["full"]["effective"]; al = x["air_loops"]; wl = x["weak_links"]
        L.append(f"| {r} | {x['dist_full']['bottleneck_H1']:.4g} | {fl['bottleneck_H1']:.4g} | {x['ratio_effective']['full']['bottleneck_H1']:.2f} | "
                 f"{x['dist_full']['wasserstein2_H1']:.4g} | {x['ratio_effective']['full']['wasserstein2_H1']:.2f} | "
                 f"{x['ratio_effective']['full']['bottleneck_H0']:.2f} | {x['ratio_effective']['full']['bottleneck_H2']:.2f} | "
                 f"{al['scaffold']['tau_scaffold']} / {al['hac']['tau_scaffold']} / {al['original']['tau_scaffold']} | "
                 f"{al['scaffold']['tau_original']} / {al['hac']['tau_original']} / {al['original']['tau_original']} | "
                 f"{wl.get('ratio_median')} ({wl.get('ratio_p10')}, {wl.get('ratio_min')}) |")
    (OUT / f"audit_table{sfx}.md").write_text("\n".join(L) + f"\n\nDecision rule: {old['decision_rule']}\n")
    print("\n".join(L)); print(old["decision_rule"])


if __name__ == "__main__":
    args = sys.argv[1:]
    m = "hac"
    if "--model" in args:
        i = args.index("--model"); m = args[i + 1]; del args[i:i + 2]
    main(*args, model=m)
