"""Combined table: variant x {size, image metrics (full frame, wheel box, spoke pixels), topological
distances (bottleneck / W2 per dimension) and their ratio to the half-voxel noise floor at
128/256/384/512, air-loop counts, and weakest-link density ratios}. Writes
results/combined/combined_table.{csv,md} and the disagreement scatter (wheel PSNR drop vs H1
bottleneck / noise floor at 256^3, 384^3 and 512^3).

Sources: results/variants (128/256/384), results/variants_seeds (extra random seeds, 256/384),
results/variants_512 (512, all), results/render_wheel_png/*_metrics.json (full + wheel metrics;
quant8 full-frame only, from results/render), results/explain/spoke_mask_all_views.json (spoke-pixel
metric, 25 views), results/explain/structure_levels.json (weakest links).

    python scripts/combined_table.py
"""
import csv
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORIG_BYTES = 1520726124
RES = (128, 256, 384, 512)
VARIANTS = ["fcgs_1e-4", "fcgs_4e-4", "fcgs_16e-4", "quant16", "quant8",
            "opac30", "opac50", "opac70", "rand30", "rand30_s1", "rand30_s2", "rand50", "rand70"]
LABEL = {"fcgs_1e-4": "FCGS λ=1e-4", "fcgs_4e-4": "FCGS λ=4e-4", "fcgs_16e-4": "FCGS λ=16e-4",
         "quant16": "quant 16-bit", "quant8": "quant 8-bit (degenerate)",
         "opac30": "opacity prune 30%", "opac50": "opacity prune 50%", "opac70": "opacity prune 70%",
         "rand30": "random prune 30%", "rand30_s1": "random prune 30% (seed 1)",
         "rand30_s2": "random prune 30% (seed 2)", "rand50": "random prune 50%", "rand70": "random prune 70%"}
# family -> (validated categorical slot, marker) ; dataviz palette slots 1-4, light mode
FAMILY = {"fcgs": ("#2a78d6", "o"), "quant": ("#eb6834", "s"), "opac": ("#1baf7a", "^"), "rand": ("#eda100", "D")}
# label offsets (points) for the crowded cluster, drawn with leader lines
OFFSET = {"fcgs_1e-4": (-14, 26), "fcgs_4e-4": (-10, 52), "fcgs_16e-4": (-24, 72),
          "quant16": (-6, 40), "opac30": (16, 18), "rand30": (14, -16)}
NO_LABEL = {"rand30_s1", "rand30_s2"}           # plotted, but labelled once via rand30 ("3 seeds")


def family(n):
    return next(k for k in FAMILY if n.startswith(k))


def jload(p):
    p = pathlib.Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def topo_record(n):
    rec = jload(f"results/variants/{n}/{n}.json") or jload(f"results/variants_seeds/{n}/{n}.json")
    r512 = jload(f"results/variants_512/{n}/{n}.json")
    res = {int(k): v for k, v in rec["res"].items()}
    if r512:
        res[512] = r512["res"]["512"]
    return rec, res


def size_info(rec):
    if rec.get("bitstream_bytes"):
        return rec["bitstream_bytes"], "bitstream (entropy-coded)"
    nb = rec["variant_info"].get("nominal_payload_bits_per_gaussian")
    if nb:
        return "", f"nominal {nb} bits/Gaussian for xyz+scales (de-quantized .ply = original size)"
    return rec["ply_bytes"], ".ply, not entropy-coded"


def main():
    out = pathlib.Path("results/combined"); out.mkdir(parents=True, exist_ok=True)
    o = jload("results/render_wheel_png/original_metrics.json")
    spokes = (jload("results/explain/spoke_mask_all_views.json") or {}).get("summary", {})
    weak = jload("results/explain/structure_levels.json") or {}
    rows = []
    for n in VARIANTS:
        rec, res = topo_record(n)
        img = jload(f"results/render_wheel_png/{n}_metrics.json") or jload(f"results/render/{n}_metrics.json")
        size, size_kind = size_info(rec)
        base = dict(variant=n, label=LABEL[n], n_gaussians=rec["n_gaussians"], n_in_box=rec["zones"]["n_in_box"],
                    size_bytes=size, ratio=round(ORIG_BYTES / size, 1) if size else "", size_kind=size_kind,
                    psnr=round(img["psnr"], 3), ssim=round(img["ssim"], 4), lpips=round(img["lpips"], 4),
                    d_psnr=round(o["psnr"] - img["psnr"], 3), d_ssim=round(o["ssim"] - img["ssim"], 4),
                    d_lpips=round(img["lpips"] - o["lpips"], 4))
        if "wheel" in img:
            w = img["wheel"]
            base.update(wheel_d_psnr=round(o["wheel"]["psnr"] - w["psnr"], 3), wheel_d_ssim=round(o["wheel"]["ssim"] - w["ssim"], 4),
                        wheel_d_lpips=round(w["lpips"] - o["wheel"]["lpips"], 4))
        if n in spokes:
            base.update(spoke_d_psnr=spokes[n]["d_psnr_spokes"], spoke_err_share=spokes[n]["err_share_on_spokes"])
        for res_k, d in sorted(res.items()):
            row = dict(base, res=res_k, H1_above_cutoff=d["H1_above_cutoff"], H1_air_loops=d["H1_air_loops"])
            wl = weak.get(str(res_k), {}).get("variants", {}).get(n)
            if wl:
                row.update(weak_link_median=wl["weak_median"], typical_spoke_median=wl["typical_median"])
            for k, v in d["dist_full"].items():
                row[k] = v
            for k, v in d["dist_over_noise_floor"].items():
                row[f"{k}_over_floor"] = v
            rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "combined_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    def cell(rr, k, res=RES, fmt="{:.2f}"):
        return " / ".join(fmt.format(rr[x][k]) if x in rr and isinstance(rr[x].get(k), float)
                          else (str(rr[x][k]) if x in rr and k in rr[x] else "–") for x in res)

    md = [f"Original render (25 test views): full frame PSNR {o['psnr']:.2f} / SSIM {o['ssim']:.3f} / LPIPS {o['lpips']:.3f}; "
          f"wheel box PSNR {o['wheel']['psnr']:.2f}. Original air loops 31 / 35 / 37 / 37 at 128 / 256 / 384 / 512. "
          "Spoke ΔPSNR: PSNR vs ground truth on spoke pixels only (projected spoke-zone Gaussians, 10.7% of the wheel box). "
          "Weak links: median density ratio variant ÷ original at the original's 37 air-loop birth cells.\n",
          "| variant | size | ΔPSNR full | ΔPSNR wheel | ΔPSNR spokes | B(H₁)/η 128/256/384/512 | W2(H₁)/η 128/256/384/512 | air loops 128/256/384/512 | weak links 256/384/512 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n in VARIANTS:
        rr = {r["res"]: r for r in rows if r["variant"] == n}
        r = next(iter(rr.values()))
        size = f"{r['size_bytes'] / 1e6:.1f} MB ({r['ratio']}×)" if r["size_bytes"] else r["size_kind"].split(" (")[0]
        f = lambda k: f"{r[k]:.2f}" if k in r else "–"
        md.append(f"| {LABEL[n]} | {size} | {r['d_psnr']:.2f} | {f('wheel_d_psnr')} | {f('spoke_d_psnr')} | "
                  f"{cell(rr, 'bottleneck_H1_over_floor')} | {cell(rr, 'wasserstein2_H1_over_floor')} | "
                  f"{cell(rr, 'H1_air_loops')} | {cell(rr, 'weak_link_median', res=(256, 384, 512))} |")
    (out / "combined_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # disagreement scatter: wheel-box PSNR drop vs H1 bottleneck / noise floor, one panel per finer resolution
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.4), sharey=True)
    for ax, res in zip(axs, (256, 384, 512)):
        for r in rows:
            if r["res"] != res or "wheel_d_psnr" not in r:
                continue
            c, m = FAMILY[family(r["variant"])]
            ax.scatter(r["bottleneck_H1_over_floor"], r["wheel_d_psnr"], s=95 if m == "^" else 70, c=c, marker=m,
                       edgecolors="white", linewidths=2, zorder=3)
            if r["variant"] in NO_LABEL:
                continue
            off = OFFSET.get(r["variant"], (6, 4))
            lab = LABEL[r["variant"]] + (" (3 seeds)" if r["variant"] == "rand30" else "")
            ax.annotate(lab, (r["bottleneck_H1_over_floor"], r["wheel_d_psnr"]), xytext=off, textcoords="offset points",
                        fontsize=8.5, color="#333333", ha="right" if off[0] < 0 else "left",
                        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6) if r["variant"] in OFFSET else None)
        ax.axvline(1.0, color="#777777", lw=1, ls="--", zorder=1)
        ax.text(1.03, 0.96, "noise floor", transform=ax.get_xaxis_transform(), fontsize=8.5, color="#555555")
        ax.set_xscale("log"); ax.set_xlim(0.3, 40)
        ax.set_xticks([0.3, 0.5, 1, 2, 5, 10, 20]); ax.set_xticklabels(["0.3", "0.5", "1", "2", "5", "10", "20"])
        ax.set_xlabel("H₁ bottleneck distance to original ÷ half-voxel noise floor (log)")
        ax.set_title(f"{res}³ grid", fontsize=11)
        ax.grid(True, which="major", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel("wheel-region PSNR drop vs original render (dB)")
    handles = [plt.Line2D([], [], ls="", marker=m, color=c, markersize=8, label=lab)
               for lab, (c, m) in zip(["FCGS", "quantization 16-bit", "opacity pruning", "random pruning"], FAMILY.values())]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9.5)
    fig.text(0.5, -0.02, "quant 8-bit omitted (degenerate, not rendered on the wheel). The noise floor shrinks with resolution, "
             "so every ratio grows; the ranking does not change. Same PSNR drop, very different topological change.",
             ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "disagreement_scatter.png", dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
