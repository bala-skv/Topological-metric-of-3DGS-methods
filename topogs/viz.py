"""Figures: density slices / projections, persistence diagrams, barcodes."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _show(ax, img, extent, title, vmax):
    im = ax.imshow(np.log10(img.T + 1e-3), origin="lower", extent=extent, cmap="magma",
                   vmin=-3, vmax=np.log10(vmax))
    ax.set_title(title, fontsize=10); ax.set_aspect("equal")
    return im


def density_panels(field, grid, path, title=""):
    """Front max-projection (u-w) + three u-w slices at different depths + a top (u-v) slice
    through the box centre height. Colour = log10 density."""
    ax_u, ax_v, ax_w = grid.axes()
    ext_uw = [ax_u[0], ax_u[-1], ax_w[0], ax_w[-1]]
    ext_uv = [ax_u[0], ax_u[-1], ax_v[0], ax_v[-1]]
    vmax = float(field.max())
    ny = field.shape[1]
    fig, axs = plt.subplots(1, 5, figsize=(26, 6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1]))
    im = _show(axs[0], field.max(axis=1), ext_uw, "front: max over depth (u-w)", vmax)
    for ax, j in zip(axs[1:4], [ny // 4, ny // 2, 3 * ny // 4]):
        _show(ax, field[:, j, :], ext_uw, f"front slice at depth v = {ax_v[j]:+.3f}", vmax)
    k = field.shape[2] // 2
    _show(axs[4], field[:, :, k], ext_uv, f"top slice at height w = {ax_w[k]:+.3f} (u-v)", vmax)
    for ax in axs:
        ax.set_xlabel("u (box frame)")
    fig.colorbar(im, ax=axs, shrink=0.8, label="log10 density")
    fig.suptitle(title)
    fig.savefig(path, dpi=90, bbox_inches="tight"); plt.close(fig)


def to_density_units(dgm):
    """GUDHI pairs (-b, -a) -> (birth density b, death density a), b > a."""
    return -dgm


def diagram_and_barcode(res, path, title="", cutoff=None, max_bars=60):
    cols = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
    fig, axs = plt.subplots(1, 3, figsize=(21, 6))
    ax = axs[1]  # zoom on H1, the primary dimension
    dg1 = res["dgms"].get(1, np.zeros((0, 2)))
    if len(dg1):
        p = to_density_units(dg1)
        top = p[:, 0].max()
        ax.scatter(p[:, 0], p[:, 1], s=5, c=cols[1], alpha=0.5)
        ax.plot([0, top], [0, top], "k-", lw=0.5)
        if cutoff:
            ax.plot([cutoff, top], [0, top - cutoff], "k--", lw=0.7)
            ax.set_title(f"H1 only (zoom): {int(np.sum(p[:, 0] - p[:, 1] > cutoff))} of {len(p)} above cutoff")
        ax.set_xlabel("birth density"); ax.set_ylabel("death density")
    ax = axs[0]
    hi = 0.0
    for d, dg in res["dgms"].items():
        if len(dg):
            p = to_density_units(dg)
            ax.scatter(p[:, 0], p[:, 1], s=4, c=cols[d], alpha=0.5, label=f"H{d} ({len(dg)})")
            hi = max(hi, p.max())
    ax.plot([0, hi], [0, hi], "k-", lw=0.5)
    if cutoff:
        ax.plot([cutoff, hi], [0, hi - cutoff], "k--", lw=0.7, label=f"persistence = {cutoff:.3g}")
    ax.set_xlabel("birth density (feature appears)"); ax.set_ylabel("death density (feature disappears)")
    ax.set_title("persistence diagram (superlevel sets)"); ax.legend(fontsize=8)
    ax = axs[2]
    y = 0
    for d in sorted(res["dgms"]):
        dg = res["dgms"][d]
        if not len(dg):
            continue
        p = to_density_units(dg)
        order = np.argsort(-(p[:, 0] - p[:, 1]))[:max_bars // 3]
        for i in order:
            ax.plot([p[i, 1], p[i, 0]], [y, y], c=cols[d], lw=2); y += 1
        y += 2
    ax.set_xlabel("density"); ax.set_yticks([])
    ax.set_title(f"barcode: {max_bars // 3} longest bars per dimension (H0 blue, H1 orange, H2 green)")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)
