"""HAC step 2-3: convert exported neural Gaussians to the INRIA .ply convention, validate the export with the
STOCK INRIA rasterizer, and write the crop-box subset that the density-field code reads.

Runs on ada in FCGS_env alone (its diff_gaussian_rasterization is the stock graphdeco build; HAC_env must
NOT be active, it shadows that module). Reads the output of scripts/export_neural_gaussians.py.

Per camera:
  INRIA conversion of the decoded set: opacity -> logit, scale -> log, rot -> normalized (w,x,y,z),
  f_dc = (rgb - 0.5) / C0. The scene is then read back through topogs.io_ply.GaussianScene (sigmoid, exp,
  normalize: exactly what the density field uses) and rendered with the stock rasterizer at SH degree 0.
  (a) all decoded Gaussians (before the zero-scale drop) vs the repo's own render of the same set
  (b) exported set (zero-scale Gaussians dropped) vs the same own render
  (b') exported set vs the repo's standard test render (prefilter_voxel culling)
  PSNR on float renders (clamped to [0,1]) over the full frame and the wheel bounding box
  (render_eval.wheel_bbox: projected original Gaussians in the crop box, dilated 5%), plus max |diff|.
  Then the crop subset (extract_crop_subset.crop_mask, a superset of what any analysis grid touches) of the
  exported set is written as a minimal INRIA .ply: x,y,z,nx,ny,nz,f_dc_0..2,opacity,scale_0..2,rot_0..3.

    python scripts/hac_validate_export.py --exp DIR --tag "" --ply-out DIR --mask-ply original_crop.ply --box configs/crop_box.json
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scripts.extract_crop_subset import crop_mask  # noqa: E402
from scripts.render_eval import wheel_bbox, wheel_points  # noqa: E402
from topogs.io_ply import GaussianScene, save_ply  # noqa: E402

C0 = 0.28209479177387814
NAMES = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
         + ["scale_%d" % i for i in range(3)] + ["rot_%d" % i for i in range(4)])


def to_inria(d):
    """Activated decoded values -> INRIA raw structured array (float32), as the .ply stores them."""
    n = len(d["opacity"])
    raw = np.zeros(n, dtype=[(k, "f4") for k in NAMES])
    raw["x"], raw["y"], raw["z"] = d["xyz"].T
    o = d["opacity"].astype(np.float64)
    with np.errstate(divide="ignore"):
        raw["opacity"] = np.log(o) - np.log1p(-o)                      # logit; o == 1 -> +inf
        ls = np.log(d["scaling"].astype(np.float64))                  # scale 0 -> -inf
    for i in range(3):
        raw["scale_%d" % i] = ls[:, i]
    q = d["rot"].astype(np.float64); q = q / np.linalg.norm(q, axis=1, keepdims=True)
    for i in range(4):
        raw["rot_%d" % i] = q[:, i]
    fdc = (d["rgb"].astype(np.float64) - 0.5) / C0
    for i in range(3):
        raw["f_dc_%d" % i] = fdc[:, i]
    return raw


def stock_render(scene, cam, dev):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)
    r = scene.raw
    shs = t(np.stack([r["f_dc_0"], r["f_dc_1"], r["f_dc_2"]], 1)[:, None, :])          # (N,1,3): SH degree 0
    H, W = [int(x) for x in cam["hw"]]
    s = GaussianRasterizationSettings(image_height=H, image_width=W, tanfovx=float(np.tan(cam["fov"][0] / 2)),
                                      tanfovy=float(np.tan(cam["fov"][1] / 2)), bg=torch.zeros(3, device=dev),
                                      scale_modifier=1.0, viewmatrix=t(cam["view"]), projmatrix=t(cam["full"]),
                                      sh_degree=0, campos=t(cam["campos"]), prefiltered=False, debug=False)
    xyz = t(scene.means)
    out = GaussianRasterizer(raster_settings=s)(
        means3D=xyz, means2D=torch.zeros_like(xyz), shs=shs, colors_precomp=None,
        opacities=t(scene.opacities[:, None]), scales=t(scene.scales), rotations=t(scene.quats), cov3D_precomp=None)
    return out[0].clamp(0, 1).cpu().numpy()


def compare(a, b, bbox):
    def one(x, y):
        mse = float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))
        return dict(psnr=(float("inf") if mse == 0 else round(float(10 * np.log10(1.0 / mse)), 3)),
                    max_abs=round(float(np.abs(x - y).max()), 6))
    out = dict(full=one(a, b))
    if bbox is not None:
        x0, x1, y0, y1 = bbox[:4]
        out["wheel"] = one(a[:, y0:y1, x0:x1], b[:, y0:y1, x0:x1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True); ap.add_argument("--tag", default="")
    ap.add_argument("--ply-out", required=True); ap.add_argument("--mask-ply", required=True)
    ap.add_argument("--box", required=True); ap.add_argument("--min-psnr", type=float, default=45.0)
    ap.add_argument("--keep-zero", action="store_true", help="also write the crop subset with zero-scale Gaussians kept")
    ap.add_argument("--diag-png", action="store_true", help="save culled / unculled / |diff|x4 PNG for the first camera")
    a = ap.parse_args()
    import diff_gaussian_rasterization as dgr
    rast = dict(file=dgr.__file__, has_visible_filter=hasattr(dgr.GaussianRasterizer, "visible_filter"))
    print("rasterizer", rast, flush=True)
    assert not rast["has_visible_filter"], "not the stock rasterizer"
    dev = "cuda"
    exp = json.load(open(os.path.join(a.exp, "export%s.json" % a.tag)))
    cz = np.load(os.path.join(a.exp, "cameras%s.npz" % a.tag))
    box = json.load(open(a.box))
    pts = wheel_points(a.mask_ply, a.box)
    os.makedirs(a.ply_out, exist_ok=True)
    rec = dict(which=exp["which"], tag=a.tag, stock_rasterizer=rast, min_psnr=a.min_psnr, cameras={})
    ok_all = True
    for name in exp["cameras"]:
        d = dict(np.load(os.path.join(a.exp, name + a.tag + ".npz")))
        sh = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in d.items()}
        assert sh == exp["cameras"][name]["sha256"], "npz differs from export record"
        cam = {k: cz["%s__%s" % (name, k)] for k in ["view", "full", "campos", "fov", "hw"]}
        H, W = [int(x) for x in cam["hw"]]
        camd = dict(full=torch.from_numpy(cam["full"]), view=torch.from_numpy(cam["view"]))
        bbox = wheel_bbox(pts, camd, W, H)
        own_all = np.load(os.path.join(a.exp, name + a.tag + "_own_all.npy"))
        own_cull = np.load(os.path.join(a.exp, name + a.tag + "_own_cull.npy"))
        raw = to_inria(d)
        scene_all = GaussianScene(raw)
        zero = np.any(d["scaling"] == 0, axis=1)
        exported = GaussianScene(raw[~zero].copy())
        img_all = stock_render(scene_all, cam, dev)
        img_exp = stock_render(exported, cam, dev)
        c = dict(wheel_bbox=None if bbox is None else list(bbox),
                 a_conversion_all_vs_own_all=compare(img_all, own_all, bbox),
                 b_exported_vs_own_all=compare(img_exp, own_all, bbox),
                 b2_exported_vs_own_cull=compare(img_exp, own_cull, bbox),
                 own_all_vs_own_cull=compare(own_all, own_cull, bbox),
                 n_exported=int(exported.n), n_zero_scale_dropped=int(zero.sum()),
                 n_logit_posinf=int(np.sum(np.isposinf(raw["opacity"][~zero]))),
                 n_nonfinite_exported={k: int(np.sum(~np.isfinite(exported.raw[k]))) for k in NAMES},
                 roundtrip_max_err=dict(
                     opacity=float(np.abs(exported.opacities - d["opacity"][~zero]).max()),
                     scale_rel=float(np.abs(exported.scales / d["scaling"][~zero] - 1).max()),
                     quat=float(np.abs(exported.quats - d["rot"][~zero]).max()),
                     rgb=float(np.abs(C0 * np.stack([exported.raw["f_dc_%d" % i] for i in range(3)], 1) + 0.5 - d["rgb"][~zero]).max())))
        cpath = os.path.join(a.exp, name + a.tag + "_cullset.npz")
        if os.path.exists(cpath):
            # like-for-like: the culled set (what the repo's standard render draws), converted, zero-scale kept,
            # stock render vs the repo's standard render; and does culling change what lies in the crop box?
            dc = dict(np.load(cpath)); rawc = to_inria(dc); sc_c = GaussianScene(rawc)
            c["b3_cullset_vs_own_cull"] = compare(stock_render(sc_c, cam, dev), own_cull, bbox)
            kc = crop_mask(sc_c, box); ka = crop_mask(scene_all, box)
            c["crop_cullset_vs_all"] = dict(
                n_crop_all=int(ka.sum()), n_crop_cullset=int(kc.sum()),
                identical=bool(ka.sum() == kc.sum() and hashlib.sha256(raw[ka].tobytes()).hexdigest()
                               == hashlib.sha256(rawc[kc].tobytes()).hexdigest()))
        # where does the repo's culled render differ from its unculled one? (pixels off by > 1/255)
        dpx = np.abs(own_all - own_cull).max(0) > 1.0 / 255
        bw = int(0.05 * min(H, W)); border = np.zeros((H, W), bool)
        border[:bw] = border[-bw:] = True; border[:, :bw] = border[:, -bw:] = True
        c["cull_diff"] = dict(frac_pixels=round(float(dpx.mean()), 5),
                              frac_of_diff_in_5pct_border=round(float(dpx[border].sum() / max(1, dpx.sum())), 4))
        if bbox is not None:
            x0, x1, y0, y1 = bbox[:4]
            c["cull_diff"]["frac_pixels_wheel"] = round(float(dpx[y0:y1, x0:x1].mean()), 5)
        if a.diag_png and name == list(exp["cameras"])[0]:
            from PIL import Image
            q = lambda x: (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8).transpose(1, 2, 0)
            Image.fromarray(np.concatenate([q(own_cull), q(own_all), q(np.abs(own_all - own_cull) * 4)], 1)).save(
                os.path.join(a.ply_out, "%s%s_cull_vs_all_diffx4.png" % (name, a.tag)))
        keep = crop_mask(exported, box)
        sub = GaussianScene(exported.raw[keep].copy())
        ply = os.path.join(a.ply_out, "%s%s_crop.ply" % (name, a.tag))
        save_ply(sub, ply)
        c["n_crop_subset"] = int(sub.n); c["crop_ply"] = ply
        c["crop_raw_sha256"] = hashlib.sha256(sub.raw.tobytes()).hexdigest()
        if a.keep_zero:
            # sensitivity set: zero-scale Gaussians KEPT (log-scale -inf -> scale 0; the analysis filter widens
            # every axis to sqrt(0 + f^2) = f, exactly as it does for the many near-zero axes of other Gaussians)
            kz = crop_mask(scene_all, box)
            subz = GaussianScene(raw[kz].copy())
            plyz = os.path.join(a.ply_out, "%s%s_keepzero_crop.ply" % (name, a.tag))
            save_ply(subz, plyz)
            c["n_crop_subset_keepzero"] = int(subz.n)
            c["n_crop_zero_scale"] = int(np.sum(kz & zero))
            c["crop_keepzero_raw_sha256"] = hashlib.sha256(subz.raw.tobytes()).hexdigest()
        checks = ["a_conversion_all_vs_own_all", "b_exported_vs_own_all", "b2_exported_vs_own_cull"]
        c["pass_by_check"] = {k: bool(all(v["psnr"] >= a.min_psnr for v in c[k].values())) for k in checks}
        c["pass"] = bool(all(c["pass_by_check"].values()))
        ok_all &= c["pass"]
        rec["cameras"][name] = c
        np.save(os.path.join(a.exp, name + a.tag + "_stock_exp.npy"), img_exp)
        print(name, json.dumps(c), flush=True)
    rec["all_pass"] = bool(ok_all)
    json.dump(rec, open(os.path.join(a.ply_out, "validate%s.json" % a.tag), "w"), indent=2)
    print("ALL_PASS", ok_all)


if __name__ == "__main__":
    main()
