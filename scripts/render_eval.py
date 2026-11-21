"""Render the Mip-NeRF 360 test views (every 8th image by name) of one scene and compute
PSNR / SSIM / LPIPS against the ground-truth photos. Runs on ada inside the FCGS env, using the
same stock INRIA `diff_gaussian_rasterization` the pretrained scene was trained with.

Follows the INRIA evaluation conventions (render.py + metrics.py of gaussian-splatting):
 - cameras from the model's cameras.json (camera-to-world rotation + centre, full-res fx/fy);
   only the field of view is used, the image size is that of images_4 (eval=True, -r 1)
 - test split: sorted image names, idx % 8 == 0
 - black background, SH degree 3, znear 0.01, zfar 100
 - renders and GT quantized to 8 bit (INRIA writes PNGs, then reads them back)
 - PSNR per view then mean; SSIM with 11x11 Gaussian window (sigma 1.5); LPIPS with VGG

The scene is either a .ply (--ply) or a self-made variant rebuilt from the original
(--variant results/variants/<name>/<name>.json --keep-dir dir-with-keep.npy), checked against
the variant's raw_sha256 so the render uses exactly the data the topology was computed on.

    python scripts/render_eval.py --name original --ply PLY --cameras cameras.json \
        --images images_4 --out out_dir [--save-views 3]
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from topogs.compress import quantize  # noqa: E402
from topogs.io_ply import GaussianScene, load_ply  # noqa: E402


# ---------------------------------------------------------------- scene -> GPU tensors
def scene_tensors(scene, dev):
    r = scene.raw
    xyz = np.stack([r["x"], r["y"], r["z"]], 1)
    dc = np.stack([r["f_dc_0"], r["f_dc_1"], r["f_dc_2"]], 1)[:, None, :]           # (N,1,3)
    rest = np.stack([r["f_rest_%d" % i] for i in range(45)], 1).reshape(-1, 3, 15)   # channel-major
    rest = rest.transpose(0, 2, 1)                                                  # (N,15,3)
    shs = np.concatenate([dc, rest], 1).astype(np.float32)                          # (N,16,3)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)
    return dict(xyz=t(xyz), shs=t(shs), opacity=t(scene.opacities[:, None]),
                scales=t(scene.scales), rots=t(scene.quats))


# ---------------------------------------------------------------- cameras
def projection_matrix(znear, zfar, fovx, fovy):
    tx, ty = math.tan(fovx / 2), math.tan(fovy / 2)
    top, bottom, right, left = ty * znear, -ty * znear, tx * znear, -tx * znear
    P = torch.zeros(4, 4)
    P[0, 0] = 2 * znear / (right - left); P[1, 1] = 2 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left); P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0; P[2, 2] = zfar / (zfar - znear); P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def test_cameras(cameras_json):
    cams = json.load(open(cameras_json))
    cams = sorted(cams, key=lambda c: c["img_name"])
    return [c for i, c in enumerate(cams) if i % 8 == 0]


def camera_setup(c, W, H, dev):
    c2w = np.eye(4); c2w[:3, :3] = np.array(c["rotation"]); c2w[:3, 3] = np.array(c["position"])
    w2c = np.linalg.inv(c2w)
    fovx = 2 * math.atan(c["width"] / (2 * c["fx"])); fovy = 2 * math.atan(c["height"] / (2 * c["fy"]))
    view = torch.tensor(w2c, dtype=torch.float32).t()                      # INRIA stores transposed
    proj = projection_matrix(0.01, 100.0, fovx, fovy).t()
    full = view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    return dict(view=view.to(dev), full=full.to(dev), campos=torch.tensor(c["position"], dtype=torch.float32).to(dev),
                tanfovx=math.tan(fovx / 2), tanfovy=math.tan(fovy / 2), W=W, H=H)


def render(g, cam):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    s = GaussianRasterizationSettings(image_height=cam["H"], image_width=cam["W"], tanfovx=cam["tanfovx"],
                                      tanfovy=cam["tanfovy"], bg=torch.zeros(3, device=g["xyz"].device),
                                      scale_modifier=1.0, viewmatrix=cam["view"], projmatrix=cam["full"],
                                      sh_degree=3, campos=cam["campos"], prefiltered=False, debug=False)
    means2d = torch.zeros_like(g["xyz"])
    color, _radii = GaussianRasterizer(raster_settings=s)(
        means3D=g["xyz"], means2D=means2d, shs=g["shs"], colors_precomp=None, opacities=g["opacity"],
        scales=g["scales"], rotations=g["rots"], cov3D_precomp=None)
    return color.clamp(0, 1)


# ---------------------------------------------------------------- metrics (INRIA definitions)
def q8(x):
    return torch.round(x * 255.0) / 255.0


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float(20 * torch.log10(1.0 / torch.sqrt(mse)))


def _gauss_window(ws=11, sigma=1.5, ch=3):
    g = torch.tensor([math.exp(-(x - ws // 2) ** 2 / float(2 * sigma ** 2)) for x in range(ws)])
    g = (g / g.sum()).unsqueeze(1)
    w = g.mm(g.t()).float().unsqueeze(0).unsqueeze(0)
    return w.expand(ch, 1, ws, ws).contiguous()


def ssim(a, b, ws=11):
    a, b = a.unsqueeze(0), b.unsqueeze(0)
    w = _gauss_window(ws).to(a.device)
    mu1 = F.conv2d(a, w, padding=ws // 2, groups=3); mu2 = F.conv2d(b, w, padding=ws // 2, groups=3)
    s11 = F.conv2d(a * a, w, padding=ws // 2, groups=3) - mu1 ** 2
    s22 = F.conv2d(b * b, w, padding=ws // 2, groups=3) - mu2 ** 2
    s12 = F.conv2d(a * b, w, padding=ws // 2, groups=3) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))
    return float(m.mean())


# ---------------------------------------------------------------- wheel-region mask (extra metric)
def wheel_points(mask_ply, box_json):
    """World positions of the ORIGINAL's Gaussians (opacity > 0.02) inside the analysis crop box.
    The same points define the mask for every variant."""
    box = json.load(open(box_json)); s = load_ply(mask_ply)
    R = np.array(box["R_box"]); P = (s.means - np.array(box["centre_world"])) @ R.T
    m = np.all(np.abs(P) < np.array(box["half"]), axis=1) & (s.opacities > 0.02)
    return s.means[m]


def wheel_bbox(pts, cam, W, H, dilate=0.05):
    """Pixel bounding box of the projected wheel points (INRIA projection + ndc2Pix), dilated by 5%
    of its size; None if fewer than half of the points land in front of the camera and in the image."""
    full = cam["full"].cpu().numpy().astype(np.float64)
    view = cam["view"].cpu().numpy().astype(np.float64)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], 1)
    z = (hom @ view)[:, 2]
    ph = hom @ full; ndc = ph[:, :3] / ph[:, 3:4]
    px = ((ndc[:, 0] + 1) * W - 1) * 0.5; py = ((ndc[:, 1] + 1) * H - 1) * 0.5
    ok = (z > 0.2) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    if ok.mean() < 0.5:
        return None
    x0, x1, y0, y1 = px[ok].min(), px[ok].max(), py[ok].min(), py[ok].max()
    dx, dy = dilate * (x1 - x0), dilate * (y1 - y0)
    x0, x1 = int(max(0, x0 - dx)), int(min(W, x1 + dx + 1)); y0, y1 = int(max(0, y0 - dy)), int(min(H, y1 + dy + 1))
    return (x0, x1, y0, y1, float(ok.mean()))


# ---------------------------------------------------------------- scene loading
def load_scene(args):
    if args.ply:
        return load_ply(args.ply), None
    rec = json.load(open(args.variant))
    orig = load_ply(args.original)
    info = rec["variant_info"]
    if info["kind"] in ("prune_opacity", "prune_random"):
        bits = os.path.join(args.keep_dir, "keep_bits.npy")    # np.packbits(mask): ~0.8 MB vs 7-17 MB
        if os.path.exists(bits):
            keep = np.unpackbits(np.load(bits))[:orig.n].astype(bool)
        else:
            keep = np.load(os.path.join(args.keep_dir, "keep.npy"))
        scene = GaussianScene(orig.raw[keep].copy())
    elif info["kind"] == "quantize":
        scene, _ = quantize(orig, info["bits"], tuple(info["fields"]))
    else:
        raise ValueError(info["kind"])
    sha = hashlib.sha256(scene.raw.tobytes()).hexdigest()
    if sha != rec["raw_sha256"]:
        raise SystemExit("rebuilt variant does not match the analysed one: %s vs %s" % (sha, rec["raw_sha256"]))
    return scene, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ply"); ap.add_argument("--variant"); ap.add_argument("--original"); ap.add_argument("--keep-dir")
    ap.add_argument("--cameras", required=True); ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--save-views", type=int, default=3)
    ap.add_argument("--mask-ply", help="original crop-subset .ply; enables wheel-region metrics")
    ap.add_argument("--box", help="crop box JSON (configs/crop_box.json)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")
    t0 = time.time()
    scene, sha = load_scene(args)
    g = scene_tensors(scene, dev)
    import lpips
    lp = lpips.LPIPS(net="vgg").to(dev).eval()
    rows = []
    wpts = wheel_points(args.mask_ply, args.box) if args.mask_ply else None
    with torch.no_grad():
        for i, c in enumerate(test_cameras(args.cameras)):
            gt = Image.open(os.path.join(args.images, c["img_name"] + ".JPG")).convert("RGB")
            W, H = gt.size
            gt = torch.from_numpy(np.asarray(gt, dtype=np.float32) / 255.0).permute(2, 0, 1).to(dev)
            cam = camera_setup(c, W, H, dev)
            img = q8(render(g, cam))
            row = dict(view=c["img_name"], psnr=psnr(img, gt), ssim=ssim(img, gt),
                       lpips=float(lp(img.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1)))
            if wpts is not None:
                bb = wheel_bbox(wpts, cam, W, H)
                if bb is not None:
                    x0, x1, y0, y1, frac = bb
                    a_, b_ = img[:, y0:y1, x0:x1], gt[:, y0:y1, x0:x1]
                    row["wheel"] = dict(bbox=[x0, x1, y0, y1], area_frac=round((x1 - x0) * (y1 - y0) / float(W * H), 4),
                                        pts_in_view=round(frac, 3), psnr=psnr(a_, b_), ssim=ssim(a_, b_),
                                        lpips=float(lp(a_.unsqueeze(0) * 2 - 1, b_.unsqueeze(0) * 2 - 1)))
            rows.append(row)
            if i < args.save_views:
                Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(
                    os.path.join(args.out, "%s_%s.png" % (args.name, c["img_name"])))
            print(json.dumps(row), flush=True)
    res = dict(name=args.name, n_gaussians=int(scene.n), raw_sha256=sha or hashlib.sha256(scene.raw.tobytes()).hexdigest(),
               n_views=len(rows), image_size=[W, H],
               psnr=float(np.mean([r["psnr"] for r in rows])), ssim=float(np.mean([r["ssim"] for r in rows])),
               lpips=float(np.mean([r["lpips"] for r in rows])), seconds=round(time.time() - t0, 1),
               gpu=torch.cuda.get_device_name(0), per_view=rows)
    wv = [r["wheel"] for r in rows if "wheel" in r]
    if wv:
        res["wheel"] = dict(n_views=len(wv), mean_area_frac=float(np.mean([w["area_frac"] for w in wv])),
                            psnr=float(np.mean([w["psnr"] for w in wv])), ssim=float(np.mean([w["ssim"] for w in wv])),
                            lpips=float(np.mean([w["lpips"] for w in wv])),
                            note="extra: metrics on the bounding box of the projected wheel (same box for every variant)")
    json.dump(res, open(os.path.join(args.out, "%s_metrics.json" % args.name), "w"), indent=2)
    print("MEAN", args.name, "PSNR %.3f SSIM %.4f LPIPS %.4f" % (res["psnr"], res["ssim"], res["lpips"]), flush=True)
    if wv:
        w = res["wheel"]
        print("WHEEL", args.name, "views %d area %.3f PSNR %.3f SSIM %.4f LPIPS %.4f" % (
            w["n_views"], w["mean_area_frac"], w["psnr"], w["ssim"], w["lpips"]), flush=True)


if __name__ == "__main__":
    main()
