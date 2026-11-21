"""HAC step 2: decode a trained HAC or Scaffold-GS model into explicit (neural) Gaussians, per camera.

Runs on ada in HAC_env (python 3.7, torch 1.12), with the working directory set to the model's own repo
(HAC/ or Scaffold-GS/), because it reuses that repo's `generate_neural_gaussians`, `render` and
`prefilter_voxel` unchanged. HAC and Scaffold-GS both define top-level `scene` / `gaussian_renderer`
packages, so each model needs its own process.

Per camera it writes (to --out, node-local scratch):
  <cam>.npz         decoded Gaussians with no frustum culling (visible_mask=None), opacity > 0 only:
                    xyz, rgb, opacity, scaling (activated / linear), rot (unit quaternion w,x,y,z)
  <cam>_own_cull.npy the repo's standard test render (prefilter_voxel culling, as in train.py render_set)
  <cam>_own_all.npy  the repo's render with visible_mask=None (exactly the exported set, repo rasterizer)
  <cam>_own_cull.png the standard render quantized like torchvision.save_image (for the check vs the
                     stored training-time test PNG)
and cameras.npz (the exact view / projection matrices used, reused by the stock-rasterizer check),
export.json (counts, rasterizer in use, sha256 of every exported array).

Camera construction: cameras.json (camera-to-world rotation + centre, full-resolution fx/fy) -> W2C,
the repo's getProjectionMatrix (znear 0.01, zfar 100), image size by the repos' loader rule for -r -1
(width 1600, height int(h / (w / 1600))). camera_center = inverse(world_view_transform)[3, :3], as in
the repos' Camera class.

    cd ~/topogs-fcgs/hac/HAC && python .../export_neural_gaussians.py --which hac \
        --model ../outputs/hac_bicycle_l0.004 --cams _DSC8679,... --out /ssd_scratch/.../hac
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.getcwd())          # the model's repo (HAC/ or Scaffold-GS/)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_model(which, model_dir, iteration, hac_args):
    from scene.gaussian_model import GaussianModel
    cfg = open(os.path.join(model_dir, "cfg_args")).read()
    ns = eval(cfg, {"Namespace": SimpleNamespace})                      # repo convention (get_combined_args)
    pc_dir = os.path.join(model_dir, "point_cloud", "iteration_%d" % iteration)
    if which == "hac":
        pc = GaussianModel(ns.feat_dim, ns.n_offsets, ns.voxel_size, ns.update_depth, ns.update_init_factor,
                           ns.update_hierachy_factor, ns.use_feat_bank,
                           n_features_per_level=hac_args["n_features"], log2_hashmap_size=hac_args["log2"],
                           log2_hashmap_size_2D=hac_args["log2_2D"], decoded_version=True)
        pc.load_ply_sparse_gaussian(os.path.join(pc_dir, "point_cloud.ply"))
        pc.load_mlp_checkpoints(os.path.join(pc_dir, "checkpoint.pth"))
    else:
        pc = GaussianModel(ns.feat_dim, ns.n_offsets, ns.voxel_size, ns.update_depth, ns.update_init_factor,
                           ns.update_hierachy_factor, ns.use_feat_bank, ns.appearance_dim, ns.ratio,
                           ns.add_opacity_dist, ns.add_cov_dist, ns.add_color_dist)
        pc.load_ply_sparse_gaussian(os.path.join(pc_dir, "point_cloud.ply"))
        pc.load_mlp_checkpoints(pc_dir)
    pc.eval()
    return pc, ns


def make_camera(c, width):
    from utils.graphics_utils import getProjectionMatrix
    W = int(width); scale = c["width"] / float(width); H = int(c["height"] / scale)
    c2w = np.eye(4); c2w[:3, :3] = np.array(c["rotation"]); c2w[:3, 3] = np.array(c["position"])
    w2c = np.linalg.inv(c2w).astype(np.float32)
    fovx = 2 * math.atan(c["width"] / (2 * c["fx"])); fovy = 2 * math.atan(c["height"] / (2 * c["fy"]))
    wv = torch.tensor(w2c).transpose(0, 1).cuda()
    proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    full = wv.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    return SimpleNamespace(uid=0, image_name=c["img_name"], FoVx=fovx, FoVy=fovy, image_width=W, image_height=H,
                           world_view_transform=wv, projection_matrix=proj, full_proj_transform=full,
                           camera_center=wv.inverse()[3, :3], znear=0.01, zfar=100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=["hac", "scaffold"])
    ap.add_argument("--model", required=True); ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--cams", required=True); ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--out", required=True); ap.add_argument("--tag", default="")
    ap.add_argument("--n-features", type=int, default=4); ap.add_argument("--log2", type=int, default=13)
    ap.add_argument("--log2-2d", type=int, default=15)
    ap.add_argument("--save-cull-set", action="store_true",
                    help="also save the culled set (visible_mask = prefilter_voxel), for the like-for-like render check")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import diff_gaussian_rasterization as dgr
    from gaussian_renderer import generate_neural_gaussians, prefilter_voxel, render
    rast = dict(file=dgr.__file__, has_visible_filter=hasattr(dgr.GaussianRasterizer, "visible_filter"))
    print("rasterizer", rast, "cwd", os.getcwd(), flush=True)

    torch.manual_seed(0)
    with torch.no_grad():
        pc, ns = load_model(a.which, a.model, a.iteration,
                            dict(n_features=a.n_features, log2=a.log2, log2_2D=a.log2_2d))
        bg = torch.tensor([1, 1, 1] if ns.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        cams_json = {c["img_name"]: c for c in json.load(open(os.path.join(a.model, "cameras.json")))}
        n_anchor = int(pc.get_anchor.shape[0])
        rec = dict(which=a.which, model=os.path.abspath(a.model), iteration=a.iteration, tag=a.tag, rasterizer=rast,
                   n_anchors=n_anchor, n_offsets=int(pc.n_offsets), white_background=bool(ns.white_background),
                   torch=torch.__version__, cameras={})
        if a.which == "hac":
            rec["hac"] = dict(decoded_version=bool(pc.decoded_version), n_features=a.n_features, log2=a.log2,
                              log2_2D=a.log2_2d, n_anchor_with_active_offset=int(pc.get_mask_anchor.sum().item()))
        cam_mats = {}
        for name in a.cams.split(","):
            t0 = time.time()
            cam = make_camera(cams_json[name], a.width)
            out = generate_neural_gaussians(cam, pc, visible_mask=None)
            xyz, rgb, opac, scal, rot = [x.float().contiguous() for x in out[:5]]
            arr = dict(xyz=xyz.cpu().numpy(), rgb=rgb.cpu().numpy(), opacity=opac.reshape(-1).cpu().numpy(),
                       scaling=scal.cpu().numpy(), rot=rot.cpu().numpy())
            vm = prefilter_voxel(cam, pc, pipe, bg)
            if a.save_cull_set:   # diagnostic: the Gaussian set the repo's standard (culled) render uses
                xs = [x.float().cpu().numpy() for x in generate_neural_gaussians(cam, pc, visible_mask=vm)[:5]]
                np.savez(os.path.join(a.out, name + a.tag + "_cullset.npz"), xyz=xs[0], rgb=xs[1],
                         opacity=xs[2].reshape(-1), scaling=xs[3], rot=xs[4])
            own_cull = render(cam, pc, pipe, bg, visible_mask=vm)["render"].clamp(0, 1)
            own_all = render(cam, pc, pipe, bg, visible_mask=None)["render"].clamp(0, 1)
            np.savez(os.path.join(a.out, name + a.tag + ".npz"), **arr)
            np.save(os.path.join(a.out, name + a.tag + "_own_cull.npy"), own_cull.cpu().numpy())
            np.save(os.path.join(a.out, name + a.tag + "_own_all.npy"), own_all.cpu().numpy())
            png = own_cull.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            from PIL import Image
            Image.fromarray(png).save(os.path.join(a.out, name + a.tag + "_own_cull.png"))
            s = arr["scaling"]; o = arr["opacity"]
            c = dict(n_decoded=n_anchor * int(pc.n_offsets), n_opacity_pos=int(len(o)),
                     n_zero_scale_any=int(np.sum(np.any(s == 0, axis=1))),
                     n_zero_scale_per_axis=[int(np.sum(s[:, i] == 0)) for i in range(3)],
                     n_opacity_eq_1=int(np.sum(o >= 1.0)), opacity_min=float(o.min()), opacity_max=float(o.max()),
                     n_nonfinite={k: int(np.sum(~np.isfinite(v))) for k, v in arr.items()},
                     rot_norm_dev_max=float(np.abs(np.linalg.norm(arr["rot"], axis=1) - 1).max()),
                     n_visible_anchors_prefilter=int(vm.sum().item()),
                     image_hw=[cam.image_height, cam.image_width],
                     sha256={k: sha(v) for k, v in arr.items()}, seconds=round(time.time() - t0, 2))
            rec["cameras"][name] = c
            cam_mats[name] = dict(view=cam.world_view_transform.cpu().numpy(), full=cam.full_proj_transform.cpu().numpy(),
                                  campos=cam.camera_center.cpu().numpy(), fov=np.array([cam.FoVx, cam.FoVy]),
                                  hw=np.array([cam.image_height, cam.image_width]))
            print(name, json.dumps({k: v for k, v in c.items() if k != "sha256"}), flush=True)
        np.savez(os.path.join(a.out, "cameras%s.npz" % a.tag),
                 **{"%s__%s" % (n, k): v for n, d in cam_mats.items() for k, v in d.items()})
        rec["max_memory_allocated_MiB"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    json.dump(rec, open(os.path.join(a.out, "export%s.json" % a.tag), "w"), indent=2)


if __name__ == "__main__":
    main()
