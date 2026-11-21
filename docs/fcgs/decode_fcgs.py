import time

import torch
import torch.nn as nn
from model.FCGS_model import FCGS
import os
from gaussian_renderer import GaussianModel
from argparse import ArgumentParser
import sys
from typing import NamedTuple

def save_ply_fast(g, path):
    import numpy as np
    from plyfile import PlyData, PlyElement
    os.makedirs(os.path.dirname(path), exist_ok=True)
    xyz = g._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = g._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = g._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = g._opacity.detach().cpu().numpy()
    scale = g._scaling.detach().cpu().numpy()
    rotation = g._rotation.detach().cpu().numpy()
    names = g.construct_list_of_attributes()
    elements = np.empty(xyz.shape[0], dtype=[(a, 'f4') for a in names])
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    for i, a in enumerate(names):
        elements[a] = attributes[:, i]
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)


def train(args):

    step_num = len(os.listdir(os.path.join(args.bit_path_from, str(args.lmd))))
    lmd = args.lmd
    chunk_size_list = [int(v) for v in args.chunk_sizes.split(',')]  # [topogs] was hard-coded [200_0000, 100_0000, 100_0000]; must match between encode and decode
    print('chunk_size_list:', chunk_size_list)

    CM = FCGS(
        Q=1,
        resolutions_list=[300, 400, 500],
        resolutions_list_3D=[70, 80, 90],
        norm_radius=args.nr,
    ).cuda()
    CM.load_state_dict(torch.load(f'./checkpoints/checkpoint_{lmd}.pkl'), strict=True)

    g_xyz_list = []
    g_fea_list = []
    CM.eval()
    with torch.no_grad():
        for s in range(step_num):
            bit_save_path = os.path.join(args.bit_path_from, f"{lmd}/{s}")
            g_xyz_out, g_fea_out = CM.decomprss(root_path=bit_save_path, chunk_size_list=chunk_size_list)
            g_xyz_list.append(g_xyz_out.cpu())  # [topogs] VRAM: collect step outputs on CPU (output-neutral)
            g_fea_list.append(g_fea_out.cpu())
            del g_xyz_out, g_fea_out
            torch.cuda.empty_cache()
    print('decode loop done; max_memory_allocated_MiB: %.0f  max_memory_reserved_MiB: %.0f' % (torch.cuda.max_memory_allocated()/2**20, torch.cuda.max_memory_reserved()/2**20))
            
    g_xyz = torch.cat(g_xyz_list, dim=0)
    g_fea = torch.cat(g_fea_list, dim=0)

    with torch.no_grad():
        gaussians = GaussianModel(3)  # dataset.sh_degree = 3
    f_dc, f_rst, op, sc, ro = torch.split(g_fea, split_size_or_sections=[3, 45, 1, 3, 4], dim=-1)
    gaussians._xyz = nn.Parameter(g_xyz)
    gaussians._features_dc = nn.Parameter(f_dc.view(-1, 1, 3))
    gaussians._features_rest = nn.Parameter(f_rst.view(-1, 15, 3))
    gaussians._opacity = nn.Parameter(op.view(-1, 1))
    gaussians._scaling = nn.Parameter(sc.view(-1, 3))
    gaussians._rotation = nn.Parameter(ro.view(-1, 4))

    save_ply_fast(gaussians, args.ply_path_to)  # [topogs] same bytes as GaussianModel.save_ply, without the 6M-tuple Python list (host RAM)

    print(f"Decompressed ply file saved to {args.ply_path_to}!")


if __name__ == "__main__":
    parser = ArgumentParser(description="dataset_param")
    parser.add_argument("--lmd", default=1e-4, choices=[1e-4, 2e-4, 4e-4, 8e-4, 16e-4], type=float)
    parser.add_argument("--nr", default=3, type=float)
    parser.add_argument("--determ", default=1, type=float)
    parser.add_argument("--chunk_sizes", default="2000000,1000000,1000000", type=str)  # [topogs] fea,feq,geo arithmetic-coder chunk sizes
    parser.add_argument("--bit_path_from", default="./bitstreams/tmp/", type=str)
    parser.add_argument("--ply_path_to", default="./bitstreams/tmp/point_cloud.ply", type=str)
    args = parser.parse_args(sys.argv[1:])
    train(args)

