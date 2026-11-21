# FCGS install log (ada cluster)

All work under ~/topogs-fcgs/. The login node was used only for downloads/clones/light file ops; builds and runs via srun/sbatch.

## 2026-11-16 ~18:00 -- setup, download, clones (login node)
```
mkdir -p ~/topogs-fcgs/{third_party,data/bicycle,envs,results/fcgs,logs,scripts}
cd ~/topogs-fcgs/data/bicycle
wget -q -O point_cloud.ply https://huggingface.co/johnowhitaker/gaussian_splatter_models/resolve/main/bicycle/point_cloud/iteration_30000/point_cloud.ply
```
- Download took ~1 min (HF fast from ada). Size 1,520,726,124 bytes.
- `sha256sum point_cloud.ply` = 64d357cb25bd85f710f8551a18d830f8497277fbd8c5805adfd72ffe9ca78227 (matches expected).

```
cd ~/topogs-fcgs/third_party
git clone --recursive https://github.com/YihangChen-ee/FCGS.git
git clone https://github.com/MPEGGroup/mpeg-pcc-tmc13.git
```
- FCGS commit: 31e59a46f7e51505b46a7bbe6e53268ee3155fbf (2025-11-09)
  - submodule diff-gaussian-rasterization 59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d, glm 5c46b9c07008ae65cb81ab79cd677ecc1934b903
  - simple-knn, freqencoder, gridencoder, gridcreater, arithmetic are plain directories in the FCGS repo (not git submodules).
  - checkpoints/checkpoint_{0.0001,0.0002,0.0004,0.0008,0.0016}.pkl present (10,223,425 bytes each).
- tmc3: master HEAD a3d15c5e73bae20fbe2ec79be60994038a66dc8d = tag release-v23.0-rc2 (2023-09-27).
  - Checked that every CLI option FCGS passes in model/gpcc_utils.py (gpcc_encode/gpcc_decode) is defined in tmc3/TMC3.cpp of this version.
- Python module names to import (from setup.py / FCGS imports): `diff_gaussian_rasterization` (._C), `simple_knn._C`, `_freqencoder`, `_gridencoder`, `_gridcreater`, `arithmetic`.

(Session restart happened here; resumed, clones re-checked OK. Input sha256 re-verified at the end in job 85829 -- see below.)

## Compute node probe (job 85580, gnode004)
- Compute nodes: Ubuntu 22.04.5, system gcc/g++ 11.4.0, cmake module 3.31.7 works. (Login node is Rocky 9 / gcc 11.5 -- builds happen on compute nodes, so gcc 11.4 is what's used.)

## tmc3 build (job 85581 -> failed; job 85583 -> OK)
- Job 85581 (gnode012): CMake compiler test failed: `fatal error: error closing /tmp/ccBJv661.s: No space left on device`. Node-local /tmp on gnode012 is 100% full (1.8T disk, 0 free) -- not ours. Cancelled my job 85581.
- Workaround (all job scripts): `export TMPDIR=~/topogs-fcgs/tmp`.
- Job 85583: `module load u22/cmake/3.31.7; cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 8` -> built in 21 s.
  Binary: ~/topogs-fcgs/third_party/mpeg-pcc-tmc13/build/tmc3/tmc3 (put on PATH in job scripts; FCGS code unchanged).

## conda env
- Own yml: ~/topogs-fcgs/scripts/fcgs_env_base.yml (derived from FCGS environment.yml). Deviations:
  channels `pytorch, conda-forge, nodefaults` (dropped `pyg` = unused, dropped `defaults` to avoid Anaconda ToS prompts);
  local CUDA-extension pip entries removed (built separately, scripts/build_ext.sbatch).
  Env vars in jobs: CONDA_PKGS_DIRS=~/topogs-fcgs/conda_pkgs, PIP_CACHE_DIR=~/topogs-fcgs/pip_cache, TMPDIR=~/topogs-fcgs/tmp, CONDA_ALWAYS_YES=1 (keeps caches inside the project dir).
- Job 85583: `conda env create -p ~/topogs-fcgs/envs/FCGS_env -f scripts/fcgs_env_base.yml` FAILED in 17 s:
  `PackagesNotFoundError: python=3.7.13` (conda-forge's last 3.7 build is 3.7.12; 3.7.13 was only on `defaults`).
- Changed pin to python=3.7.12, resubmitted as job 85588.
- Job 85588 (gnode012, ~8 min): env created. Resolved: python 3.7.12 (conda-forge), pytorch 1.12.1 py3.7_cuda11.6_cudnn8.3.2_0 (pytorch ch.), torchvision 0.13.1 py37_cu116, cudatoolkit 11.6.2 (conda-forge), plyfile 0.8.1, numpy 1.21.6, lpips 0.1.4 + scipy 1.7.3 (pip).
  BUT `import torch` failed: `libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent` -- conda-forge resolved mkl 2025.3.1; mkl >= 2024.1 dropped that symbol (known torch<2.3 issue).
  Fix: `conda install -p envs/FCGS_env -c conda-forge --override-channels mkl=2024.0` (done at start of the extension job 85654), then `conda clean -p -t -y` with CONDA_PKGS_DIRS=~/topogs-fcgs/conda_pkgs (only my own cache) to free disk.
- Disk after env create: 6.1 GB free (conda_pkgs 6.3 GB incl. tarballs; env itself hard-links into it).

## CUDA extensions + verification (job 85654, gnode046, RTX 2080 Ti, 40 min total)
Script: scripts/build_ext.sbatch. Toolchain: `module load u22/cuda/11.7` (nvcc V11.7.64, CUDA_HOME=/data3/u22/cuda-11.7), system gcc 11.4.0 (Ubuntu 22.04 node), python 3.7.12, torch 1.12.1+cu116.
```
export CUDA_HOME=$(dirname $(dirname $(command -v nvcc))) TORCH_CUDA_ARCH_LIST="7.5;8.6" MAX_JOBS=8 TMPDIR=~/topogs-fcgs/tmp
pip install -v --no-build-isolation submodules/<ext>     # per extension, logs in logs/ext_<ext>.log
```
- mkl fix applied first: mkl 2025.3.1 -> 2024.0.0; `import torch` OK afterwards. conda clean freed ~2 GB (8.2 GB free after).
- Build times: diff-gaussian-rasterization 87 s, simple-knn 81 s, freqencoder 81 s (no issue #21 failure with --no-build-isolation), gridencoder 236 s, gridcreater 1683 s (!! 28 min: many template instantiations x 2 archs), arithmetic 83 s. All exit 0.
- Only warnings: "CUDA version (11.7) has a minor version mismatch with ... PyTorch (11.6)" and "no g++ version bounds defined for CUDA 11.7" -- harmless.
- pip builds in-tree, so untracked `build/` and `*.egg-info` dirs now exist under third_party/FCGS/submodules/*/ (no tracked file changed).
- Import test (GPU job): torch 1.12.1 cuda 11.6 is_available=True, NVIDIA GeForce RTX 2080 Ti. IMPORT OK: diff_gaussian_rasterization, diff_gaussian_rasterization._C, simple_knn._C, _freqencoder, _gridencoder, _gridcreater, arithmetic; `import model.FCGS_model` OK (from FCGS root).
- `command -v tmc3` -> ~/topogs-fcgs/third_party/mpeg-pcc-tmc13/build/tmc3/tmc3 (with PATH set in job).
- diff_gaussian_rasterization (for later rendering use): stock graphdeco-inria/diff-gaussian-rasterization, commit 59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d (2023-08-27, "Added submodule license"), package version 0.0.0.
  GaussianRasterizationSettings fields: image_height, image_width, tanfovx, tanfovy, bg, scale_modifier, viewmatrix, projmatrix, sh_degree, campos, prefiltered, debug.
  GaussianRasterizer.forward returns 2 values (color [3,H,W], radii [N]) -- verified with a 100-Gaussian test render. No depth/alpha output.

## Smoke test lambda 1e-4, attempt 1 (job 85748, gnode046, RTX 2080 Ti) -- CUDA OOM
- Stock `encode_single_scene.py --lmd 0.0001 --determ 1` (cwd = FCGS root, tmc3 on PATH). GPCC (tmc3) part worked (bits/0.0001/0/xyz_gpcc.bin 2.0 MB + mask.b written), then:
  `RuntimeError: CUDA out of memory. Tried to allocate 548.00 MiB (GPU 0; 10.57 GiB total capacity; 8.70 GiB already allocated; 96.06 MiB free; 9.74 GiB reserved in total by PyTorch)` in FCGS_model.compress line 431 (context_analyzer_fea), step 0, after 34 s.
- nvidia-smi showed ~4.1 GB in use before compress starts: the whole 6.1M scene is held on GPU twice (GaussianModel params + the g_fea concat).
- Failed outputs moved to logs/failed_85748_lmd_0.0001_2080ti_OOM/.
- 3080 Ti is not an option: `-C 3080ti` -> "Requested node configuration is not available" (3080 Ti nodes are only in partition `ihub`, whose AllowAccounts excludes our accounts cvit/research). 3090s (plafnet2/rrc) not used (need to report back first).

## Workaround: memory-lean copies of the two driver scripts (FCGS repo itself untouched)
- ~/topogs-fcgs/scripts/encode_fcgs.py and decode_fcgs.py (copies of encode_single_scene.py / decode_single_scene.py), run with cwd = FCGS root and PYTHONPATH=FCGS root.
  - encode: full scene kept on CPU, only one step (<= per_step_size Gaussians) moved to GPU at a time; GaussianModel deleted + empty_cache. Output-neutral (same tensors go into CM.compress). New flag --per_step_size (default 1,000,000 = stock). Prints torch max_memory_allocated/reserved.
  - decode: each step's decoded tensors moved to CPU right away; PLY written with a vectorized equivalent of GaussianModel.save_ply (same dtype/field order/bytes, avoids a 6M-row Python tuple list that would need ~12 GB host RAM).
  - Exact diffs: scripts/encode_fcgs.diff, scripts/decode_fcgs.diff (reproduced below).
- Run script scripts/run_lambda.sbatch LAMBDA [sizes...]: tries per_step_size values in order (stock 1,000,000 first) until encode+decode both pass; nvidia-smi sampled every 200 ms.
- Attempt 2: job 85758 (`sbatch -C 2080ti scripts/run_lambda.sbatch 0.0001 1000000 500000`).

### encode_fcgs.diff
```diff
--- enc_orig.py	2026-11-16 19:41:25.758019311 +0530
+++ encode_fcgs.py	2026-11-16 19:41:28.105669298 +0530
@@ -18,9 +18,10 @@
     g_xyz = gaussians._xyz.detach()
     N_gaussian = g_xyz.shape[0]
 
-    per_step_size = 100_0000
-    if N_gaussian > 100_0000 and N_gaussian < 110_0000:
+    per_step_size = int(args.per_step_size)  # [topogs] was hard-coded 100_0000
+    if per_step_size == 100_0000 and N_gaussian > 100_0000 and N_gaussian < 110_0000:
         per_step_size = 110_0000
+    print('per_step_size:', per_step_size)
 
     _features_dc = gaussians._features_dc.detach().view(N_gaussian, -1)  # [N, 1, 3] -> [N, 3]
     _features_rest = gaussians._features_rest.detach().view(N_gaussian, -1)  # [N, 15, 3] -> [N, 45]
@@ -28,6 +29,10 @@
     _scaling = gaussians._scaling.detach()  # [N, 3]
     _rotation = gaussians._rotation.detach()  # [N, 4]
     g_fea = torch.cat([_features_dc, _features_rest, _opacity, _scaling, _rotation], dim=-1)  # [N, 56]
+    # [topogs] VRAM: keep the full scene on CPU, move one step at a time to the GPU (output-neutral)
+    g_xyz = g_xyz.cpu(); g_fea = g_fea.cpu()
+    del gaussians, _features_dc, _features_rest, _opacity, _scaling, _rotation
+    torch.cuda.empty_cache()
 
     step_num = int(np.ceil(N_gaussian / per_step_size))
     lmd = args.lmd
@@ -48,11 +53,12 @@
         for s in range(step_num):
             bit_save_path = os.path.join(args.bit_path_to, f"{lmd}/{s}")
             os.makedirs(bit_save_path, exist_ok=True)
-            g_xyz_in = g_xyz[s*per_step_size:s*per_step_size+per_step_size]
-            g_fea_in = g_fea[s*per_step_size:s*per_step_size+per_step_size]
+            g_xyz_in = g_xyz[s*per_step_size:s*per_step_size+per_step_size].cuda()
+            g_fea_in = g_fea[s*per_step_size:s*per_step_size+per_step_size].cuda()
             ttl_size += CM.compress(g_xyz_in, g_fea_in, root_path=bit_save_path, chunk_size_list=chunk_size_list, determ_codec=args.determ)[3]
     torch.cuda.synchronize(); t2 = time.time()
     print('time:', t2-t1)
+    print('max_memory_allocated_MiB: %.0f  max_memory_reserved_MiB: %.0f' % (torch.cuda.max_memory_allocated()/2**20, torch.cuda.max_memory_reserved()/2**20))
 
     print(f"{args.ply_path_from} compressed! Save bitstreams to {args.bit_path_to}.")
     orig_size = os.path.getsize(args.ply_path_from)/1024/1024
@@ -65,6 +71,7 @@
     parser.add_argument("--nr", default=3, type=float)
     parser.add_argument("--determ", default=1, type=float)
     parser.add_argument("--bit_path_to", default="./bitstreams/tmp/", type=str)
+    parser.add_argument("--per_step_size", default=100_0000, type=int)  # [topogs]
     parser.add_argument("--ply_path_from", default="./xxx/point_cloud.ply", type=str)
     args = parser.parse_args(sys.argv[1:])
     train(args)
```
### decode_fcgs.diff
```diff
--- dec_orig.py	2026-11-16 19:41:28.090989555 +0530
+++ decode_fcgs.py	2026-11-16 19:41:28.105714933 +0530
@@ -9,6 +9,25 @@
 import sys
 from typing import NamedTuple
 
+def save_ply_fast(g, path):
+    import numpy as np
+    from plyfile import PlyData, PlyElement
+    os.makedirs(os.path.dirname(path), exist_ok=True)
+    xyz = g._xyz.detach().cpu().numpy()
+    normals = np.zeros_like(xyz)
+    f_dc = g._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
+    f_rest = g._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
+    opacities = g._opacity.detach().cpu().numpy()
+    scale = g._scaling.detach().cpu().numpy()
+    rotation = g._rotation.detach().cpu().numpy()
+    names = g.construct_list_of_attributes()
+    elements = np.empty(xyz.shape[0], dtype=[(a, 'f4') for a in names])
+    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
+    for i, a in enumerate(names):
+        elements[a] = attributes[:, i]
+    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)
+
+
 def train(args):
 
     step_num = len(os.listdir(os.path.join(args.bit_path_from, str(args.lmd))))
@@ -30,8 +49,11 @@
         for s in range(step_num):
             bit_save_path = os.path.join(args.bit_path_from, f"{lmd}/{s}")
             g_xyz_out, g_fea_out = CM.decomprss(root_path=bit_save_path, chunk_size_list=chunk_size_list)
-            g_xyz_list.append(g_xyz_out)
-            g_fea_list.append(g_fea_out)
+            g_xyz_list.append(g_xyz_out.cpu())  # [topogs] VRAM: collect step outputs on CPU (output-neutral)
+            g_fea_list.append(g_fea_out.cpu())
+            del g_xyz_out, g_fea_out
+            torch.cuda.empty_cache()
+    print('decode loop done; max_memory_allocated_MiB: %.0f  max_memory_reserved_MiB: %.0f' % (torch.cuda.max_memory_allocated()/2**20, torch.cuda.max_memory_reserved()/2**20))
             
     g_xyz = torch.cat(g_xyz_list, dim=0)
     g_fea = torch.cat(g_fea_list, dim=0)
@@ -46,7 +68,7 @@
     gaussians._scaling = nn.Parameter(sc.view(-1, 3))
     gaussians._rotation = nn.Parameter(ro.view(-1, 4))
 
-    gaussians.save_ply(args.ply_path_to)
+    save_ply_fast(gaussians, args.ply_path_to)  # [topogs] same bytes as GaussianModel.save_ply, without the 6M-tuple Python list (host RAM)
 
     print(f"Decompressed ply file saved to {args.ply_path_to}!")
 
```

## Smoke test attempt 2 (job 85758, gnode046, 2080 Ti) -- still OOM
- encode_fcgs.py at per_step_size 1,000,000: OOM `Tried to allocate 1.61 GiB (... 7.72 GiB already allocated; 990 MiB free; 8.87 GiB reserved)` at FCGS_model.py:437 (Decoder_fea_hyp) -- got past line 431 now, but model intermediates of a 1M step alone exceed 10.5 GB.
- per_step_size 500,000: OOM `Tried to allocate 1.51 GiB` in encodings_cuda.encoder_gaussian_mixed (CDF tensor of size chunk x symbol-range, feq stage, chunk_size 1,000,000).

## Second driver knob: arithmetic-coder chunk sizes (--chunk_sizes, must match in decode)
- Added `--chunk_sizes fea,feq,geo` to both copies (stock 2000000,1000000,1000000). Chunk size only splits the symbol stream into more independently-coded chunks with the same probabilities -> reconstruction identical, rate changes only by tiny per-chunk headers.
- Also `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256` (allocator only).
- driver_params.txt (per_step_size, chunk_sizes) is written into each results/fcgs/lmd_*/ dir -- decoding these bitstreams requires `--chunk_sizes 500000,250000,250000`.

## Smoke test attempt 3 (job 85762, gnode044, 2080 Ti) -- WORKS at per_step_size 500,000
`CHUNKS=500000,250000,250000 sbatch --export=ALL -C 2080ti scripts/run_lambda.sbatch 0.0001 1000000 500000`
- per_step_size 1,000,000 + small chunks: still OOM (`Tried to allocate 412 MiB; 9.62 GiB already allocated`). => stock step size cannot run on 11 GB 2080 Ti.
- per_step_size 500,000: encode OK (FCGS time 150.0 s, wall 170 s incl. ply load; 13 steps), decode OK (wall 195 s).
  torch max_memory_allocated: encode 7718 MiB, decode 3608 MiB; reserved 10068 MiB; nvidia-smi peak 9989 / 9995 MiB. Host max RSS: encode 6.7 GB, decode 9.2 GB.
  bits: 116,085,955 bytes (du -sb, 2867 files). FCGS print: "Original size: 1450.2774 MB. Compressed size: 110.3723 MB. Compression ratio: 13.1399 X".
  Decoded: 6,131,954 vertices, same property names/order/dtypes (incl. nx,ny,nz), 0 NaN/Inf. Opacity still logits, scales still log.
  NOTE: decoded vertex ORDER differs from input (FCGS Morton-sorts + randperm-shuffles each step); no per-index correspondence with the original.
- Consequence of per_step_size 500k: per-step context models / GPCC voxel bbox see half as many points -> rates (and xyz voxel grid) not identical to the paper's stock config (1M). Feature quantization per Gaussian is unaffected.
- Disk after: 6.9 GB free.
- Lambda 4e-4: job 85773 (same settings, per_step_size 500000 only).

## Lambda 4e-4 (job 85773, gnode044, 2080 Ti) -- OK
`CHUNKS=500000,250000,250000 sbatch --export=ALL -C 2080ti scripts/run_lambda.sbatch 0.0004 500000`
- encode wall 126 s (FCGS time 107.5 s), decode wall 154 s. torch max_alloc enc 8690 / dec 3972 MiB; nvidia-smi peak 9971 / 9979 MiB.
- bits 89,465,360 bytes (3112 files); FCGS print: 84.9847 MB, ratio 17.0652 X.
- Decoded: 6,131,954 vertices, header identical, 0 NaN/Inf.
- Note: home FS is compressed -- the 1.52 GB decoded ply takes ~1.1 GB on disk (du). Free after: 5.8 GB.
- Lambda 16e-4: job 85789.
- decode_single_scene_validate.py: SKIPPED -- it builds `Scene(dataset)` from `--source_path` (needs the Mip-NeRF 360 bicycle images + COLMAP) to render test views.

## Lambda 16e-4, attempt 1 (job 85789, gnode043) -- OOM at per_step_size 500k
- OOM at step 8 in FCGS_model.py:438 (fea_channel_ctx): `Tried to allocate 454 MiB; 9.10 GiB already allocated`. At higher lambda more Gaussians take the latent "fea" path, so model memory per step grows. Log: logs/encode_failed_85789_lmd0.0016_step500000.log.

## FINAL SETTINGS: per_step_size 400,000, chunk_sizes 500000,250000,250000 for all three lambdas
- 16e-4 at 400k (job 85797, gnode046): OK. Then re-ran 1e-4 (job 85803) and 4e-4 (job 85804) at 400k so that all lambdas use identical settings. The earlier 500k results are superseded (logs kept in logs/superseded_step500k/).
- Decoded attribute stats (opacity/scale/f_dc/rot min/median/max) are identical between the 500k and 400k runs -> per_step_size changes only the rate/context, not the per-Gaussian reconstruction of features.

| lambda | job | bits bytes (du -sb) | files | steps | encode wall (FCGS time) | decode wall | torch max_alloc enc/dec MiB | nvidia-smi peak enc/dec MiB | decoded N | header identical | NaN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1e-4  | 85803 | 116,676,204 | 2865 | 16 | 155 s (137.8 s) | 180 s | 6424 / 3269 | 9983 / 9995 | 6,131,954 | yes | 0 |
| 4e-4  | 85804 |  90,037,975 | 3228 | 16 | 117 s (100.7 s) | 153 s | 7184 / 3417 | 9989 / 9977 | 6,131,954 | yes | 0 |
| 16e-4 | 85797 |  67,272,780 | 3480 | 16 |  96 s (78.8 s)  | 130 s | 8088 / 3754 | 9963 / 9829 | 6,131,954 | yes | 0 |
(input ply 1,520,726,124 bytes -> ratios 13.0x / 16.9x / 22.6x by du -sb.) nvidia-smi peak ~10 GB = caching-allocator reservation; torch max_alloc is the real tensor peak. Host max RSS: encode 6.7 GB, decode 9.2 GB.

Decoded opacity (logit) min/median/max -- orig: -7.1436 / -1.9581 / 18.3231
- 1e-4: -7.1120 / -1.9317 / 18.3507;  4e-4: -7.1884 / -1.9286 / 18.4094;  16e-4: -6.9767 / -2.0930 / 18.4882
Decoded scale_0 (log) min/median/max -- orig: -18.0090 / -5.4094 / 2.0008
- 1e-4: -18.0098 / -5.3921 / 1.9951;  4e-4: -18.0215 / -5.4232 / 2.0024;  16e-4: -17.9931 / -5.3567 / 2.0603
mean sigmoid(opacity): orig 0.3517, decoded 0.3517 / 0.3517 / 0.3518.

sha256 of decoded plys:
- lmd_0.0001: 88fc6948cb2334dbfd553edf7043657fbb7f4fadd11b8f92fe129299d9ea8e79
- lmd_0.0004: 230958e51bed55b1141922b709bda316bc5aecf27d787be9fd5d747f90aeb361
- lmd_0.0016: 8e64fc582f265e8ec235f5b14cab216b6c1844b0fdb5d7aa6048d521af005c91

### Caveat: FCGS's printed "Compressed size" is wrong on this NFS home
- FCGS printed 97.90 / 72.50 / 50.79 MB (MiB), but on-disk bitstreams are 111.27 / 85.87 / 64.16 MiB.
  For 1e-4: 116,673,286 B total - 14,014,914 B xyz_gpcc.bin = 102,658,372 B = 97.90 MiB -> the GPCC xyz part is (almost) not counted.
  Probable cause (plausible, not tested): model/gpcc_utils.py compress_gaussian_params calls os.path.getsize(bin_path) while the file is still open for writing (inside `with open(...)`), so on NFS the size is stale/partial. The bitstream files themselves are complete (decode works). Use `du -sb bits/` as the true size.
- Component breakdown, 1e-4: xyz_gpcc 14.0 MB, mask 0.74 MB, fea (latent) 11.4 MB, geo (opacity/scale/rot) 44.3 MB, rest = fe_q (SH of non-latent Gaussians).

### Other notes
- Only warning in encode/decode logs: torch "nn.functional.tanh is deprecated". No NaN/Inf anywhere.
- Decoded Gaussian order is NOT the input order (per-step Morton sort + randperm(seed 1)); identical across lambdas (same first xyz rows).
- Cleanup: removed my in-tree build dirs (submodules/*/build) and pip_cache after install; `git status` in FCGS shows only untracked __pycache__ / *.egg-info, no tracked file modified.

### FINAL driver diffs (supersede the earlier diffs above; also in scripts/*.diff)
```diff
--- enc_orig.py	2026-11-16 19:41:25.758019311 +0530
+++ encode_fcgs.py	2026-11-16 19:44:06.389741520 +0530
@@ -18,9 +18,10 @@
     g_xyz = gaussians._xyz.detach()
     N_gaussian = g_xyz.shape[0]
 
-    per_step_size = 100_0000
-    if N_gaussian > 100_0000 and N_gaussian < 110_0000:
+    per_step_size = int(args.per_step_size)  # [topogs] was hard-coded 100_0000
+    if per_step_size == 100_0000 and N_gaussian > 100_0000 and N_gaussian < 110_0000:
         per_step_size = 110_0000
+    print('per_step_size:', per_step_size)
 
     _features_dc = gaussians._features_dc.detach().view(N_gaussian, -1)  # [N, 1, 3] -> [N, 3]
     _features_rest = gaussians._features_rest.detach().view(N_gaussian, -1)  # [N, 15, 3] -> [N, 45]
@@ -28,10 +29,15 @@
     _scaling = gaussians._scaling.detach()  # [N, 3]
     _rotation = gaussians._rotation.detach()  # [N, 4]
     g_fea = torch.cat([_features_dc, _features_rest, _opacity, _scaling, _rotation], dim=-1)  # [N, 56]
+    # [topogs] VRAM: keep the full scene on CPU, move one step at a time to the GPU (output-neutral)
+    g_xyz = g_xyz.cpu(); g_fea = g_fea.cpu()
+    del gaussians, _features_dc, _features_rest, _opacity, _scaling, _rotation
+    torch.cuda.empty_cache()
 
     step_num = int(np.ceil(N_gaussian / per_step_size))
     lmd = args.lmd
-    chunk_size_list = [200_0000, 100_0000, 100_0000]
+    chunk_size_list = [int(v) for v in args.chunk_sizes.split(',')]  # [topogs] was hard-coded [200_0000, 100_0000, 100_0000]; must match between encode and decode
+    print('chunk_size_list:', chunk_size_list)
 
     CM = FCGS(
         Q=1,
@@ -48,11 +54,12 @@
         for s in range(step_num):
             bit_save_path = os.path.join(args.bit_path_to, f"{lmd}/{s}")
             os.makedirs(bit_save_path, exist_ok=True)
-            g_xyz_in = g_xyz[s*per_step_size:s*per_step_size+per_step_size]
-            g_fea_in = g_fea[s*per_step_size:s*per_step_size+per_step_size]
+            g_xyz_in = g_xyz[s*per_step_size:s*per_step_size+per_step_size].cuda()
+            g_fea_in = g_fea[s*per_step_size:s*per_step_size+per_step_size].cuda()
             ttl_size += CM.compress(g_xyz_in, g_fea_in, root_path=bit_save_path, chunk_size_list=chunk_size_list, determ_codec=args.determ)[3]
     torch.cuda.synchronize(); t2 = time.time()
     print('time:', t2-t1)
+    print('max_memory_allocated_MiB: %.0f  max_memory_reserved_MiB: %.0f' % (torch.cuda.max_memory_allocated()/2**20, torch.cuda.max_memory_reserved()/2**20))
 
     print(f"{args.ply_path_from} compressed! Save bitstreams to {args.bit_path_to}.")
     orig_size = os.path.getsize(args.ply_path_from)/1024/1024
@@ -64,7 +71,9 @@
     parser.add_argument("--lmd", default=1e-4, choices=[1e-4, 2e-4, 4e-4, 8e-4, 16e-4], type=float)
     parser.add_argument("--nr", default=3, type=float)
     parser.add_argument("--determ", default=1, type=float)
+    parser.add_argument("--chunk_sizes", default="2000000,1000000,1000000", type=str)  # [topogs] fea,feq,geo arithmetic-coder chunk sizes
     parser.add_argument("--bit_path_to", default="./bitstreams/tmp/", type=str)
+    parser.add_argument("--per_step_size", default=100_0000, type=int)  # [topogs]
     parser.add_argument("--ply_path_from", default="./xxx/point_cloud.ply", type=str)
     args = parser.parse_args(sys.argv[1:])
     train(args)
```
```diff
--- dec_orig.py	2026-11-16 19:41:28.090989555 +0530
+++ decode_fcgs.py	2026-11-16 19:44:06.389923557 +0530
@@ -9,11 +9,31 @@
 import sys
 from typing import NamedTuple
 
+def save_ply_fast(g, path):
+    import numpy as np
+    from plyfile import PlyData, PlyElement
+    os.makedirs(os.path.dirname(path), exist_ok=True)
+    xyz = g._xyz.detach().cpu().numpy()
+    normals = np.zeros_like(xyz)
+    f_dc = g._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
+    f_rest = g._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
+    opacities = g._opacity.detach().cpu().numpy()
+    scale = g._scaling.detach().cpu().numpy()
+    rotation = g._rotation.detach().cpu().numpy()
+    names = g.construct_list_of_attributes()
+    elements = np.empty(xyz.shape[0], dtype=[(a, 'f4') for a in names])
+    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
+    for i, a in enumerate(names):
+        elements[a] = attributes[:, i]
+    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)
+
+
 def train(args):
 
     step_num = len(os.listdir(os.path.join(args.bit_path_from, str(args.lmd))))
     lmd = args.lmd
-    chunk_size_list = [200_0000, 100_0000, 100_0000]
+    chunk_size_list = [int(v) for v in args.chunk_sizes.split(',')]  # [topogs] was hard-coded [200_0000, 100_0000, 100_0000]; must match between encode and decode
+    print('chunk_size_list:', chunk_size_list)
 
     CM = FCGS(
         Q=1,
@@ -30,8 +50,11 @@
         for s in range(step_num):
             bit_save_path = os.path.join(args.bit_path_from, f"{lmd}/{s}")
             g_xyz_out, g_fea_out = CM.decomprss(root_path=bit_save_path, chunk_size_list=chunk_size_list)
-            g_xyz_list.append(g_xyz_out)
-            g_fea_list.append(g_fea_out)
+            g_xyz_list.append(g_xyz_out.cpu())  # [topogs] VRAM: collect step outputs on CPU (output-neutral)
+            g_fea_list.append(g_fea_out.cpu())
+            del g_xyz_out, g_fea_out
+            torch.cuda.empty_cache()
+    print('decode loop done; max_memory_allocated_MiB: %.0f  max_memory_reserved_MiB: %.0f' % (torch.cuda.max_memory_allocated()/2**20, torch.cuda.max_memory_reserved()/2**20))
             
     g_xyz = torch.cat(g_xyz_list, dim=0)
     g_fea = torch.cat(g_fea_list, dim=0)
@@ -46,7 +69,7 @@
     gaussians._scaling = nn.Parameter(sc.view(-1, 3))
     gaussians._rotation = nn.Parameter(ro.view(-1, 4))
 
-    gaussians.save_ply(args.ply_path_to)
+    save_ply_fast(gaussians, args.ply_path_to)  # [topogs] same bytes as GaussianModel.save_ply, without the 6M-tuple Python list (host RAM)
 
     print(f"Decompressed ply file saved to {args.ply_path_to}!")
 
@@ -56,6 +79,7 @@
     parser.add_argument("--lmd", default=1e-4, choices=[1e-4, 2e-4, 4e-4, 8e-4, 16e-4], type=float)
     parser.add_argument("--nr", default=3, type=float)
     parser.add_argument("--determ", default=1, type=float)
+    parser.add_argument("--chunk_sizes", default="2000000,1000000,1000000", type=str)  # [topogs] fea,feq,geo arithmetic-coder chunk sizes
     parser.add_argument("--bit_path_from", default="./bitstreams/tmp/", type=str)
     parser.add_argument("--ply_path_to", default="./bitstreams/tmp/point_cloud.ply", type=str)
     args = parser.parse_args(sys.argv[1:])
```

Re-run recipe: `CHUNKS=500000,250000,250000 sbatch --export=ALL -C 2080ti scripts/run_lambda.sbatch <0.0001|0.0004|0.0016> 400000`

## Final verification (job 85829, CPU only, srun -c1 --mem=8G)
- Input sha256 re-checked: 64d357cb25bd85f710f8551a18d830f8497277fbd8c5805adfd72ffe9ca78227 (unchanged).
- Raw PLY header bytes (up to end_header, 1532 bytes, no comment lines) byte-identical to the original for all three decoded plys.
- Decoded xyz arrays (values AND order) are exactly equal across the three lambdas (np.array_equal) -> free per-index correspondence between lambdas (but not with the input order).
