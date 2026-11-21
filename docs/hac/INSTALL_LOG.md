# HAC + Scaffold-GS install log (ada cluster)

Everything lives under `~/topogs-fcgs/hac/` on ada (env under `~/topogs-fcgs/envs/HAC_env`). The login node
is used only for clones and light file operations. Builds and runs go through sbatch on RTX 2080 Ti nodes.
Job scripts are copied in this folder (`docs/hac/*.sbatch`, `docs/hac/*.py`).

Approved protocol (user, 17 Nov): train from the full-size `images/` folder with the default `-r -1`. The loader
then downscales to 1600 px width (`utils/camera_utils.py`, both repos). Use `--eval` (llffhold 8). Do not use `-r 1`,
and do not switch to `images_4`.

Disk at start (17 Nov, 21:05 IST): `ada5:/home2/bala_skv 50G 47G 3.3G 94%`.

## 1. Clones (login node, 17 Nov ~21:13)
```
mkdir -p ~/topogs-fcgs/hac && cd ~/topogs-fcgs/hac
git clone --recursive https://github.com/YihangChen-ee/HAC.git
git clone --recursive https://github.com/city-super/Scaffold-GS.git
```
- HAC: `110dcf0a2d1603f65f94827fbdc58b9c9c3fcfbb` (2025-11-09, "Update README.md"), 23 MB. No git submodules.
  The extensions are zips in `submodules/`: `diff-gaussian-rasterization.zip`, `simple-knn.zip`, `gridencoder.zip`, `arithmetic.zip`.
- Scaffold-GS: `59c833b56bbbf510f3f64d40f81721995caced66` (2024-09-26, "Update README.md"), 20 MB.
  `submodules/diff-gaussian-rasterization` and `submodules/simple-knn` are plain directories (no `.gitmodules`).
  GitHub was slow from ada: about 2 min per clone.
- **One env serves both.** `diff -r` of HAC's unzipped `diff-gaussian-rasterization` and `simple-knn` against
  Scaffold-GS's directories shows them identical. The only difference is one whitespace change in
  `diff_gaussian_rasterization/__init__.py` (`return radii` vs `return  radii`), plus macOS `.DS_Store` / `__MACOSX` junk in the zips.
  Both are *modified* rasterizers: they add `GaussianRasterizer.visible_filter`, which `prefilter_voxel` calls.
  So FCGS_env's stock graphdeco rasterizer can run neither repo.
- Extra Python deps, all imported at module level: `torch_scatter` (both `scene/gaussian_model.py`), `einops`,
  `wandb` (both `train.py`, imported unconditionally), `colorama`, `cv2` (Scaffold-GS `scene/dataset_readers.py`),
  `jaxtyping` (Scaffold-GS `scene/embedding.py`). `laspy` is optional in Scaffold-GS (try/except).

## 2. Environment
### Attempt 1: conda clone (job 88356, gnode044). FAILED, no harm done
`conda create -p ~/topogs-fcgs/envs/HAC_env --clone ~/topogs-fcgs/envs/FCGS_env --offline`, with `CONDA_PKGS_DIRS=~/topogs-fcgs/conda_pkgs`.
- Failed with `OfflineError: EnforceUnusedAdapter called with url https://conda.anaconda.org/pytorch/noarch/pytorch-mutex-1.0-cuda.tar.bz2`
  (and the same error for every other package). Clone wants the package *tarballs*, and those were removed by
  `conda clean -t` during the FCGS install. Run online, it would re-download about 2+ GB of tarballs, which does not fit in ~3 GB free.
- Removed the partial `HAC_env` (117 MB). FCGS_env was not touched.

### Attempt 2: venv layered on FCGS_env (job 88357)
`~/topogs-fcgs/envs/FCGS_env/bin/python -m venv --system-site-packages ~/topogs-fcgs/envs/HAC_env`
- The venv sees FCGS_env's torch 1.12.1+cu116, numpy 1.21.6, Pillow 9.2.0, plyfile 0.8.1, lpips 0.1.4, scipy 1.7.3 and colorama 0.4.6 read-only.
  New packages go into the venv only.
- HAC's CUDA extensions are installed into the venv with `pip install --no-build-isolation --ignore-installed --no-deps`.
  They shadow FCGS's stock builds of the same module names (`diff_gaussian_rasterization`, `simple_knn`, `_gridencoder`, `arithmetic`),
  because venv site-packages come first on `sys.path`. The import test checks that every module's `__file__` is inside HAC_env,
  and that FCGS_env's own python still gets the stock rasterizer (no `visible_filter`).
- pip constraints `numpy==1.21.6 torch==1.12.1 torchvision==0.13.1 Pillow==9.2.0 scipy==1.7.3` keep the torch stack fixed.
- Toolchain is the same as FCGS: `module load u22/cuda/11.7` (nvcc 11.7), system gcc 11.4 (Ubuntu 22.04 node), `TORCH_CUDA_ARCH_LIST="7.5"`, `TMPDIR=~/topogs-fcgs/tmp`.
- Script: `build_hac_env.sbatch`.

- **Job 88357 (gnode044): partial failure.**
  - Extras installed into the venv: einops 0.6.1, wandb 0.18.7, opencv-python-headless 5.0.0.93, jaxtyping 0.2.12, plus wandb's deps
    (click, docker-pycreds, gitpython, gitdb, protobuf 4.24.4, psutil, pyyaml, sentry-sdk, setproctitle, smmap, typeguard, importlib-metadata, platformdirs, zipp).
    numpy and torch were not touched (constraints). The venv is 175 MB.
  - PyPI downloads from the compute node ran at only about 100 kB/s (wandb 16 MB, opencv 61 MB); this took about 15 min.
  - torch-scatter FAILED: `-f https://data.pyg.org/whl/torch-1.12.1+cu116.html` has no wheels (pyg publishes 1.12.x wheels under `torch-1.12.0+cu116`),
    so pip fell back to the sdist `torch_scatter-2.1.1.tar.gz` and its build failed.
    Fix: `--only-binary torch-scatter "torch-scatter==2.1.0+pt112cu116" -f https://data.pyg.org/whl/torch-1.12.0+cu116.html`.
  - All 4 extensions FAILED in about 5 s each: `RuntimeError: The detected CUDA version (11.7) mismatches the version that was used to compile PyTorch (11.6).`
    Cause: the venv ships its own setuptools 47.1.0, and with it torch 1.12's `_check_cuda_version` cannot read `.major`, so it raises instead of
    only warning about the minor mismatch (FCGS_env's setuptools 69.0.3 gives just a warning). Fix: `pip install setuptools==69.0.3` into the venv.
  - So the import test showed FCGS's stock modules (`<-- NOT IN HAC_env`, `has visible_filter: False`). It worked as a check.
    FCGS_env was confirmed unchanged (stock rasterizer, no `visible_filter`).
- Job 88374: torch-scatter is now OK (`torch-scatter 2.1.0+pt112cu116`, wheel from pyg). Extensions failed again with the same error:
  `pip install setuptools==69.0.3` gave `No matching distribution` (PyPI has no py3.7 build of 69.x; FCGS_env's copy came from conda-forge).
- Fix: `pip uninstall -y setuptools` inside the venv (removes only the venv's own 47.1.0), so FCGS_env's 69.0.3 is used: `setuptools 69.0.3 .../FCGS_env/...`.
- **Job 88375 (gnode044): OK.** Build times: diff-gaussian-rasterization 60 s, simple-knn 58 s, gridencoder 105 s, arithmetic 57 s, all exit 0
  (warnings only: CUDA 11.7 vs 11.6 minor mismatch, as for FCGS). In-tree `build/` dirs and the pip cache were removed afterwards.
  - Import test on the GPU (RTX 2080 Ti): IMPORT OK for `diff_gaussian_rasterization` (+`._C`), `simple_knn._C`, `_gridencoder`, `arithmetic`, `torch_scatter`, `einops`,
    `wandb`, `cv2`, `jaxtyping`, all from `.../envs/HAC_env/...`. `colorama` and `lpips` come from FCGS_env (unchanged versions, fine).
  - `has visible_filter: True`. The 100-Gaussian forward test returns 2 values (colour [3,64,64], radii [100]); `visible_filter radii>0: 100 / 100`.
  - `import scene, gaussian_renderer, utils.encodings` OK in HAC/; `import scene, gaussian_renderer` OK in Scaffold-GS/.
  - FCGS_env afterwards: `.../FCGS_env/.../diff_gaussian_rasterization/__init__.py has visible_filter: False`. Still the stock rasterizer, untouched.
  - HAC_env is 207 MB. Disk afterwards: 1.4 GB free.
- Final HAC_env: python 3.7.12 / torch 1.12.1+cu116 / torchvision 0.13.1 / numpy 1.21.6 (all from FCGS_env) + venv: torch-scatter 2.1.0+pt112cu116,
  einops 0.6.1, wandb 0.18.7, opencv-python-headless 5.0.0.93, jaxtyping 0.2.12, HAC's 4 extensions (version 0.0.0, built from HAC 110dcf0 zips).
- Use: `source ~/miniconda3/etc/profile.d/conda.sh; conda activate ~/topogs-fcgs/envs/FCGS_env; source ~/topogs-fcgs/envs/HAC_env/bin/activate`.

## 3. Data (17 Nov ~21:20-22:00)
- `scripts/fetch_bicycle_full.py` (copy in this folder) uses remotezip range requests into `360_v2.zip` and fetches `bicycle/images/*`, `bicycle/poses_bounds.npy` and `bicycle/sparse/0/*`.
  `zipfile` checks CRC32 on read; the script also re-checks `zlib.crc32` and the size against the central directory. On re-run, existing files are only CRC-checked.
- Fetching on ada was too slow: Google storage from ada ran at about 100–470 kB/s (5 images in ~4 min, about 2–5 h total), so I stopped it after 5 files.
  The laptop fetches at 3.5 MB/s (198 members in ~9 min, `all 198 members verified (CRC32 + size)`), and laptop→ada rsync runs at about 3.8 MB/s.
  So I fetched on the laptop (scratch dir) and ran `rsync -a --partial bicycle/ ada:topogs-fcgs/hac/data/bicycle/`.
- Verified on ada by re-running the script against `~/topogs-fcgs/hac/data` (no download, CRC-checks every file against the zip directory):
  `members 198 (images 194) total 1728.5 MB` / `all 198 members verified (CRC32 + size)`.
  Counts: 194 JPGs in `images/`, `sparse/0/{cameras.bin, images.bin, points3D.bin}`, `poses_bounds.npy`. 1,728,527,130 bytes.
- Test split: both repos sort the cameras by `image_name` and take `idx % 8 == 0` when `--eval` and `lod 0` (`scene/dataset_readers.py: readColmapSceneInfo`).
  Every 8th sorted name gives `_DSC8679 … _DSC8872` (25 views), **identical to `configs/test_views_sha256.txt`**.
  HAC's `ModelParams.eval` defaults to True anyway; Scaffold-GS's defaults to False, so `--eval` is needed there.
- Disk after data + venv: 1.4 GB free (`ada5:/home2/bala_skv 50G 49G 1.4G 98%`).

## 4. Smoke tests (2000 iterations)
Scripts: `run_train.sbatch` (`sbatch -J <name> run_train.sbatch hac|scaffold ITERS OUTNAME [extra args]`) and `memwrap.py`.
- `memwrap.py` runs the repo's `train.py` unchanged via `runpy`. It wraps `GaussianModel.update_learning_rate`, which is called once per iteration,
  to print `TOPOGS it= wall_s= s_per_it= anchors= alloc/max_alloc/max_reserved MiB` every 500 iterations, and prints the final torch peaks at exit.
- The job adds nvidia-smi sampling every 1 s, `/usr/bin/time -v` (host MaxRSS), `ulimit -n 4096`, `--mem=48G`, `WANDB_MODE=disabled` and `TORCH_HOME=~/topogs-fcgs/torch_cache`.
- Commands, exactly as the repos' scripts plus `--data_device cpu`:
  - HAC (`run_shell_mip360.py`): `python train.py -s DATA --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations N -m OUT --lmbda 0.004 --data_device cpu`
  - Scaffold-GS (`train_mip360.sh` → `train.sh`): `python train.py --eval -s DATA --lod 0 --gpu -1 --voxel_size 0.001 --update_init_factor 16 --appearance_dim 0 --ratio 1 --iterations N --port P -m OUT --data_device cpu`
- **SLURM limit:** account `cvit` has `GrpTRES gres/gpu=1` per user, so our GPU jobs run one at a time (the second shows `AssocGrpGRES`).
  Account `research` (qos medium) has no such group limit; jobs were kept on `cvit`.
- The anchor-growing schedule is the same in both repos: statistics from iteration 500 (`start_stat`), `adjust_anchor` every 100 iterations from 1500 (`update_from`) to 15000 (`update_until`).
  HAC pauses densification during 3000–3999. HAC's entropy/hash-grid branch only starts after step 10000. So a 2000-iteration test sees the start of growth only.

### HAC smoke 1 (job 88387, gnode044): training + codec OK, then crashed in the final render
- Image loading: 169 training cameras took about 1 min 50 s (full 4946x3286 JPG decode + resize to 1600x1062); host MaxRSS was 21.8 GB.
- From the TOPOGS lines:

| it | s/it | anchors | torch max_alloc MiB | max_reserved MiB |
|---|---|---|---|---|
| 500 | 0.072 | 53,005 | 690 | 1104 |
| 1000 | 0.061 | 53,005 | 690 | 1104 |
| 1500 | 0.060 | 53,005 | 690 | 1104 |
| 2000 | 0.058 | 96,636 | 765 | 1506 |

- Codec at iteration 2000 (`--test_iterations 2000`) worked:
  - `Encoded sizes in MB: anchor 0.553, feat 2.0314, scaling 0.8724, offsets 3.7871, hash 0.2114, masks 0.0022, MLPs 0.1573, Total 7.6149`.
  - Decode took 2.7 s. Test PSNR at 2k iterations was 19.25 (meaningless this early).
- The whole-process torch peak was **2796 MiB allocated / 3480 MiB reserved**. It came after training, in the evaluation/encode/render phase
  (probably LPIPS-VGG on the 1600x1062 test images).
- **Crash** in the end-of-training `render_sets` → `render_set` (train.py:423): `errormap = (rendering - gt).abs()` raised
  `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`
  `gt = view.original_image` is on `data_device` = cpu. Scaffold-GS's `train.py` has the same line. So `--data_device cpu` exists in the code, but the final render step does not support it.
- **Fix, output-neutral and with the repos untouched:** `memwrap.py` makes `Camera.original_image` a property. The image is stored on `data_device` (cpu)
  and returned as a cuda copy once the camera is constructed. Values are identical, and training already called `.cuda()` on it every iteration.
  - The alternative, `--data_device cuda`, would put about 3.9 GB of images on the GPU (194 × 1600×1062×3 float32).
- nvidia-smi `-l 1` lost its buffered tail when killed (reported peak 889 MiB < torch reserved). It is now replaced by a per-sample append loop.
- Output size at 2k iterations: `point_cloud.ply` 20 MB on disk for 96,636 anchors (about 210 B/anchor on the compressed home FS; raw 428 B), `checkpoint.pth` 3.7 MB, bitstreams 8.9 MB.
- The Scaffold-GS smoke job (88388) was cancelled after 1 min so that it could rerun with the fix. Failed outputs/logs: `~/topogs-fcgs/hac/logs/failed_smoke/`.
- Resubmitted: Scaffold-GS smoke 88400, then HAC smoke 88401.

### Scaffold-GS smoke (job 88400, gnode044): OK end to end
- Image loading took about 1 min 40 s for train and 15 s for test. Host MaxRSS was 20.6 GB. Wall time 425 s.

| it | s/it | anchors | torch max_alloc MiB | max_reserved MiB |
|---|---|---|---|---|
| 500 | 0.060 | 53,005 | 674 | 794 |
| 1000 | 0.058 | 53,005 | 674 | 796 |
| 1500 | 0.056 | 53,005 | 674 | 796 |
| 2000 | 0.053 | 96,050 | 776 | 1262 |

- The whole-process torch peak, **5186 MiB alloc / 5580 MiB reserved**, comes from the end-of-run `evaluate()`. It loads 25 renders and 25 gt images onto the GPU
  and runs LPIPS-VGG at 1600x1062, after training is over. Training itself peaked at 776 MiB.
- The final render with the `Camera.original_image` patch worked. Test (2k iterations): PSNR 21.50, SSIM 0.530, LPIPS 0.496.
- Output on disk 261 MB: `test/` 192 MB (25 × renders/errors/gt PNG), `backup/` 40 MB (the code copy made by `saveRuntimeCode`),
  `point_cloud/` 29 MB (96,050 anchors, 79 float props, plus 3 TorchScript MLPs, about 300 B/anchor on disk).
- Test PNGs are named `00000.png … 00024.png` in sorted-name order (`shuffle=False`), so `00000` = `_DSC8679`.

### HAC smoke 2 (job 88401, gnode044): OK end to end
- Same speed and memory as smoke 1 (2000 iterations: 0.0575 s/it, 96,567 anchors, training max_alloc 762 MiB).
  Whole-process peak **3620 MiB alloc / 4420 MiB reserved** (end-of-run evaluation). Host MaxRSS 22.5 GB. Wall time 517 s.
- Codec: `Encoded sizes in MB: anchor 0.5526, feat 2.036, scaling 0.8716, offsets 3.7848, hash 0.2114, masks 0.0022, MLPs 0.1573, Total 7.6159`. Decode took 2.7 s.
- **Check that the saved model is the decoded model.** In-memory test PSNR right after `conduct_decoding` (`[ITER 2000] Evaluating test`) was 19.19817.
  The end-of-run `render_set`, which reloads `point_cloud/iteration_2000` from disk with `decoded_version=True`, gave `testing_float_psnr = 19.1982`. They agree.
  (`evaluate()` on the 8-bit PNGs gives 19.1846.)
- Activation state in the saved plys:
  - HAC: 107 props, `scale_*` min 0 / max 14.2, none negative, so they are linear. That is consistent with decoded `_scaling` storing activated values.
    About 0.2% are exactly 0 after quantization. `f_mask_*` ∈ {0,1}. At 2k iterations every anchor had an active offset, so there were no zero-filled rows.
  - Scaffold-GS: 79 props, `scale_*` min −8.3 / max 2.7, 99.4% ≤ 0, so they are log-scale.
- Output 222 MB: `test/` 189 MB, `point_cloud/` 24 MB (about 210 B/anchor), `bitstreams/` 8.9 MB.

### Smoke-test verdict and memory extrapolation
- Training memory grows about 2.4 KB per anchor: +100 MiB for +43k anchors between iterations 1500 and 2000 in both repos.
  Even at 10× the anchor count (about 1M), that is about 3 GB for training, plus the 0.7 GB baseline. The end-of-run evaluation peak, 3.6–5.2 GB, runs after training.
  Both fit an 11 GB card with margin.
- **Not observed in the smoke tests:**
  - anchor growth from 2000 to 15000
  - HAC's entropy/hash-grid training branch, which starts after 10000
  - HAC's codec plus LPIPS at 30000, which runs while the full model and Adam state are still on the GPU
  The full runs log memory every 500 iterations, so these will show in the logs.
- Cleanup: deleted `test/` and `backup/` of both smoke outputs and the failed-smoke outputs. Kept the smoke `point_cloud/`, `cameras.json`, `cfg_args`, and HAC's `bitstreams/`
  (about 64 MB in total; they are small models for prototyping the export). Disk before the full runs: **1.3 GB free**.

## 5. Full runs (submitted 17 Nov ~22:20 IST)
- `sbatch -J hac_l0.004 run_train.sbatch hac 30000 hac_bicycle_l0.004` → **job 88440**, output `~/topogs-fcgs/hac/outputs/hac_bicycle_l0.004`.
- `sbatch -J scaffold_base run_train.sbatch scaffold 30000 scaffold_bicycle` → **job 88441**, output `~/topogs-fcgs/hac/outputs/scaffold_bicycle`.
- Both use `--time 08:00:00`, `--mem 48G`, one 2080 Ti, default `--test_iterations 30000` / `--save_iterations 30000`.
  HAC encodes, decodes, tests and renders at the end, as in its script.
  The GPU limit of 1 means 88441 waits (`AssocGrpGRES`) until 88440 finishes.
- After a successful exit, the job deletes `test/ours_30000/{gt,errors}`. `renders/` and `results.json` are kept. `CLEAN_PNG=0` disables this.
- Logs: `~/topogs-fcgs/hac/logs/train_<name>_<jobid>.log` and `gpumem_<out>_<jobid>.csv`.

### HAC full run 88440 (gnode044): progress (TOPOGS lines)
| it | s/it | anchors | torch max_alloc MiB | max_reserved MiB |
|---|---|---|---|---|
| 2000 | 0.058 | 96,878 | 762 | 1432 |
| 3000 | 0.062 | 190,624 | 1043 | 2058 |
| 5000 | 0.065 | 300,724 | 1319 | 3058 |
| 7000 | 0.080 | 461,330 | 1991 | 3642 |
| 9000 | 0.089 | 605,028 | 2587 | 4568 |
| 10000 | 0.094 | 669,188 | 2891 | 5018 |
| 10500 | 0.153 | 703,483 | 3044 | 5718 |
| 11500 | 0.160 | 767,578 | 3643 | 6380 |
- After 10000 the entropy/hash-grid branch is active: s/it jumps from 0.094 to 0.153, and memory rises faster.
  Anchors grow by about 65k per 1000 iterations until 15000.
| 12500 | 0.168 | 826,211 | 3940 | 6870 |
| 14000 | 0.178 | 908,567 | 4003 | 7384 |
| 15000 | 0.184 | 961,014 | 4205 | 8376 |
- Anchor growth ends at 15000 with **961,014 anchors**. Training peak: **4.2 GB torch allocated, 8.4 GB reserved** (2080 Ti has 10.57 GB usable).
  - The step still to come is the codec plus test evaluation at 30000 with the model resident. The current resident alloc is about 1.4 GB, and the smoke tests showed
    the evaluation adds about 2.8–4.5 GB. A rough estimate is about 6 GB allocated, which should fit. That estimate is unverified until the job ends.
- Time projection: 15000 iterations left at about 0.185–0.19 s/it is about 47 min. Training should end about 23:35 IST; then encode/decode, eval, a reload of all images
  (about 2 min) and render take about 10 min, so **HAC should finish about 23:45 IST (17 Nov)**.
  Scaffold-GS 88441 then starts. It has no entropy branch and similar anchor growth: about 0.06–0.12 s/it, so roughly 45–60 min plus about 8 min load/eval.
  **Expected finish about 00:45–01:00 IST (18 Nov).**
- Disk projection: HAC ply about 961k × 210 B ≈ 200 MB, bitstreams about 20–30 MB, test PNGs 190 MB (about 65 MB kept after cleanup).
  Scaffold-GS ply about 961k × 300 B ≈ 290 MB, backup 40 MB, test PNGs 190 MB. Peak while Scaffold-GS finishes is about 0.8 GB, against 1.3 GB free now.
- Disk at 22:46: 1.3 GB free.

### HAC full run 88440: FINISHED OK (17 Nov 23:34 IST, wall 4549 s = 76 min)
- 15000–30000 ran at about 0.168 s/it with anchors fixed at 961,014.
- Codec at 30000 (log line): `Encoded sizes in MB: anchor 4.971, feat 10.8682, scaling 3.4685, offsets 6.9225, hash 0.1255, masks 1.0394, MLPs 0.1573, Total 27.5525`.
  EncTime 13.4 s, DecTime 27.2 s. `bitstreams/` on disk is 33 MB (du; includes `_quantized_v.npy` and per-file overhead).
- Test metrics (25 views, 1600 px):
  - In-memory after decode: PSNR 25.0325, SSIM 0.7385, LPIPS 0.2700.
  - Reloaded from disk (`testing_float_psnr`): 25.0325, which matches.
  - From 8-bit PNGs (`evaluate`, results.json): PSNR 25.0209, SSIM 0.7382, LPIPS 0.2697.
  - For reference, HAC's paper reports bicycle at about 25.1 dB. Here bicycle was trained alone, at lambda 0.004.
- Whole-process torch peak **4560 MiB allocated / 8462 MiB reserved**. The codec and evaluation at 30000 fit. Host MaxRSS 24.9 GB.
- Output 463 MB, or 316 MB after the job deleted gt/errors: `point_cloud/` 223 MB (961k anchors), `bitstreams/` 33 MB, `test/ours_30000/renders` about 65 MB.
- Disk after: 1.4 GB free. Scaffold-GS 88441 started right after, at 23:34.

### Scaffold-GS full run 88441: FINISHED OK (18 Nov 00:18 IST, wall 2634 s = 44 min)
- Anchors: 418k at 6000, 786k at 12500, **891,555** at 15000 (final). s/it went 0.054 → 0.092 at 15000, then about 0.08.
  Training torch max_alloc was 2.76 GB.
- Test (25 views):
  - In-memory: PSNR 25.1231.
  - From 8-bit PNGs (results.json): PSNR 25.1120, SSIM 0.7447, LPIPS 0.2556.
- Whole-process torch peak **5393 MiB allocated / 9166 MiB reserved** (end-of-run evaluation). Host MaxRSS 23.0 GB.
- Output 520 MB, or 374 MB after the job deleted gt/errors: `point_cloud/` 269 MB (891,555 anchors + 3 TorchScript MLPs), `backup/` 40 MB, renders about 65 MB.
- Disk right after: 1.1 GB free.
- Comparison at the same views and protocol: HAC 25.03 dB / 0.738 / 0.270 at 27.55 MB vs Scaffold-GS 25.11 dB / 0.745 / 0.256 (uncompressed, 269 MB on disk incl. MLPs).

### Keeping outputs lean (coordinator request, 17 Nov 22:50)
- The coordinator freed space outside `hac/` (deleted the already-copied `results/render_all25/*.png` and `p512/fields/*.npy`), giving about 1.9 GB free.
  The request was to keep outputs small and copy finished outputs to the laptop before the second run fills the disk.
- Already in place: both full-run jobs delete `test/ours_30000/{gt,errors}` after a successful run (88441's stored batch script confirmed via `scontrol write batch_script`).
  The test renders cannot be skipped without editing `train.py`, because `render_sets` and `evaluate` run unconditionally at the end.
- Plan:
  - When HAC 88440 finishes, rsync `outputs/hac_bicycle_l0.004` and the logs to the laptop at `data/hac/` (git-ignored).
  - Check the copy, then delete only `test/ours_30000/renders` on ada.
  - `point_cloud/`, `bitstreams/`, `cameras.json` and `cfg_args` stay on ada, because the export needs the HAC_env GPU stack.
  - Do the same for Scaffold-GS 88441.
- **HAC copied to the laptop (17 Nov ~23:40):** `rsync -a ada:topogs-fcgs/hac/outputs/hac_bicycle_l0.004 data/hac/`, plus the log and gpumem csv into `data/hac/logs/`.
  - The sha256 of all 907 files matched between ada and the laptop.
  - `point_cloud/iteration_30000/point_cloud.ply`: 3b3cfa5ea7a0a75cbba5d01786af60a5b91e630ce0ed2f116e010203205758f0
  - `checkpoint.pth`: a5c103e8078b6a07efb70273b8533be0a4e170a364d3346c72d9b6efbdf2eb85
  - Then deleted `test/ours_30000/renders` on ada (now only on the laptop). The HAC output on ada is now 257 MB. Disk 1.5 GB free.
- **Scaffold-GS copied to the laptop (18 Nov ~00:25):** `data/hac/scaffold_bicycle` plus the log and csv in `data/hac/logs/`.
  - The sha256 of all 2079 files matched (backup/ included).
  - `point_cloud.ply`: bec4094f362d487f9088453af73459e915e864bb6f49f7645e7ccd047f2c8aaa
  - `opacity_mlp.pt`: f62ceeac..., `cov_mlp.pt`: 864d81f7..., `color_mlp.pt`: 384f4d2f...
  - Then deleted `test/ours_30000/renders` and `backup/` on ada (both now only on the laptop).
- **On ada now** (`~/topogs-fcgs/hac`, 1.8 GB total):
  - data 1.73 GB
  - `outputs/hac_bicycle_l0.004` 257 MB (point_cloud, bitstreams, cameras.json, cfg_args, results)
  - `outputs/scaffold_bicycle` 271 MB (point_cloud, cameras.json, cfg_args, results)
  - the two smoke models, 64 MB
  - **Home free: 1.2 GB.** No jobs running. Laptop `data/hac/` is 870 MB.

## 6. Code reading: how to export neural Gaussians to an INRIA-format .ply (nothing written yet)
Code versions: HAC 110dcf0, Scaffold-GS 59c833b.

### Generator: `generate_neural_gaussians(viewpoint_camera, pc, visible_mask=None, is_training=False[, step])` in `gaussian_renderer/__init__.py`
- Inputs per anchor *a* (visible subset): position `pc.get_anchor`, feature `pc._anchor_feat` (dim 32 in Scaffold-GS, 50 in HAC),
  `pc._offset` [N,10,3], `pc.get_scaling` [N,6] (exp of `_scaling`; HAC decoded: already activated, see below).
  HAC also uses `pc.get_mask` [N,10,1] (binary per-offset mask).
- Camera-dependent inputs: `ob_view = (anchor - cam_center)/|.|` and `ob_dist = |anchor - cam_center|`.
  - Scaffold-GS with the 360 settings (`add_opacity_dist = add_cov_dist = add_color_dist = False`, `appearance_dim 0`, `use_feat_bank False`):
    all three MLPs take `[feat, ob_view]`, so they depend on view direction but not distance.
  - HAC always takes `[feat, ob_view, ob_dist]`, so its MLPs depend on both direction and distance.
- Outputs per offset k (10 per anchor):
  - `neural_opacity = mlp_opacity(.)`, with Tanh output in (-1,1). HAC multiplies it by the binary offset mask.
  - Kept: `mask = neural_opacity > 0` (the opacity filter; this is the only threshold). **Which Gaussians exist depends on the camera.**
  - `color = mlp_color(.)`: Sigmoid RGB in (0,1), passed as `colors_precomp`. It is not SH, and the rasterizer is called with `shs=None`.
  - `scale_rot = mlp_cov(.)` (7 per offset). `scaling = scaling_repeat[:,3:] * sigmoid(scale_rot[:,:3])` (the anchor's scale columns 3:6 times a sigmoid),
    `rot = normalize(scale_rot[:,3:7])`.
  - `xyz = anchor + offset * scaling_repeat[:,:3]` (the anchor's scale columns 0:3). **Positions do not depend on the camera**; only whether each one survives the opacity mask does.
  - So opacity, scale, rotation and colour are all camera-dependent.
- The returned values are already activated (opacity in (0,1], linear scale, unit quaternion, RGB). To write the INRIA layout:
  `opacity = logit(o)`, `scale_i = log(s_i)`, `rot` as is (w,x,y,z order, same as INRIA), `f_dc = (rgb - 0.5)/C0` with C0 = 0.28209479177387814,
  `f_rest = 0` (45 zeros), normals 0.
- Call it with the model in eval mode (`gaussians.eval()`). In HAC, `render()` decides training vs eval from `pc.get_color_mlp.training`,
  and with `is_training=True` it adds quantization noise (step > 3000).

### Frustum culling / anchor visibility: `prefilter_voxel(cam, pc, pipe, bg)` (same file)
- Calls the modified rasterizer's `GaussianRasterizer.visible_filter(means3D=anchor, scales=get_scaling[:,:3], rotations=get_rotation)`.
  It returns radii per anchor, and `visible_mask = radii > 0` (anchor's own ellipsoid lands on screen).
- It is applied only because train/render pass `visible_mask=voxel_visible_mask` into `render()` → `generate_neural_gaussians`.
  **To disable it, call `generate_neural_gaussians(cam, pc, visible_mask=None)`.** With `None` it uses `torch.ones(N_anchor)`, so every anchor is decoded.
  The camera then only enters through `ob_view`/`ob_dist`. The rasterizer is not needed for export.

### Loading a trained model
- **Scaffold-GS** (`render.py: render_sets`): `GaussianModel(feat_dim, n_offsets, voxel_size, update_depth, update_init_factor, update_hierachy_factor, use_feat_bank, appearance_dim, ratio, add_opacity_dist, add_cov_dist, add_color_dist)`,
  then `Scene(dataset, gaussians, load_iteration=-1, shuffle=False)`, which calls `load_ply_sparse_gaussian(point_cloud/iteration_N/point_cloud.ply)` and
  `load_mlp_checkpoints(point_cloud/iteration_N/)` (TorchScript `opacity_mlp.pt`, `cov_mlp.pt`, `color_mlp.pt` via `torch.jit.load`), then `gaussians.eval()`.
  The args come from `<model>/cfg_args` (`get_combined_args`). `Scene` also loads every training image from `-s`. For export, it is cheaper to call the two
  load functions directly and build cameras from `<model>/cameras.json`, or from COLMAP for only the chosen test views.
- **HAC** has no `render.py`. Its loader is `train.py: render_sets`: `GaussianModel(..., n_features_per_level=4, log2_hashmap_size=13, log2_hashmap_size_2D=15, decoded_version=True)`,
  then `Scene(dataset, gaussians, load_iteration=-1, shuffle=False)`, which calls `load_ply_sparse_gaussian(.../point_cloud.ply)` + `load_mlp_checkpoints(.../checkpoint.pth)`
  (a plain `torch.save` dict: opacity_mlp, cov_mlp, color_mlp, encoding_xyz, grid_mlp, deform_mlp), then `gaussians.eval()`.
  HAC's `cfg_args` holds only the ModelParams group (feat_dim 50, n_offsets 10, voxel_size 0.001, ...). The hash-grid args (`--n_features 4 --log2 13 --log2_2D 15`)
  are train.py-level and are not saved, so an export script must use those defaults. They must match, or `load_mlp_checkpoints` fails on `encoding_xyz` shapes.

### HAC: the saved model IS the decoded model (important for us)
- `train.py` order at the last iteration: `training_report(...)` (line 220) runs first. At `iteration == test_iterations[-1]` (30000 by default) it calls
  `conduct_encoding` (writes `<model>/bitstreams/`), then `conduct_decoding`. `scene.save(iteration)` (line 223) only comes after that.
- `conduct_decoding` (scene/gaussian_model.py:1202) **replaces the model parameters with the decoded ones**:
  - `_anchor` comes from `_quantized_v.npy` × interval + `x_bound_min`.
  - `_anchor_feat`, `_scaling`, `_offset` come from the arithmetic-coded `.b` files.
  - `_mask` comes from `masks.b`, and the hash-grid params from `hash.b`.
  - It then sets `decoded_version=True`.
- In decoded mode `get_anchor` returns `_anchor` (no re-quantization), `get_scaling` returns `_scaling` as is (already linear, not log),
  and `get_mask` returns the binary `_mask`. `generate_neural_gaussians` then skips the quantization branch (`else: pass`).
- So `point_cloud/iteration_30000/point_cloud.ply` + `checkpoint.pth` written by `scene.save` contain the decoded tensors.
  **Verified in HAC smoke 2** (§4): the in-memory post-decode PSNR was 19.19817, and the reload from disk with `decoded_version=True` gave 19.1982. The saved `scale_*` values are linear (≥ 0) and `f_mask_*` ∈ {0,1}.
  The end-of-train `render_sets` loads them with `decoded_version=True`, and its test PSNR is the decoded PSNR.
  **The export must load with `decoded_version=True`.** With False, `get_scaling` would apply exp() a second time and `get_anchor` would re-quantize.
- Only the anchors with at least one active offset are coded (`mask_anchor`). At decode they are packed into rows `[:N]`, and rows `N:N_full` are zeros with
  mask 0. Those rows produce no Gaussians (opacity × mask = 0 fails the `> 0` filter).
- `x_bound_min/max` are not saved, but they are not needed in decoded mode (only the non-decoded path uses `calc_interp_feat`).
- Caveat to check after the run: the decoded params are the MLP inputs, but the MLPs themselves (`checkpoint.pth`) are stored as float32. That matches HAC's
  rate accounting ("MLPs" counted at 32 bit). An independent check would be to decode from `bitstreams/` into a fresh model and compare.
  That needs `patched_infos = [N_full, N, 3000]` and `x_bound_min/max`, neither of which is saved, so it is not trivially possible.

### Saved file layout
- Scaffold-GS `<model>/`: `point_cloud/iteration_30000/{point_cloud.ply (anchors: x,y,z,nx,ny,nz, f_offset_0..29, f_anchor_feat_0..31, opacity, scale_0..5, rot_0..3), opacity_mlp.pt, cov_mlp.pt, color_mlp.pt}`,
  `cameras.json`, `cfg_args`, `input.ply`, `outputs.log`, `backup/` (code copy), `test/ours_30000/{renders,errors,gt}/*.png`, `results.json`, `per_view.json`.
- HAC `<model>/`: `point_cloud/iteration_30000/{point_cloud.ply (anchors: x,y,z,nx,ny,nz, f_offset_0..29, f_mask_0..9, f_anchor_feat_0..49, opacity, scale_0..5, rot_0..3), checkpoint.pth}`,
  `bitstreams/{_quantized_v.npy, hash.b, masks.b, feat_<s>.b, scaling_<s>.b, offsets_<s>.b}` (s = batches of 3000 anchors), `cameras.json`, `cfg_args`, `input.ply`, `outputs.log`,
  `test/ours_30000/{renders,errors,gt}/*.png` + `per_view_count.json`, `results.json`, `per_view.json`.
  The rate is printed in `outputs.log` ("Encoded sizes in MB: ... Total"). Note that the anchor xyz (`_quantized_v.npy`) is not counted by `du` as an entropy-coded stream:
  HAC counts it as `N*3*anchor_round_digits` bits.

### Export pitfalls found while checking
- **Decoded HAC scales can be exactly 0.** In the smoke-2 decoded ply (96,567 anchors), the zero counts per column were `scale_0..5` = 104 / 135 / 115 / 219 / 190 / 179.
  - 402 anchors have a zero in `scale_3..5`. All of them have active offsets.
  - 326 anchors have a zero in `scale_0..2`. The smallest positive value is 4.98e-4.
  - Cause: the scaling quantization step is `Q_scaling = 0.001·(1+tanh(.))` ≤ 0.002, so small anchor scales round to 0.
  - Effect of a zero in columns 3:6: the child Gaussian's scale (`scale[3:6]·sigmoid(.)`) is 0 along that axis, a flat ellipsoid. INRIA `log(scale)` then gives −inf.
  - Effect of a zero in columns 0:3: the offsets collapse along that axis, so all 10 children sit in the anchor's plane. That is harmless for the export.
  - **The export needs a rule** for zero child scales: drop those Gaussians, or clamp to a small positive value (e.g. the smallest positive decoded scale). Record the choice under "Decisions".
    Re-count on the final 30k model; this was a 2k-iteration model.
- **`cfg_args` of these runs records `data_device='cpu'`.** Any repo script that uses `get_combined_args` will inherit it, and HAC/Scaffold-GS `render_set` then crash with the cuda/cpu
  mismatch described in §4. Pass `--data_device cuda` or run through `memwrap.py` (Camera patch).
- **Completion check for the full runs.** In both repos `scene.save(30000)` runs only after `training_report(30000)` (codec + test/train eval + LPIPS for HAC).
  So a crash in that final step means no model is saved. Scaffold-GS's last-iteration test block was not exercised by its smoke test (its `test_iterations` default is 30000).
  To check:
  - `grep -E "Encoded sizes|testing_float_psnr|EXIT=" ~/topogs-fcgs/hac/logs/train_<name>_<job>.log`
  - `ls ~/topogs-fcgs/hac/outputs/<out>/point_cloud/iteration_30000/`
  - `cat .../results.json`
