# Environments, machines and data

## Machines

| Machine | Hardware | Used for |
|---|---|---|
| Laptop | 8-core CPU, 31 GB RAM, no NVIDIA GPU (Linux, Python 3.12) | Density fields, persistence, distances, simplification, tests, figures |
| `ada` (IIIT-H SLURM cluster) | RTX 2080 Ti 11 GB nodes | FCGS encode/decode, rendering and image metrics, HAC / Scaffold-GS training and export, a few large CPU jobs |

## Analysis environment (CPU)

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests -q
```

Runs at 512³ need 11–21 GB of RAM. On the laptop each run was put in its own
memory-capped process, for example
`systemd-run --user --scope -p MemoryMax=14G -p MemorySwapMax=4G .venv/bin/python ...`.
One 512³ run (opacity pruning 70%) needed 21 GB and ran on an `ada` CPU node; it gave bit-identical diagrams.

## FCGS on `ada`

Full log, job scripts and driver diffs: [docs/fcgs/](fcgs/).

- Conda env: Python 3.7.12, torch 1.12.1+cu116, cudatoolkit 11.6.2, mkl pinned to 2024.0 (2025.x breaks `import torch`),
  nvcc 11.7, gcc 11, `TORCH_CUDA_ARCH_LIST="7.5;8.6"`.
- FCGS commit 31e59a46; rasterizer submodule 59f5f77 (stock graphdeco-inria); tmc3 = mpeg-pcc-tmc13 a3d15c5e.
- **Non-stock setting:** the stock `per_step_size=1,000,000` runs out of memory on an 11 GB card, so we used
  **400,000** with arithmetic-coder chunk sizes 500k/250k/250k. Decoding these bitstreams needs
  `decode_fcgs.py --chunk_sizes 500000,250000,250000`. Re-run one rate:
  `CHUNKS=500000,250000,250000 sbatch --export=ALL -C 2080ti docs/fcgs/run_lambda.sbatch <lambda> 400000`.
- Sizes are `du -sb` of the bitstream directory (116.7 / 90.0 / 67.3 MB at λ = 1e-4 / 4e-4 / 16e-4). FCGS's printed
  "Compressed size" leaves out the G-PCC xyz file.
- The decoded .ply is not in input order (Morton sort and shuffle per step). The density field does not depend on order.

## HAC and Scaffold-GS on `ada`

Full log and job scripts: [docs/hac/](hac/). Both models were trained from the official repositories with their
Mip-NeRF 360 settings (HAC λ = 0.004). They store anchors plus MLPs, not explicit Gaussians, so
`scripts/export_neural_gaussians.py` decodes them per camera and `scripts/hac_validate_export.py` checks the export
against the stock INRIA rasterizer before any field is computed.

## Data

`scripts/fetch_data.py` downloads only the bicycle files:

- Pretrained 3DGS scene (`bicycle/point_cloud/iteration_30000/point_cloud.ply`, `cameras.json`, `cfg_args`). INRIA's
  server is throttled, so the files come from a Hugging Face mirror of the same archive, and each file's CRC32 is
  checked against the CRC in INRIA's own zip directory (read with HTTP range requests).
- Mip-NeRF 360 bicycle images (`images_4`, `sparse/0`) by range requests into `360_v2.zip`. Test views are every 8th
  image by name; `scripts/fetch_test_views.py` fetches only those (checked against `configs/test_views_sha256.txt`).

Checksums are written to `data/SHA256SUMS`. Self-made variants are rebuilt on `ada` from the original plus the
saved keep mask (pruning) or by re-quantising, and checked against the `raw_sha256` in
`results/variants/<name>/<name>.json`.

The `configs/render_list_*.txt` files and some JSON records under `results/` contain absolute paths on `ada`.
They record where each run read its inputs; to re-run, edit the paths to your own copies.
