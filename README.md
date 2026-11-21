# Topological fidelity audit of compressed 3D Gaussian Splatting

Course project, Scientific Visualisation. Balasubramanian.

Compressed 3DGS scenes are judged by PSNR, SSIM and LPIPS, which average over pixels and barely notice damage to
thin structures. This project samples the 3D density field of a 3DGS scene around the rear wheel of the
Mip-NeRF 360 *bicycle* scene, computes cubical persistent homology (H₀, H₁, H₂) of that field, and compares
compressed variants with the original through bottleneck and 2-Wasserstein distances. Each distance is measured
against a noise floor (the distance caused by shifting the grid by half a voxel), and a change counts only if it is
above the floor at both 256³ and 384³. The floor, the cutoff and the rule were fixed on the original model before any
compressed model was analysed.

The full report is in [docs/report.md](docs/report.md) (Word version: [docs/report.docx](docs/report.docx)).

## Main results

- **Image metrics and topology disagree.** Opacity pruning 30% and FCGS λ = 16e-4 lose the same PSNR on the wheel
  (0.50 vs 0.49 dB), but opacity pruning moves the H₁ diagram 2.26 / 3.95 times the noise floor at 256³ / 384³, while
  FCGS stays at 0.93 / 1.99. Only opacity pruning passes the decision rule.
- **The audit explains the difference.** The original field has 37 air-enclosing loops at 384³, 34 of them gaps
  between spokes. Opacity pruning keeps the loops but thins the weakest spoke segment of each one to a median 0.61 of
  its original density; FCGS leaves it at 0.99.
- **The result does not depend on the grid.** The ranking of all variants barely changes from 128³ to 512³
  (Spearman ρ 0.96–1.00 between every pair of resolutions).
- **HAC vs Scaffold-GS.** HAC (λ = 0.004) keeps every spoke loop and stays below the camera-spread floor
  (0.73–0.78×). The audit only works with a validated export: dropping HAC's zero-scale Gaussians would have faked a
  spoke break.
- **Persistence-based simplification** (h-maxima / h-minima) removes most peaks and all cavities while changing
  ≤ 3.1% of voxels at 256³, and leaves all 35 air-enclosing loops and their weakest links unchanged.

## Repository layout

| Path | Contents |
|---|---|
| `topogs/` | Library: `.ply` loader (`io_ply`), density field (`density`), analysis grids (`grids`), persistence and distances (`topology`), self-made compression baselines (`compress`), plotting helpers (`viz`) |
| `scripts/` | Entry points, run from the repository root (`python scripts/<name>.py ...`); usage is in each docstring |
| `configs/` | Frozen analysis settings (`analysis.yaml`, `analysis_hac.yaml`), crop box, variant lists, render lists |
| `results/` | Small outputs: JSON records, diagrams of the HAC fields, tables (`.md`, `.csv`) and figures |
| `docs/` | Report (`report.md`, `report.docx`), proposal, environment notes (`setup.md`), FCGS and HAC install logs and cluster job scripts |
| `tests/` | Synthetic sanity tests (shapes with known homology) and tests of the compression baselines |

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests -q          # 15 passed, 1 skipped
```

The analysis runs on a CPU. FCGS, HAC / Scaffold-GS and rendering need an NVIDIA GPU and their own conda
environments; see [docs/setup.md](docs/setup.md), [docs/fcgs/](docs/fcgs/) and [docs/hac/](docs/hac/).

## Reproducing the pipeline

Run from the repository root, in this order. Steps marked *(GPU)* ran on the cluster.

| # | Step | Command |
|---|---|---|
| 1 | Fetch the pretrained scene and images | `python scripts/fetch_data.py --out data` |
| 2 | Choose and check the crop box | `python scripts/inspect_scene.py configs/crop.yaml`, `python scripts/antialias_study.py configs/antialias.yaml` |
| 3 | Density fields of the original (and the half-voxel shifted grid) | `python scripts/compute_field.py configs/field_original.yaml` (also `field_original_shift.yaml`, `field_original_512.yaml`) |
| 4 | Persistence of the original | `python scripts/run_persistence.py configs/persistence_original.yaml` (also the `_shift` and `_512` configs) |
| 5 | Noise floor and cutoff | `python scripts/noise_floor.py configs/analysis.yaml results/persistence/original results/persistence/original_shift` |
| 6 | FCGS encode/decode *(GPU)* | `docs/fcgs/run_lambda.sbatch` (see `docs/setup.md` for the non-stock chunk sizes) |
| 7 | Build variants, fields, diagrams, distances | `python scripts/run_variants.py configs/variants.yaml` (also `variants_seeds.yaml`, `variants_512.yaml`) |
| 8 | Render test views and image metrics *(GPU)* | `scripts/render_eval.sbatch` → `scripts/render_eval.py` |
| 9 | Combined table and Figure 2 | `python scripts/combined_table.py` |
| 10 | Where the change is | `python scripts/explain_change.py opac30 256`, `python scripts/structure_levels.py 256 384`, `python scripts/weak_link_figure.py 384` |
| 11 | Image error on spoke pixels, render sheets | `python scripts/spoke_mask_check.py`, `python scripts/render_sheet.py` |
| 12 | Resolution stability | `python scripts/resolution_stability.py` |
| 13 | HAC audit *(training and export on GPU)* | `docs/hac/run_train.sbatch` → `scripts/hac_frame_check.py` → `hac_export.sbatch` → `hac_fields.py` → `hac_persistence.py` → `hac_floors.py` → `hac_audit.py` → `hac_image_side.py` (steps 1–7 in each docstring) |
| 14 | Persistence-based simplification | `python scripts/simplify_field.py` (then `--plot` to redraw) |

Runs at 512³ need 11–21 GB of RAM; run each in its own memory-capped process (see `docs/setup.md`).

## Report figures and tables

| In the report | Made by | Output |
|---|---|---|
| Table 1 (noise floor) | `noise_floor.py` | `results/persistence/original_shift/noise_floor*.json`, `configs/analysis.yaml` |
| Table 4 (cost of persistence) | `run_persistence.py` | `results/persistence/original/original_stats.json` |
| Figure 1 (diagram and barcode) | `run_persistence.py`; separate panels: `plot_diagram_panels.py` | `results/persistence/original/`, `.../panels/` |
| Table 5, Figure 2 (image vs topology) | `combined_table.py`; separate panels: `plot_scatter_panels.py` | `results/combined/combined_table.md`, `disagreement_scatter.png`, `panels/` |
| Figure 3 (weak link before/after) | `weak_link_figure.py` | `results/explain/weak_link_r384.png` |
| Table 6, Figure 4 (spoke pixels, weak link in a render) | `spoke_mask_check.py`, `render_sheet.py` | `results/explain/spoke_mask_all_views.json`, `results/render_wheel_png/weak_link_render_DSC8679.png` |
| Table 7, Figure 5 (resolution stability) | `resolution_stability.py` | `results/combined/resolution_stability.{md,png}` |
| Table 8 (HAC vs Scaffold-GS) | `hac_audit.py`, `hac_image_side.py` | `results/hac/audit_table.md`, `audit.json`, `image_side.json` |
| Table 9, Figure 6 (simplification) | `simplify_field.py`; separate panels: `plot_simplify_panels.py` | `results/simplify/simplify.{md,png}`, `panels/` |

## Not included in the repository

Large files are excluded by `.gitignore` and are regenerated by the scripts above:

- the pretrained scene, images and every compressed `.ply` (`data/`; the original alone is 1.5 GB);
- density fields and persistence diagrams (`*.npy`, `*.npz`; one 512³ field is about 0.5 GB, the HAC fields 5.7 GB);
- per-view test renders (`results/render_wheel_png/*__DSC*.png`, `results/render_all25/*.png`).

Each field has a JSON record next to where the `.npy` would be (grid, spacing, filter, timing, git commit), and each
variant has one with the SHA-256 of its `.ply`, so a re-run can be checked against them. Some records and the `configs/render_list_*.txt`
files contain absolute cluster paths; they show where a run read its inputs and must be edited to re-run.

## External code

- 3D Gaussian Splatting, pretrained bicycle model: Kerbl et al., 2023 (graphdeco-inria).
- FCGS: Chen et al., 2025 (`YihangChen-ee/FCGS`), run unmodified apart from the chunk size.
- HAC: Chen et al., 2024; Scaffold-GS: Lu et al., 2024; trained from the official repositories.
- GUDHI 3.13 (cubical complexes, bottleneck and Hera Wasserstein distances), scikit-image (morphological reconstruction).
