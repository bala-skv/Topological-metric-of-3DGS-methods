#!/bin/bash -l
#SBATCH -J hac-export
#SBATCH -p u22
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH -C 2080ti
#SBATCH --time=01:30:00
#SBATCH -o /home2/bala_skv/topogs-fcgs/hac/logs/export_%j.log
# HAC steps 2-3 on ada: export neural Gaussians (HAC_env, one process per model, cwd = model repo),
# identity re-export of camera 0 (separate processes), stock-rasterizer validation + crop subsets (FCGS_env).
# Big intermediates on node-local /ssd_scratch (deleted at the end); only crop .ply, JSON, PNG go to home.
#   sbatch scripts/hac_export.sbatch          (from ~/topogs-fcgs/topogs_repo; job 1 = 95699)
#   sbatch --export=ALL,OUTH=/home2/bala_skv/topogs-fcgs/hac/export2,EXPFLAGS=--save-cull-set,VALFLAGS='--keep-zero --diag-png' scripts/hac_export.sbatch   (job 2)
set -u
R=/home2/bala_skv/topogs-fcgs; H=$R/hac; CODE=$R/topogs_repo
CAMS=_DSC8679,_DSC8703,_DSC8735,_DSC8760,_DSC8792,_DSC8816,_DSC8848,_DSC8872   # configs/hac_export.yaml
SCR=/ssd_scratch/bala_skv/hac_export_$SLURM_JOB_ID
OUTH=${OUTH:-$H/export}            # job 2: OUTH=$H/export2
EXPFLAGS=${EXPFLAGS:-}             # job 2: --save-cull-set
VALFLAGS=${VALFLAGS:-}             # job 2: --keep-zero --diag-png
mkdir -p $SCR $OUTH
export TMPDIR=$R/tmp TORCH_HOME=$R/torch_cache PYTHONUNBUFFERED=1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $R/envs/FCGS_env
HACPY=$R/envs/HAC_env/bin/python
FCGSPY=$R/envs/FCGS_env/bin/python
echo "host $(hostname) job $SLURM_JOB_ID $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h ~ | tail -1; df -h /ssd_scratch | tail -1
fail=0
run() { echo "+ $*"; "$@" || { echo "FAILED: $*"; fail=1; }; }
cd $H/Scaffold-GS
run $HACPY $CODE/scripts/export_neural_gaussians.py --which scaffold --model $H/outputs/scaffold_bicycle --cams $CAMS --out $SCR/scaffold $EXPFLAGS
run $HACPY $CODE/scripts/export_neural_gaussians.py --which scaffold --model $H/outputs/scaffold_bicycle --cams ${CAMS%%,*} --out $SCR/scaffold --tag _rep $EXPFLAGS
cd $H/HAC
run $HACPY $CODE/scripts/export_neural_gaussians.py --which hac --model $H/outputs/hac_bicycle_l0.004 --cams $CAMS --out $SCR/hac $EXPFLAGS
run $HACPY $CODE/scripts/export_neural_gaussians.py --which hac --model $H/outputs/hac_bicycle_l0.004 --cams ${CAMS%%,*} --out $SCR/hac --tag _rep $EXPFLAGS
cd $CODE
COMMON="--mask-ply $R/results/fcgs_crop/original_crop.ply --box $CODE/configs/crop_box.json --min-psnr 45"
for M in scaffold hac; do
  for T in "" _rep; do
    run $FCGSPY scripts/hac_validate_export.py --exp $SCR/$M --tag "$T" --ply-out $OUTH/$M $COMMON $VALFLAGS
  done
  cp $SCR/$M/export*.json $SCR/$M/cameras*.npz $SCR/$M/*_own_cull.png $OUTH/$M/ 2>/dev/null
  # one stock-vs-own visual pair per model (first camera), as 8-bit PNG
  $FCGSPY -c "
import numpy as np; from PIL import Image
for k in ['stock_exp','own_cull']:
    a=np.load('$SCR/$M/${CAMS%%,*}_'+k+'.npy'); Image.fromarray((np.clip(a,0,1)*255+0.5).astype(np.uint8).transpose(1,2,0)).save('$OUTH/$M/${CAMS%%,*}_'+k+'_8bit.png')"
done
du -sh $SCR/* $OUTH/*
rm -rf $SCR
df -h ~ | tail -1
echo "EXIT fail=$fail $(date)"
