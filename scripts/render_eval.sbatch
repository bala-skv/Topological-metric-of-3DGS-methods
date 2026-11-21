#!/bin/bash -l
#SBATCH -J topogs-render
#SBATCH -p u22
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o /home2/bala_skv/topogs-fcgs/logs/render_%j.log
# usage: sbatch -C 2080ti [--export=ALL,OUT=dir,SAVE=25] scripts/render_eval.sbatch LISTFILE
# OUT: output dir (default results/render_wheel); SAVE: number of test views saved as PNG (default 3)
# LISTFILE lines:  NAME ply PATH    |    NAME variant JSON KEEPDIR
R=/home2/bala_skv/topogs-fcgs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $R/envs/FCGS_env
export TMPDIR=$R/tmp TORCH_HOME=$R/torch_cache
cd $R/topogs_repo
echo "host $(hostname) job $SLURM_JOB_ID $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
COMMON="--mask-ply $R/results/fcgs_crop/original_crop.ply --box $R/topogs_repo/configs/crop_box.json --cameras $R/data/bicycle/cameras.json --images $R/data/bicycle/images_4 --out ${OUT:-$R/results/render_wheel} --save-views ${SAVE:-3}"
while read -r NAME KIND A B; do
  [ -z "$NAME" ] && continue
  echo "=== $NAME"
  if [ "$KIND" = ply ]; then
    python scripts/render_eval.py --name $NAME --ply $A $COMMON
  else
    python scripts/render_eval.py --name $NAME --variant $A --keep-dir $B --original $R/data/bicycle/point_cloud.ply $COMMON
  fi
done < ${1:?listfile}
echo "done $(date)"
