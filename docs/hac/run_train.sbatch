#!/bin/bash -l
#SBATCH -p u22
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -C 2080ti
#SBATCH --time=08:00:00
#SBATCH -o /home2/bala_skv/topogs-fcgs/hac/logs/train_%x_%j.log
# usage: sbatch -J <name> run_train.sbatch hac|scaffold ITERATIONS OUTNAME [extra train.py args...]
# HAC:      python train.py -s DATA --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations N -m OUT --lmbda 0.004   (run_shell_mip360.py)
# Scaffold: python train.py --eval -s DATA --lod 0 --gpu -1 --voxel_size 0.001 --update_init_factor 16 --appearance_dim 0 --ratio 1 --iterations N --port P -m OUT   (train_mip360.sh + train.sh)
# Both + --data_device cpu (training images kept in host RAM; output-neutral). Resolution flag left at default -1 (-> 1600 px width).
set -u
WHICH=${1:?hac|scaffold}; ITERS=${2:?iterations}; NAME=${3:?outname}; shift 3
R=/home2/bala_skv/topogs-fcgs
H=$R/hac
DATA=$H/data/bicycle
OUT=$H/outputs/$NAME
export TMPDIR=$R/tmp TORCH_HOME=$R/torch_cache WANDB_MODE=disabled PYTHONUNBUFFERED=1
mkdir -p $TMPDIR $H/outputs
ulimit -n 4096
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $R/envs/FCGS_env
source $R/envs/HAC_env/bin/activate
echo "host $(hostname) job $SLURM_JOB_ID which=$WHICH iters=$ITERS out=$OUT extra=[$*] $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
df -h ~ | tail -1
[ -e $OUT ] && { echo "output $OUT exists, refusing to overwrite"; exit 1; }
if [ $WHICH = hac ]; then
  cd $H/HAC
  CMD="train.py -s $DATA --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 --iterations $ITERS -m $OUT --lmbda 0.004 --data_device cpu"
else
  cd $H/Scaffold-GS
  CMD="train.py --eval -s $DATA --lod 0 --gpu -1 --voxel_size 0.001 --update_init_factor 16 --appearance_dim 0 --ratio 1 --iterations $ITERS --port $((10000 + SLURM_JOB_ID % 20000)) -m $OUT --data_device cpu"
fi
echo "CMD: python $CMD $*"
GM=$H/logs/gpumem_${NAME}_${SLURM_JOB_ID}.csv
( while true; do nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader >> $GM; sleep 1; done ) & SMI=$!   # appended per sample (nvidia-smi -l lost its buffered tail on kill)
SECONDS=0
/usr/bin/time -v python $H/memwrap.py $CMD "$@" 2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -vE "^(Training|Rendering) progress:.*[0-9]+/[0-9]+ \[[0-9:]+<[0-9:?]+, *[0-9.?]+(it/s|s/it)(, Loss=[0-9.]+)?\]$"
EC=${PIPESTATUS[0]}
kill $SMI
echo "EXIT=$EC wall_s=$SECONDS"
echo "peak_gpu_MiB (nvidia-smi, 1 s sampling): $(awk -F', ' '{gsub(/ MiB/,"",$2); if($2>m)m=$2} END{print m}' $GM)"
du -sh $OUT; du -sh $OUT/* 2>/dev/null
# disk (home ~1 GB free): after a successful run, drop the end-of-run gt/ and errors/ PNGs (gt = the 1600-px test images,
# errors = |render-gt|; both regenerable, metrics are already in results.json). renders/ is kept. CLEAN_PNG=0 keeps everything.
if [ $EC -eq 0 ] && [ "${CLEAN_PNG:-1}" = 1 ]; then rm -rf $OUT/test/ours_*/gt $OUT/test/ours_*/errors; echo "removed gt/ errors/ PNGs"; du -sh $OUT; fi
df -h ~ | tail -1; date
