#!/bin/bash -l
#SBATCH -J fcgs-run
#SBATCH -p u22
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -o /home2/bala_skv/topogs-fcgs/logs/run_%x_%j.log
# usage: sbatch -C 2080ti scripts/run_lambda.sbatch LAMBDA [per_step_size ...]   (sizes tried in order until encode+decode both pass)
LMD=${1:?lambda}; shift
SIZES="${@:-1000000}"
R=/home2/bala_skv/topogs-fcgs
IN=$R/data/bicycle/point_cloud.ply
OUT=$R/results/fcgs/lmd_$LMD
mkdir -p $OUT/bits
# optional: PLYDIR=/ssd_scratch/... writes the decoded ply to node-local disk (home quota), deleted at the end after checks
PLY=${PLYDIR:-$OUT}/point_cloud.ply; mkdir -p $(dirname $PLY)
echo "decoded ply -> $PLY"
export TMPDIR=$R/tmp; mkdir -p $TMPDIR
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $R/envs/FCGS_env
export PATH=$R/third_party/mpeg-pcc-tmc13/build/tmc3:$PATH
export PYTHONPATH=$R/third_party/FCGS
CHUNKS=${CHUNKS:-2000000,1000000,1000000}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}
echo "CHUNKS=$CHUNKS PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "host $(hostname) job $SLURM_JOB_ID lambda $LMD sizes $SIZES $(date)"; command -v tmc3; df -h ~ | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $R/third_party/FCGS   # checkpoints are loaded relative to cwd
peak() { awk -F', ' '{gsub(/ MiB/,"",$2); if($2>m)m=$2} END{print m}' $1; }
OK=0
for SZ in $SIZES; do
  rm -rf $OUT/bits/$LMD $PLY
  echo "=== per_step_size $SZ"
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -lms 200 > $OUT/gpumem_encode.csv & SMI=$!
  SECONDS=0
  /usr/bin/time -v python $R/scripts/encode_fcgs.py --lmd $LMD --per_step_size $SZ --chunk_sizes $CHUNKS --ply_path_from $IN --bit_path_to $OUT/bits --determ 1 > $OUT/encode.log 2>&1
  EC=$?; echo "ENCODE exit=$EC wall_s=$SECONDS"; kill $SMI; echo "peak_gpu_MiB_encode (nvidia-smi): $(peak $OUT/gpumem_encode.csv)"
  grep -E "per_step_size|chunk_size_list|time:|max_memory|Compressed size|out of memory|Error" $OUT/encode.log | head
  [ $EC -ne 0 ] && { cp $OUT/encode.log $OUT/encode_failed_step$SZ.log; continue; }
  echo "bits_bytes: $(du -sb $OUT/bits | cut -f1)  files: $(find $OUT/bits -type f | wc -l)  steps: $(ls $OUT/bits/$LMD | wc -l)"
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -lms 200 > $OUT/gpumem_decode.csv & SMI=$!
  SECONDS=0
  /usr/bin/time -v python $R/scripts/decode_fcgs.py --lmd $LMD --chunk_sizes $CHUNKS --bit_path_from $OUT/bits --ply_path_to $PLY > $OUT/decode.log 2>&1
  DC=$?; echo "DECODE exit=$DC wall_s=$SECONDS"; kill $SMI; echo "peak_gpu_MiB_decode (nvidia-smi): $(peak $OUT/gpumem_decode.csv)"
  grep -E "max_memory|Decompressed|out of memory|Error" $OUT/decode.log | head
  [ $DC -ne 0 ] && { cp $OUT/decode.log $OUT/decode_failed_step$SZ.log; continue; }
  OK=1; echo "USED per_step_size $SZ"; printf "per_step_size=%s\nchunk_sizes=%s\nPYTORCH_CUDA_ALLOC_CONF=%s\njob=%s host=%s\n" $SZ $CHUNKS $PYTORCH_CUDA_ALLOC_CONF $SLURM_JOB_ID $(hostname) > $OUT/driver_params.txt; break
done
grep -E "Maximum resident|Elapsed" $OUT/encode.log $OUT/decode.log
grep -niE "nan|warn" $OUT/encode.log $OUT/decode.log | head
ls -la $OUT
[ $OK -eq 1 ] && { python $R/scripts/compare_ply.py $IN $PLY; ls -l $PLY; sha256sum $PLY; }
[ -n "$PLYDIR" ] && { rm -rf $PLYDIR; echo "removed node-local $PLYDIR"; }
df -h ~ | tail -1; date
