#!/bin/bash -l
#SBATCH -J fcgs-env
#SBATCH -p u22
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o /home2/bala_skv/topogs-fcgs/logs/build_env_%j.log
set -x
R=/home2/bala_skv/topogs-fcgs
export CONDA_PKGS_DIRS=$R/conda_pkgs PIP_CACHE_DIR=$R/pip_cache CONDA_ALWAYS_YES=1
export TMPDIR=$R/tmp; mkdir -p $TMPDIR
df -h /tmp | tail -1
hostname; date; df -h ~ | tail -1
# --- tmc3 ---
module load u22/cmake/3.31.7
cd $R/third_party/mpeg-pcc-tmc13
SECONDS=0
[ -x build/tmc3/tmc3 ] || { rm -rf build; cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 8; }
echo "tmc3 build seconds: $SECONDS"
ls -la build/tmc3/tmc3 && build/tmc3/tmc3 --help 2>&1 | head -3
# --- conda env ---
source ~/miniconda3/etc/profile.d/conda.sh
SECONDS=0
conda env create -p $R/envs/FCGS_env -f $R/scripts/fcgs_env_base.yml
echo "conda env create exit $? seconds: $SECONDS"
df -h ~ | tail -1
conda activate $R/envs/FCGS_env
python -c "import torch, torchvision, plyfile, lpips; print(torch.__version__, torch.version.cuda, torchvision.__version__)"
conda list | grep -Ei "pytorch|cudatoolkit|python |plyfile|torchvision"
du -sh $R/envs/FCGS_env $R/conda_pkgs
date
