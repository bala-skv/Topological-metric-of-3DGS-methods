#!/bin/bash -l
#SBATCH -J fcgs-ext
#SBATCH -p u22
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -C 2080ti
#SBATCH --time=01:00:00
#SBATCH -o /home2/bala_skv/topogs-fcgs/logs/build_ext_%j.log
R=/home2/bala_skv/topogs-fcgs
export PIP_CACHE_DIR=$R/pip_cache
export TMPDIR=$R/tmp; mkdir -p $TMPDIR
hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source ~/miniconda3/etc/profile.d/conda.sh
export CONDA_PKGS_DIRS=$R/conda_pkgs CONDA_ALWAYS_YES=1
# fix: torch 1.12.1 + mkl>=2024.1 -> "undefined symbol: iJIT_NotifyEvent"
if ! $R/envs/FCGS_env/bin/python -c "import torch" 2>/dev/null; then
  conda install -p $R/envs/FCGS_env -c conda-forge --override-channels "mkl=2024.0"
fi
# free disk: remove downloaded tarballs + unused extracted pkgs, only in my own pkgs dir
conda clean -p -t -y
du -sh $R/conda_pkgs; df -h ~ | tail -1
conda activate $R/envs/FCGS_env
python -c "import torch; print('torch import ok', torch.__version__)" || exit 1
conda list | grep -E "^(mkl|pytorch|cudatoolkit|python) "
module load u22/cuda/11.7
export CUDA_HOME=$(dirname $(dirname $(command -v nvcc)))
export TORCH_CUDA_ARCH_LIST="7.5;8.6"
export MAX_JOBS=8
echo "CUDA_HOME=$CUDA_HOME"; nvcc --version | tail -2; gcc --version | head -1; python --version
cd $R/third_party/FCGS
for ext in diff-gaussian-rasterization simple-knn freqencoder gridencoder gridcreater arithmetic; do
  SECONDS=0
  pip install -v --no-build-isolation submodules/$ext > $R/logs/ext_${ext}.log 2>&1
  echo "EXT $ext exit=$? seconds=$SECONDS"
  grep -iE "^error|error:|warning: .*mismatch|CUDA version" $R/logs/ext_${ext}.log | head -5
done
pip list 2>/dev/null | grep -iE "diff|knn|freq|grid|arith|torch|lpips|plyfile"
export PATH=$R/third_party/mpeg-pcc-tmc13/build/tmc3:$PATH
command -v tmc3
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
import importlib
for m in ["diff_gaussian_rasterization", "diff_gaussian_rasterization._C", "simple_knn._C", "_freqencoder", "_gridencoder", "_gridcreater", "arithmetic"]:
    try:
        mod = importlib.import_module(m); print("IMPORT OK", m, getattr(mod, "__file__", ""))
    except Exception as e:
        print("IMPORT FAIL", m, repr(e))
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
print("Settings fields:", GaussianRasterizationSettings._fields)
import model.FCGS_model
print("model.FCGS_model import ok")
# tiny rasterizer forward test
import math
dev = "cuda"
N = 100
s = GaussianRasterizationSettings(image_height=64, image_width=64, tanfovx=math.tan(0.5), tanfovy=math.tan(0.5),
    bg=torch.zeros(3, device=dev), scale_modifier=1.0, viewmatrix=torch.eye(4, device=dev), projmatrix=torch.eye(4, device=dev),
    sh_degree=0, campos=torch.zeros(3, device=dev), prefiltered=False, debug=False)
r = GaussianRasterizer(s)
means = torch.rand(N, 3, device=dev) * 0.5; means[:, 2] += 2.0
out = r(means3D=means, means2D=torch.zeros_like(means), opacities=torch.ones(N, 1, device=dev) * 0.5,
        colors_precomp=torch.rand(N, 3, device=dev), scales=torch.ones(N, 3, device=dev) * 0.05,
        rotations=torch.tensor([[1., 0, 0, 0]], device=dev).repeat(N, 1))
print("rasterizer forward returns", len(out), "values; shapes", [tuple(o.shape) for o in out])
PY
date
