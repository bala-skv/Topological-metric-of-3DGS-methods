#!/bin/bash -l
#SBATCH -J hac-env
#SBATCH -p u22
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -C 2080ti
#SBATCH --time=01:30:00
#SBATCH -o /home2/bala_skv/topogs-fcgs/hac/logs/build_hac_env_%j.log
# Builds HAC_env = python venv layered on FCGS_env (--system-site-packages; FCGS_env itself is never modified)
# + HAC's extra deps + HAC's CUDA extensions. (conda --clone was tried first, job 88356: it needs the package
# tarballs, which were cleaned from conda_pkgs; re-downloading them does not fit the disk.)
# HAC's extensions are installed into the venv with --ignore-installed and shadow FCGS_env's stock builds
# (venv site-packages come first on sys.path); the import test asserts every module resolves inside HAC_env.
# Scaffold-GS uses the same env: its diff-gaussian-rasterization / simple-knn sources are identical to HAC's zips.
set -u
R=/home2/bala_skv/topogs-fcgs
H=$R/hac
E=$R/envs/HAC_env
export TMPDIR=$R/tmp; mkdir -p $TMPDIR
export PIP_CACHE_DIR=$R/pip_cache_hac CONDA_PKGS_DIRS=$R/conda_pkgs CONDA_ALWAYS_YES=1
hostname; date; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
df -h ~ | tail -1
source ~/miniconda3/etc/profile.d/conda.sh
[ -x $E/bin/python ] || $R/envs/FCGS_env/bin/python -m venv --system-site-packages $E || { echo "VENV FAILED"; exit 1; }
df -h ~ | tail -1
conda activate $R/envs/FCGS_env   # conda env vars / libs of the base
source $E/bin/activate
command -v python pip; python -c "import sys; print(sys.prefix, sys.base_prefix)"
pip install -q --upgrade "pip<24.1"   # into the venv only (FCGS_env pip untouched)
# venv ships setuptools 47.1, whose Version objects lack .major -> torch 1.12 cpp_extension raises on the
# nvcc 11.7 vs torch cu116 minor mismatch instead of warning (job 88357). Use FCGS_env's version (69.0.3, from
# conda-forge; PyPI has no py3.7 build of it, job 88374) by removing the venv-local copy.
pip uninstall -y setuptools
python -c "import setuptools; print('setuptools', setuptools.__version__, setuptools.__file__)"
pip freeze > $H/logs/pip_freeze_before.txt
# constraints: keep the torch stack exactly as in FCGS_env
printf "numpy==1.21.6\ntorch==1.12.1\ntorchvision==0.13.1\nPillow==9.2.0\nscipy==1.7.3\n" > $H/logs/constraints.txt
# prebuilt wheel (pyg publishes 1.12.x wheels under torch-1.12.0; the 1.12.1 page made pip fall back to the sdist, job 88357)
pip install -c $H/logs/constraints.txt --only-binary torch-scatter "torch-scatter==2.1.0+pt112cu116" -f https://data.pyg.org/whl/torch-1.12.0+cu116.html || echo "PIP torch-scatter FAILED"
pip install -c $H/logs/constraints.txt einops wandb colorama opencv-python-headless jaxtyping || echo "PIP extras FAILED"
pip freeze > $H/logs/pip_freeze_after.txt
diff $H/logs/pip_freeze_before.txt $H/logs/pip_freeze_after.txt
df -h ~ | tail -1
module load u22/cuda/11.7
export CUDA_HOME=$(dirname $(dirname $(command -v nvcc)))
export TORCH_CUDA_ARCH_LIST="7.5" MAX_JOBS=8
echo "CUDA_HOME=$CUDA_HOME"; nvcc --version | tail -2; gcc --version | head -1; python --version
mkdir -p $H/ext_src && cd $H/ext_src
for z in diff-gaussian-rasterization simple-knn gridencoder arithmetic; do
  [ -d $z ] || unzip -q $H/HAC/submodules/$z.zip -x "__MACOSX/*"
done
ls
for ext in diff-gaussian-rasterization simple-knn gridencoder arithmetic; do
  SECONDS=0
  pip install -v --no-build-isolation --ignore-installed --no-deps ./$ext > $H/logs/ext_${ext}.log 2>&1
  echo "EXT $ext exit=$? seconds=$SECONDS"
  grep -iE "^error|error:" $H/logs/ext_${ext}.log | head -5
done
rm -rf $H/ext_src/*/build $PIP_CACHE_DIR
pip list 2>/dev/null | grep -iE "diff|knn|grid|arith|scatter|einops|wandb|colorama|opencv|jaxtyping|numpy|^torch|lpips"
df -h ~ | tail -1
export TORCH_HOME=$R/torch_cache
cd $H/HAC
python - <<'PY'
import torch, importlib, math
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
for m in ["diff_gaussian_rasterization", "diff_gaussian_rasterization._C", "simple_knn._C", "_gridencoder", "arithmetic", "torch_scatter", "einops", "wandb", "colorama", "cv2", "jaxtyping", "lpips"]:
    try:
        mod = importlib.import_module(m); f = getattr(mod, "__file__", ""); print("IMPORT OK", m, f, "" if "HAC_env" in (f or "") else "<-- NOT IN HAC_env")
    except Exception as e:
        print("IMPORT FAIL", m, repr(e))
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
print("has visible_filter:", hasattr(GaussianRasterizer, "visible_filter"))
import scene, gaussian_renderer, utils.encodings
print("HAC modules import ok (scene, gaussian_renderer, utils.encodings)")
dev = "cuda"; N = 100
s = GaussianRasterizationSettings(image_height=64, image_width=64, tanfovx=math.tan(0.5), tanfovy=math.tan(0.5),
    bg=torch.zeros(3, device=dev), scale_modifier=1.0, viewmatrix=torch.eye(4, device=dev), projmatrix=torch.eye(4, device=dev),
    sh_degree=1, campos=torch.zeros(3, device=dev), prefiltered=False, debug=False)
r = GaussianRasterizer(s)
means = torch.rand(N, 3, device=dev) * 0.5; means[:, 2] += 2.0
rot = torch.tensor([[1., 0, 0, 0]], device=dev).repeat(N, 1)
out = r(means3D=means, means2D=torch.zeros_like(means), shs=None, colors_precomp=torch.rand(N, 3, device=dev),
        opacities=torch.ones(N, 1, device=dev) * 0.5, scales=torch.ones(N, 3, device=dev) * 0.05, rotations=rot, cov3D_precomp=None)
print("rasterizer forward returns", len(out), "values; shapes", [tuple(o.shape) for o in out])
vis = r.visible_filter(means3D=means, scales=torch.ones(N, 3, device=dev) * 0.05, rotations=rot, cov3D_precomp=None)
print("visible_filter radii>0:", int((vis > 0).sum()), "/", N)
PY
cd $H/Scaffold-GS
python -c "import scene, gaussian_renderer; print('Scaffold-GS modules import ok')"
# FCGS_env must be unchanged: still the stock rasterizer (no visible_filter)
deactivate; conda deactivate
$R/envs/FCGS_env/bin/python -c "import diff_gaussian_rasterization as d; from diff_gaussian_rasterization import GaussianRasterizer as G; print('FCGS_env rasterizer', d.__file__, 'has visible_filter:', hasattr(G,'visible_filter'))"
du -sh $E; df -h ~ | tail -1; date
