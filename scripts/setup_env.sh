#!/usr/bin/env bash
# Phase 0: Python env for d3-flash on A100 (driver 570 -> CUDA <= 12.8, so cu126 wheels; upstream validated torch 2.13).
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh   # EOJEV_ENVS, EOJEV_CONDA and optional PIP_CACHE_DIR come from env.local.sh
ENV=$EOJEV_ENVS/eojev
source "$EOJEV_CONDA/etc/profile.d/conda.sh"
[ -d "$ENV" ] || conda create -y -p "$ENV" -c conda-forge --override-channels python=3.12
conda activate "$ENV"
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install "transformers==5.17.0" "accelerate>=1.10" "safetensors>=0.6" "huggingface_hub>=1.0" \
    "flash-linear-attention[cuda]==0.5.2" einops "pillow>=10" opencv-python-headless
# harness / analysis
python -m pip install numpy pandas pyarrow scipy scikit-learn pyyaml tqdm matplotlib datasets pycountry
python - <<'PY'
import torch, transformers, fla, torchvision
print("torch", torch.__version__, "cuda", torch.version.cuda, "| torchvision", torchvision.__version__,
      "| transformers", transformers.__version__, "| fla", getattr(fla, "__version__", "?"))
PY
echo "ENV READY $(date)"
