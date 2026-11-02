#!/usr/bin/env bash
# Python envs for specialists whose code needs older libraries (docs/05_specialist_models.md):
#   eo-sa2va     TerraScope, EarthMind (Sa2VA code: transformers 4.42.3)
#   eo-earthdial EarthDial (InternVL2 / Phi-3 code pinned to transformers 4.37.2; its generation breaks on 4.42)
#   eo-vhm       VHM-7B (LLaVA code: transformers 4.31.0, torch 2.0.1)
#   eo-geochrono GeoChrono (Qwen3-VL-4B + LoRA + TempEnc: transformers 4.57.1, torch 2.8)
#   eo-teochat   TEOChat (Video-LLaVA code: pins from its pyproject.toml, transformers 4.31.0, torch 2.2.1 cu121)
#   eo-sismotion SIS-Motion (Self-in-Space motion code: pins from environments/conda/motion.yml, transformers 4.57.1,
#                torch 2.6.0 cu124; its Qwen2.5-VL subclass breaks on transformers 5)
# Usage: scripts/setup_specialist_envs.sh sa2va|earthdial|vhm|geochrono|teochat|sismotion
set -euo pipefail
which=$1
cd "$(dirname "$0")/.."
# EOJEV_ENVS, EOJEV_CONDA and optional PIP_CACHE_DIR / CONDA_PKGS_DIRS come from env.local.sh; set CONDA_PKGS_DIRS
# when building several envs in parallel (they raced in a shared ~/miniconda3/pkgs)
source env.sh
ENVS=$EOJEV_ENVS
source "$EOJEV_CONDA/etc/profile.d/conda.sh"
common="pillow numpy<2 pyyaml tqdm sentencepiece protobuf einops huggingface_hub safetensors opencv-python-headless"
case $which in
  sa2va)
    E=$ENVS/eo-sa2va; [ -d $E ] || conda create -y -p $E -c conda-forge --override-channels python=3.10
    conda activate $E
    python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
    python -m pip install transformers==4.42.3 timm==1.0.9 peft==0.11.1 "accelerate<1.0" $common ;;
  earthdial)
    E=$ENVS/eo-earthdial; [ -d $E ] || conda create -y -p $E -c conda-forge --override-channels python=3.10
    conda activate $E
    python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
    python -m pip install transformers==4.37.2 timm==0.9.12 peft==0.11.1 "accelerate<1.0" $common ;;
  vhm)
    E=$ENVS/eo-vhm; [ -d $E ] || conda create -y -p $E -c conda-forge --override-channels python=3.10
    conda activate $E
    python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
    python -m pip install transformers==4.31.0 peft==0.4.0 timm==0.6.13 accelerate==0.21.0 $common ;;
  geochrono)
    E=$ENVS/eo-geochrono; [ -d $E ] || conda create -y -p $E -c conda-forge --override-channels python=3.12
    conda activate $E
    python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
    python -m pip install transformers==4.57.1 peft==0.15.2 accelerate==1.10.1 qwen-vl-utils $common ;;
  teochat)
    # inference subset of TEOChat's pyproject.toml (no training / demo deps); bitsandbytes for its 8-bit eval loading
    # (0.41.0 imports scipy without declaring it); decord + pytorchvideo only because the LanguageBind video processor
    # is imported unconditionally
    E=$ENVS/eo-teochat; [ -d $E ] || conda create -y -p $E -c conda-forge --override-channels python=3.10
    conda activate $E
    python -m pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
    python -m pip install transformers==4.31.0 "tokenizers>=0.12.1,<0.14" sentencepiece==0.1.99 accelerate==0.26.1 \
      peft==0.7.1 bitsandbytes==0.41.0 "scipy<1.14" einops==0.6.1 einops-exts==0.0.4 timm==0.6.12 decord==0.6.0 \
      opencv-python-headless==4.7.0.72 $common
    pv=2ac6ed361309cb25144636ad82dd8bd6a4970e0c  # pytorchvideo main, which TEOChat's README installs from git
    python -m pip install "git+https://github.com/facebookresearch/pytorchvideo.git@$pv" ;;
  sismotion)
    # inference subset of Self-in-Space's environments/conda/motion.yml (no datasets / swanlab / qwen-vl-utils / av,
    # which only its training and video-decoding code use); deepspeed only because motion/src/uav/model/
    # motion_encoder.py imports it unconditionally (DS_BUILD_OPS=0: no CUDA ops are compiled); timm, einops and scipy
    # for the vendored VideoFlow MOFNet. No flash-attn: the adapter runs sdpa like the other adapters. Not $common:
    # it pins numpy<2, the authors pin numpy 2.2.6.
    E=$ENVS/eo-sismotion; [ -d $E ] || conda create -y -p $E -c conda-forge --override-channels python=3.10
    conda activate $E
    python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    python -m pip install transformers==4.57.1 accelerate==1.12.0 peft==0.18.1 numpy==2.2.6 pillow==12.0.0 \
      einops==0.8.2 timm==1.0.24 scipy==1.15.3 safetensors==0.7.0 tqdm==4.67.3 pyyaml huggingface_hub
    DS_BUILD_OPS=0 python -m pip install deepspeed==0.18.9 ;;
esac
python -c "import torch, transformers; print('$which', 'torch', torch.__version__, 'transformers', transformers.__version__)"
echo "ENV READY $which $(date)"
