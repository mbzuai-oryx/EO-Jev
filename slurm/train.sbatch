#!/usr/bin/env bash
#SBATCH -J eojev-train
#SBATCH -p a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
# LoRA training of d3-flash on one GPU (run from the repo root):
#   ARGS="--out-dir $EOJEV_RUNS/train/eo-jev" sbatch slurm/train.sbatch
# Resumable: a rerun with the same --out-dir continues from its state.pt.
set -uo pipefail
: "${ARGS:?set ARGS}"
source env.sh
source "$EOJEV_CONDA/etc/profile.d/conda.sh"
conda activate $EOJEV_ENVS/eojev
export TRITON_CACHE_DIR=$EOJEV_MODELS_ROOT/triton_cache PYTHONPATH=$PWD OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$(hostname) GPU ${CUDA_VISIBLE_DEVICES:-} $(date)"
python -m train.train $ARGS
echo "EXIT $? $(date)"
