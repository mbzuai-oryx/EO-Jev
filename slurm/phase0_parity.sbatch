#!/usr/bin/env bash
#SBATCH -J eojev-parity
#SBATCH -p a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
# Phase 0: run the official engine (d3_engine:D3Engine) on rebuilt Decision Index rows and compare with the
# official d3-flash results (run from the repo root: sbatch slurm/phase0_parity.sbatch).
set -euo pipefail
source env.sh
source "$EOJEV_CONDA/etc/profile.d/conda.sh"
conda activate $EOJEV_ENVS/eojev
export TRITON_CACHE_DIR=$EOJEV_MODELS_ROOT/triton_cache
SNAP=$HF_HUB_CACHE/models--vllm-sr--d3-flash/snapshots/521525effa79469d899a570726547dd7f00da3af
OFFICIAL=$(ls -d $EOJEV_DATASET_HUB/datasets--vllm-sr--d3-flash-decision-index/snapshots/*)/runs/d3-flash/results.jsonl.gz
ROWS=$EOJEV_DATASETS/raw/decision-index-work/parity-rows.jsonl
OUT=$EOJEV_RUNS/phase0/parity
PYTHONPATH=$SNAP python -m decision_index run --engine d3_engine:D3Engine \
    --option model=$SNAP --option device=cuda:0 --rows "$ROWS" --out "$OUT"
python bench/parity_compare.py "$OFFICIAL" "$OUT/results.jsonl" "$EOJEV_RUNS/phase0/parity_report.json"
