#!/usr/bin/env bash
# EarthMind-Bench (C11) and the VHM eval set (C14: honesty / hallucination probes) -> $EOJEV_DATASETS/raw.
# Runs alongside download_week2.sh at a lower request rate (both back off on 429).
set -uo pipefail
source "$(dirname "$0")/../env.sh"
source "$EOJEV_CONDA/etc/profile.d/conda.sh"
conda activate $EOJEV_ENVS/eojev
export HF_HUB_DISABLE_XET=1
F="python $(dirname "$0")/hf_fetch.py"
$F FitzPC/VHM_eval_dataset vhm-eval --rate 2
$F sy1998/EarthMind-Bench earthmind-bench --allow "json/*_unmatched.json" "img/test/*" README.md --rate 2
$F ljx620/CDVQA cdvqa-wds --allow "test2/*" manifest.json --rate 2   # CDVQA test-2 with SECOND image pairs (25 GB)
echo "DONE $(date)"
