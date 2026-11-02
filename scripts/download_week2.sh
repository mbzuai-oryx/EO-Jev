#!/usr/bin/env bash
# Week-2/3 core benchmark data (evaluation splits only) -> $EOJEV_DATASETS/raw, via the paced fetcher.
# Needs a valid HF login (anonymous requests get rate-limited) and the fMoW gate accepted. Each repo is verified;
# a failed repo is reported and the next one still runs. Rerunning resumes.
set -uo pipefail
source "$(dirname "$0")/../env.sh"
source "$EOJEV_CONDA/etc/profile.d/conda.sh"
conda activate $EOJEV_ENVS/eojev
export HF_HUB_DISABLE_XET=1
F="python $(dirname "$0")/hf_fetch.py"
$F timm/bigearthnet-v2-rgb bigearthnet-v2-rgb --allow "data/test-*" README.md
$F xiang709/VRSBench vrsbench --allow "VRSBench_EVAL_*" Images_val.zip Annotations_val.zip README.md
$F danielz01/fMoW fmow-wilds --allow "WILDS/test-*" README.md
$F weihao1115/dvl_suite dvl --allow test.zip README.md
$F initiacms/XLRS-Bench-lite xlrs-lite
$F Kingdrone-Junjue/DisasterM3 disasterm3 --allow "DisasterM3_Bench/*" README.md
$F bczhou/UrBench urbench --allow "city-retrieval/*" "image-retrieval/*" "camera-localization/*" "orientation/*" README.md
$F An-Xiao/CHOICE choice          # 11k files: ~1 h at the paced rate, so last
echo "DONE $(date)"
