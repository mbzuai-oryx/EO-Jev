#!/usr/bin/env bash
# CHOICE images (resume after a dropped connection) -> $EOJEV_DATASETS/raw/choice.
set -uo pipefail
source "$(dirname "$0")/../env.sh"
source "$EOJEV_CONDA/etc/profile.d/conda.sh"
conda activate $EOJEV_ENVS/eojev
export HF_HUB_DISABLE_XET=1
python "$(dirname "$0")/hf_fetch.py" An-Xiao/CHOICE choice
echo "DONE $(date)"
