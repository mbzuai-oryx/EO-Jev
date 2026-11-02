# Source before running anything in eo-jev.
# Machine-specific paths live in env.local.sh next to this file (git-ignored): copy env.local.sh.example to
# env.local.sh and set them there. Models/runs and datasets can be two folders or the same one.
_eojev_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -f "$_eojev_dir/env.local.sh" ]; then
  source "$_eojev_dir/env.local.sh"
else
  echo "eo-jev: $_eojev_dir/env.local.sh not found; copy env.local.sh.example to it and set your paths" >&2
fi
unset _eojev_dir
: "${EOJEV_MODELS_ROOT:?set EOJEV_MODELS_ROOT in env.local.sh}" "${EOJEV_DATASETS:?set EOJEV_DATASETS in env.local.sh}"
: "${EOJEV_ENVS:?set EOJEV_ENVS in env.local.sh}"
export EOJEV_MODELS_ROOT EOJEV_DATASETS EOJEV_ENVS
export EOJEV_CONDA=${EOJEV_CONDA:-$HOME/miniconda3}      # conda install whose etc/profile.d/conda.sh the scripts source
export EOJEV_RUNS=$EOJEV_MODELS_ROOT/runs
export HF_HUB_CACHE=$EOJEV_MODELS_ROOT/hf_cache/hub       # model repos only
export HF_XET_CACHE=$EOJEV_MODELS_ROOT/hf_cache/xet
export HF_DATASETS_CACHE=$EOJEV_DATASETS/hf_datasets      # datasets.load_dataset() cache
# Dataset repos must NOT use HF_HUB_CACHE (that is with the models): download them with
# snapshot_download(..., repo_type="dataset", cache_dir=$EOJEV_DATASET_HUB) or local_dir under $EOJEV_DATASETS/raw.
export EOJEV_DATASET_HUB=$EOJEV_DATASETS/hf_hub
export D3_FLASH_REV=v3.1.0                                   # revision evaluated by the Decision Index boards
export QWEN35_9B_REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a  # d3-flash's exact base revision
