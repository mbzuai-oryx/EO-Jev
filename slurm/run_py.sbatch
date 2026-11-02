#!/usr/bin/env bash
#SBATCH -J eojev-py
#SBATCH -p a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
# One Python module on one GPU, for checks that are not benchmark runs (e.g. bench.specialists.check_native):
#   ENVP=$EOJEV_ENVS/eo-earthdial MODULE=bench.specialists.check_native ARGS="earthdial 60" \
#     sbatch slurm/run_py.sbatch          (or SCRIPT=<path to a .py file> instead of MODULE)
set -uo pipefail
: "${MODULE:-}${SCRIPT:-}"; [ -n "${MODULE:-}${SCRIPT:-}" ] || { echo "set MODULE or SCRIPT"; exit 1; }
source env.sh
echo "$(hostname) GPU ${CUDA_VISIBLE_DEVICES:-} $(date)"
PY="${ENVP:-$EOJEV_ENVS/eojev}/bin/python"
if [ -n "${MODULE:-}" ]; then PYTHONPATH=$PWD $PY -m "$MODULE" ${ARGS:-}; else PYTHONPATH=$PWD $PY "$SCRIPT" ${ARGS:-}; fi
rc=$?
echo "EXIT $rc $(date)"
exit $rc  # so that --dependency=afterok sees a failed step
