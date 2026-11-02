#!/usr/bin/env bash
#SBATCH -J eojev-convert
#SBATCH -p a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
# A question-file converter (bench/converters/<name>.py) as a CPU job, so long conversions survive the login
# session (docs/06; converter spec in $EOJEV_MODELS_ROOT/notes/converter_spec.md):
#   MODULE=bench.converters.rshr [ARGS="..."] sbatch slurm/convert_cpu.sbatch
set -uo pipefail
: "${MODULE:?set MODULE}"
source env.sh
echo "$(hostname) $(date) $MODULE ${ARGS:-}"
PYTHONPATH=$PWD "$EOJEV_ENVS/eojev/bin/python" -m "$MODULE" ${ARGS:-}
rc=$?
echo "EXIT $rc $(date)"
exit $rc  # so that --dependency=afterok sees a failed step
