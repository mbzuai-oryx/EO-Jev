#!/usr/bin/env bash
#SBATCH -J eojev-env
#SBATCH -p a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
# The main env (eojev) as a CPU job (run from the repo root: sbatch slurm/setup_env.sbatch): scripts/setup_env.sh,
# then the installed package list -> $EOJEV_ENVS/eojev-freeze.txt
set -euo pipefail
source env.sh
echo "$(hostname) $(date)"
mkdir -p "$EOJEV_ENVS"
bash scripts/setup_env.sh
"$EOJEV_ENVS/eojev/bin/python" -m pip freeze > "$EOJEV_ENVS/eojev-freeze.txt"
echo "EXIT 0 $(date)"
