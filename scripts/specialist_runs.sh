#!/usr/bin/env bash
# Phase-1 SOTA re-measurement (docs/05_specialist_models.md): which specialist runs on which rows file.
# Builds ONE slurm/run_vlm.sbatch job (3 GPUs) running the given models x modes one entry after another; each entry
# uses its model's Python env (@env=...). One job because this account's GPU QoS allows 2 running jobs and the
# user's other project holds one.
#   scripts/specialist_runs.sh "<model> <model> ..." [logprob|generate|both] [--dry]
# Modes: logprob, generate, both, or logprob-native / generate-native / native (the model's own prompt format,
# bench/specialists/native.py, on the files plan_native lists; results in <model>/<mode>-native/), or all. FILES="a/b c/d" overrides the plan's row files.
# Repair pass after a run (error records dropped and redone, finished entries skipped), e.g. queued behind job N:
#   REOPEN=1 SBATCH_ARGS="--dependency=afterany:N" scripts/specialist_runs.sh "<all models>" both
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh
C=$EOJEV_DATASETS/converted

source scripts/specialist_plan.sh

models=$1; modes=${2:-both}; dry=${3:-}
[ "$modes" = both ] && modes="logprob generate"
[ "$modes" = native ] && modes="logprob-native generate-native"
[ "$modes" = all ] && modes="logprob generate logprob-native generate-native"
rows=""; outs=""; extras=""
for mode in $modes; do
  for model in $models; do
    p=$(plan "$model"); env=${p%%|*}
    flag=""; [[ $mode == *-native ]] && flag=" --native"
    files=${p#*|}; [ -n "$flag" ] && files=$(plan_native "$model")
    for f in ${FILES:-$files}; do
      rows+="$C/$f.jsonl "; outs+="$EOJEV_RUNS/specialists/$model/$mode/$(basename "$f") "
      extras+="@env=$env --model $model --mode ${mode%-native}$flag|"
    done
  done
done
cmd=(env ROWS="${rows% }" OUT="${outs% }" EXTRAS="${extras%|}" sbatch -J eojev-spec ${SBATCH_ARGS:-} slurm/run_vlm.sbatch)
if [ -n "$dry" ]; then
  echo "${rows% }" | tr ' ' '\n' | paste - <(echo "${extras%|}" | tr '|' '\n') | sed "s#$C/##"
else
  "${cmd[@]}"
fi
