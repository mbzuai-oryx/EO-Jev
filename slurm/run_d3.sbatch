#!/usr/bin/env bash
#SBATCH -J eojev-d3
#SBATCH -p a100
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
# d3-flash over one or more rows files with 4 single-GPU tasks (run from the repo root):
#   ROWS="<a.jsonl> <b.jsonl>" OUT="<run dir a> <run dir b>" [EXTRA="--image-transform rot90"] sbatch slurm/run_d3.sbatch
#   per-entry options: EXTRAS="--model qwen3.5-9b|--blind" (one '|'-separated field per ROWS entry; overrides EXTRA)
#   fewer GPUs (e.g. to leave one for other work): sbatch --ntasks=3 slurm/run_d3.sbatch
#   an EXTRAS field starting with "@probe" runs bench.probes.extract (probe features) instead of the runner
# One job whose tasks may land on different nodes. Tasks claim chunks of each rows file dynamically
# (bench/runners/d3.py), so a task whose GPU does not initialize just exits and the others finish the file (dead GPUs
# do happen on shared clusters). A watchdog ends a step 10 min after all its chunks are done, so a task that hangs
# at launch cannot block the remaining entries.
set -uo pipefail
: "${ROWS:?set ROWS}" "${OUT:?set OUT}"
source env.sh
source "$EOJEV_CONDA/etc/profile.d/conda.sh"
conda activate $EOJEV_ENVS/eojev
export TRITON_CACHE_DIR=$EOJEV_MODELS_ROOT/triton_cache
# Some clusters' submit filters reject `sleep`; wait with a timed read on an idle pipe instead.
exec 9<> <(:)
snooze() { read -r -t "$1" -u 9 || true; }
read -r -a rows_list <<< "$ROWS"
read -r -a out_list <<< "$OUT"
[ "${#rows_list[@]}" -eq "${#out_list[@]}" ] || { echo "ROWS and OUT need the same number of entries"; exit 1; }
if [ -n "${EXTRAS:-}" ]; then
  IFS='|' read -r -a extras_list <<< "$EXTRAS"
  [ "${#extras_list[@]}" -eq "${#rows_list[@]}" ] || { echo "EXTRAS needs one '|' field per ROWS entry"; exit 1; }
fi

for k in "${!rows_list[@]}"; do
  out="${out_list[$k]}"
  mkdir -p "$out"
  # Locks without a .done marker belong to a run that died; release them so this run redoes those chunks
  # (each chunk file is resumed row by row, so nothing finished is recomputed).
  for lock in "$out"/chunk*.lock; do
    [ -e "$lock" ] && [ ! -e "${lock%.lock}.done" ] && \rm -f "$lock"
  done
  extra_k="${EXTRA:-}"
  [ -n "${EXTRAS:-}" ] && extra_k="${extras_list[$k]}"
  echo "== ${rows_list[$k]} -> $out [$extra_k] ($(date))"
  module=bench.runners.d3; n_chunks=64
  if [[ "$extra_k" == @probe* ]]; then  # probe feature extraction (bench/probes/extract.py) instead of a plain run
    module=bench.probes.extract; extra_k="${extra_k#@probe}"; n_chunks=32
  fi
  ROWS_K="${rows_list[$k]}" OUT_K="$out" EXTRA_K="$extra_k" MODULE_K="$module" srun --kill-on-bad-exit=0 bash -c \
    'python -m $MODULE_K --rows "$ROWS_K" --out-dir "$OUT_K" $EXTRA_K > "$OUT_K/task$SLURM_PROCID.log" 2>&1' &
  step=$!
  all_done_since=""
  while kill -0 "$step" 2>/dev/null; do
    shopt -s nullglob; markers=("$out"/chunk*.done); shopt -u nullglob
    if [ "${#markers[@]}" -ge "$n_chunks" ]; then
      all_done_since=${all_done_since:-$(date +%s)}
      if [ $(( $(date +%s) - all_done_since )) -ge 600 ]; then
        echo "   watchdog: all $n_chunks chunks done 10 min ago; ending the step's straggling tasks"
        kill -TERM "$step"; snooze 30; kill -KILL "$step" 2>/dev/null
      fi
    fi
    snooze 30
  done
  wait "$step" 2>/dev/null
  grep -h "finished\|load failed" "$out"/task*.log
  shopt -s nullglob; done_markers=("$out"/chunk*.done); shopt -u nullglob
  echo "   chunks done: ${#done_markers[@]}"
done
echo "ALL DONE $(date)"
