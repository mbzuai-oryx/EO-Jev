#!/usr/bin/env bash
#SBATCH -J eojev-spec
#SBATCH -p a100
#SBATCH --ntasks=3
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH -t 48:00:00
#SBATCH -o logs/%x-%j.out
# EO specialist models (bench/runners/vlm.py) over one or more rows files, 3 single-GPU tasks by default (the user's
# other project keeps one GPU job on this account, so this leaves its slot free). From the repo root:
#   ROWS="<a.jsonl> <b.jsonl>" OUT="<run dir a> <run dir b>" EXTRAS="--model geozero --mode logprob|--model ..." \
#     [ENVP=$EOJEV_ENVS/eo-sa2va] sbatch slurm/run_vlm.sbatch
# An EXTRAS field may start with "@env=<name>" to run that entry with $EOJEV_ENVS/<name>.
# Same chunk claiming, dead-GPU handling and watchdog as slurm/run_d3.sbatch (see there).
set -uo pipefail
: "${ROWS:?set ROWS}" "${OUT:?set OUT}"
source env.sh
source "$EOJEV_CONDA/etc/profile.d/conda.sh"
conda activate "${ENVP:-$EOJEV_ENVS/eojev}"
export TRITON_CACHE_DIR=$EOJEV_MODELS_ROOT/triton_cache
# The cluster submit filter rejects `sleep`; wait with a timed read on an idle pipe instead.
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
  if [ -n "${REOPEN:-}" ]; then  # repair pass: drop error records and reopen their chunks; skip finished entries
    python scripts/reopen_failed_chunks.py "$out" && { echo "== $out complete, skipped"; continue; }
  fi
  # Locks without a .done marker belong to a run that died; release them so this run redoes those chunks
  # (each chunk file is resumed row by row, so nothing finished is recomputed).
  for lock in "$out"/chunk*.lock; do
    [ -e "$lock" ] && [ ! -e "${lock%.lock}.done" ] && \rm -f "$lock"
  done
  extra_k="${EXTRA:-}"
  [ -n "${EXTRAS:-}" ] && extra_k="${extras_list[$k]}"
  echo "== ${rows_list[$k]} -> $out [$extra_k] ($(date))"
  module=bench.runners.vlm; n_chunks=64
  py=python
  if [[ "$extra_k" =~ @env=([^[:space:]]+) ]]; then  # per-entry Python env, e.g. "@env=eo-sa2va --model terrascope ..."
    py=$EOJEV_ENVS/${BASH_REMATCH[1]}/bin/python
    extra_k="${extra_k/@env=${BASH_REMATCH[1]}/}"
  fi
  ROWS_K="${rows_list[$k]}" OUT_K="$out" EXTRA_K="$extra_k" MODULE_K="$module" PY_K="$py" srun --kill-on-bad-exit=0 bash -c \
    '$PY_K -m $MODULE_K --rows "$ROWS_K" --out-dir "$OUT_K" $EXTRA_K > "$OUT_K/task$SLURM_PROCID.log" 2>&1' &
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
