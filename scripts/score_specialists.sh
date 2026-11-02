#!/usr/bin/env bash
# Score every specialist run (runs/specialists/<model>/<mode>/<rows name>) with the same scorer as d3:
# writes score_<by>.json next to the results, plus parse.json (share of generated answers that could not be parsed).
#   bash scripts/score_specialists.sh        (BOOT=0 for a quick pass; ROOT=<dir> to score another tree, e.g. runs/smoke;
#   runs with fewer than 64 finished chunks are skipped unless PARTIAL=1)
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh
PY=$EOJEV_ENVS/eojev/bin/python
C=$EOJEV_DATASETS/converted; BOOT=${BOOT:-200}
shopt -s nullglob
for dir in ${ROOT:-$EOJEV_RUNS/specialists}/*/*/*/; do
  dir=${dir%/}; name=$(basename "$dir")
  files=("$dir"/chunk*.jsonl); [ ${#files[@]} -eq 0 ] && continue
  done_markers=("$dir"/chunk*.done)
  if [ ${#done_markers[@]} -lt 64 ] && [ -z "${PARTIAL:-}" ]; then  # still running: never score a partial run
    rm -f "$dir"/score_*.json "$dir"/parse.json; continue
  fi
  # scored already and no result file changed since (FORCE=1 rescores everything)
  if [ -z "${FORCE:-}" ] && [ -e "$dir/parse.json" ] && \
     [ -z "$(find "$dir" -maxdepth 1 -name 'chunk*' -newer "$dir/parse.json" -print -quit)" ]; then
    continue
  fi
  rows=("$C"/*/"$name".jsonl); [ ${#rows[@]} -eq 1 ] || { echo "no unique rows file for $name"; continue; }
  bys="benchmark task_family"
  [[ $name == fmow* ]] && bys="$bys qid"
  [[ $name == choice* ]] && bys="$bys level1"
  [[ $name == disasterm3* ]] && bys="$bys post_modality"
  for by in $bys; do
    $PY -m bench.metrics.score --rows "${rows[0]}" --results "${files[@]}" --by "$by" --bootstrap "$BOOT" \
      --out "$dir/score_$by.json" > /dev/null || echo "FAILED $dir ($by)"
  done
  $PY -I - "$dir" "${files[@]}" <<'PYEOF'
import json, sys
out, files = sys.argv[1], sys.argv[2:]
n = bad = err = 0
for f in files:
    for line in open(f):
        r = json.loads(line)
        if r["status"] != "ok":
            err += 1
            continue
        for a in r["answers"].values():
            n += 1
            bad += bool(a.get("unparsed"))
json.dump({"questions": n, "unparsed": bad, "row_errors": err}, open(f"{out}/parse.json", "w"))
print(f"scored {out}: {n} questions, {bad} unparsed, {err} row errors")
PYEOF
done
