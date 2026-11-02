#!/usr/bin/env bash
# Run each pending converter as soon as its download is verified (the fetchers print "OK <repo>").
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh
source "$EOJEV_CONDA/etc/profile.d/conda.sh"
conda activate $EOJEV_ENVS/eojev
declare -A ready=(
  [disasterm3]="OK Kingdrone-Junjue/DisasterM3|logs/download_week2.log"
  [choice]="OK An-Xiao/CHOICE -> .* \(11056|logs/download_choice.log"
  [earthmind]="OK sy1998/EarthMind-Bench|logs/download_extra.log"
  [cdvqa]="OK ljx620/CDVQA|logs/download_extra.log"
  [rsvqa_hr]="OK rsvqa-hr/Images.tar|logs/download_rsvqa.log"
)
declare -A cmd=(
  [disasterm3]="python -m bench.converters.disasterm3" [choice]="python -m bench.converters.choice"
  [earthmind]="python -m bench.converters.earthmind" [cdvqa]="python -m bench.converters.cdvqa"
  [rsvqa_hr]="python -m bench.converters.rsvqa hr"
)
pending=("${!ready[@]}")
while [ ${#pending[@]} -gt 0 ]; do
  left=()
  for k in "${pending[@]}"; do
    pattern=${ready[$k]%|*}; log=${ready[$k]#*|}
    if grep -qE "$pattern" "$log" 2>/dev/null; then
      echo "$(date) converting $k"; ${cmd[$k]} >| "logs/convert_$k.log" 2>&1 && echo "$(date) done $k" || echo "$(date) FAILED $k"
    else
      left+=("$k")
    fi
  done
  pending=("${left[@]}")
  [ ${#pending[@]} -gt 0 ] && sleep 120
done
echo "ALL CONVERTED $(date)"
