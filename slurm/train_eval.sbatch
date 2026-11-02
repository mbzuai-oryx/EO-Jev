#!/usr/bin/env bash
#SBATCH -J eojev-tev
#SBATCH -p a100
#SBATCH --ntasks=4
#SBATCH --nodes=1-4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
# One fine-tuning run end to end, in order (run from the repo root):
#   1. smoke       train.smoke_cpu: checks of the training rows: targets, sampler, prompt lengths
#   2. train       train.train: LoRA training (TRAIN_GPUS of one node) -> $EOJEV_RUNS/train/$NAME
#   3. export      train.export: merged package and its export check  -> $EOJEV_RUNS/packages/$NAME
#   4. benchmarks  slurm/run_d3.sbatch on all the job's GPUs: the 27-row suite (no SAR), counting, Decision Index
#                                                                      -> $EOJEV_RUNS/$NAME/<d3 run dir>
#   5. analyses    compare_package (vs d3-flash, and vs $PREV when set), diagnose_package, yesno_threshold
# The EO-Jev model of the paper (train/train.py's defaults), as two jobs: training on 4 GPUs of one node (torchrun;
# about 25 hours on A100s; if the time limit ends it first, submit it again: training resumes from its --out-dir),
# then the evaluation (STEPS picks the steps):
#   STEPS="smoke train export" TRAIN_GPUS=4 NAME=eo-jev sbatch -J eojev-train --ntasks=1 --nodes=1 \
#     --gpus-per-task=4 --cpus-per-task=32 -t 24:00:00 slurm/train_eval.sbatch
#   STEPS="benchmarks analyses" NAME=eo-jev sbatch -J eojev-eval --ntasks=3 --nodes=1-3 \
#     --dependency=afterok:<training job> slurm/train_eval.sbatch
# TRAIN_ARGS="..." overrides train.train options; PREV=<name> adds a comparison with another package.
# A merged adapter (train.merge_adapters) is exported with STEPS="export" ADAPTER=<merged file>.
# A failure in steps 1-3 ends the job with exit status 1.
set -uo pipefail
STEPS=${STEPS:-smoke train export benchmarks analyses}
has() { [[ " $STEPS " == *" $1 "* ]]; }
: "${NAME:?set NAME}"
TRAIN_ARGS=${TRAIN_ARGS:-}
source env.sh
source "$EOJEV_CONDA/etc/profile.d/conda.sh"
conda activate $EOJEV_ENVS/eojev
export TRITON_CACHE_DIR=$EOJEV_MODELS_ROOT/triton_cache PYTHONPATH=$PWD OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=$EOJEV_RUNS
CV=$EOJEV_DATASETS/converted
PKG=$R/packages/$NAME
one() { srun --ntasks=1 --nodes=1 --kill-on-bad-exit=1 "$@"; }
step() { echo "== $1 $(date)"; }
echo "$(hostname) $NAME [$STEPS]: ${TRAIN_ARGS:-}"

if has smoke; then
  step smoke
  one python -m train.smoke_cpu || { echo "smoke checks failed"; exit 1; }
fi

if has train; then
  step train
  if [ "${TRAIN_GPUS:-1}" -gt 1 ]; then  # one task holding TRAIN_GPUS GPUs (--gpus-per-task=TRAIN_GPUS)
    one torchrun --standalone --nproc_per_node=$TRAIN_GPUS -m train.train --out-dir "$R/train/$NAME" $TRAIN_ARGS \
      || { echo "training failed"; exit 1; }
  else
    one python -m train.train --out-dir "$R/train/$NAME" $TRAIN_ARGS || { echo "training failed"; exit 1; }
  fi
fi

if has export; then
  step export
  # ADAPTER=<file> exports a given adapter instead of the run's last one (e.g. one from train.merge_adapters)
  ADAPTER=${ADAPTER:-$(printf '%s\n' "$R/train/$NAME"/adapter_step*.safetensors | sort -V | sed -n '$p')}
  echo "   adapter: $ADAPTER"
  one python -m train.export --adapter "$ADAPTER" --out "$PKG" || { echo "export or its check failed"; exit 1; }
fi

if has benchmarks; then
  step benchmarks
  [ -f "$PKG/EOJEV_EXPORT.json" ] || { echo "no exported package in $PKG"; exit 1; }
  pairs="decision_index/parity_rows:decision_index det_count/det_count_nwpu_vhr10:det_count_nwpu_vhr10
det_count/det_count_dota_v2_val:det_count_dota_v2_val det_count/det_count_vhr10_absent:det_count_vhr10_absent
xbd/xbd_damage_3k:xbd_damage xbd/xbd_disaster_type_xbdlabels:xbd_type_xbd scene/scene_all:scene_nway
fmow/fmow_wilds_ood_2k:fmow_wilds vhm/vhm_hnstd:vhm_hnstd geobench_vlm/geobench_vlm_single:geobench_vlm
lhrs_bench/lhrs_bench_circular:lhrs_bench urbench/urbench_geo:urbench_geo earthmind/earthmind_mc_optical:earthmind_optical
cdvqa/cdvqa_test2_5k:cdvqa rsvqa/rsvqa_hr_test2_5k:rsvqa_hr rsvqa/rsvqa_lr:rsvqa_lr bigearthnet/bigearthnet_v2_test_1k:bigearthnet
choice/choice_mc:choice vrsbench/vrsbench_vqa_mc_5k:vrsbench_vqa_mc dvl/dvl_choice:dvl disasterm3/disasterm3_mc_5k:disasterm3"
  rows=""; outs=""; extras=""
  for pr in $pairs; do
    rows="$rows $CV/${pr%%:*}.jsonl"; outs="$outs $R/$NAME/${pr##*:}"; extras="$extras|--model $PKG"
  done
  ROWS="${rows# }" OUT="${outs# }" EXTRAS="${extras#|}" bash slurm/run_d3.sbatch
fi

if has analyses; then
  step analyses
  python -m bench.analysis.compare_package "$NAME"
  [ -n "${PREV:-}" ] && python -m bench.analysis.compare_package "$NAME" --baseline "$PREV"
  python -m bench.analysis.diagnose_package "$NAME"
  python -m bench.analysis.yesno_threshold "$NAME"
fi
echo "ALL STEPS DONE [$STEPS] $(date)"
