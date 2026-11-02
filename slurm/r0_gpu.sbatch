#!/usr/bin/env bash
#SBATCH -J eojev-r0
#SBATCH -p a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
# Trainer checks on one GPU, in order (run from the repo root: sbatch slurm/r0_gpu.sbatch):
#   parity  train.parity: gate R0, the trainer's forward pass vs the shipped runtime, plus gradient and merge checks
#   replay  train.label_replay: d3-flash's own probabilities on the replay rows
#   d1      probe D1: bench.probes.d1_rows, then bench.probes.extract on 3 sets, then bench.probes.fit
#   rerun   after a move to new hardware: d3-flash again on 4 stored sets (text, one image, two images, yes/no),
#           compared question by question with the stored runs (bench.analysis.rerun_check) -> $EOJEV_RUNS/rerun
# STEPS="parity replay" runs a subset. The job's exit status is 0 only if parity, replay and rerun all succeeded.
set -uo pipefail
source env.sh
source "$EOJEV_CONDA/etc/profile.d/conda.sh"
conda activate $EOJEV_ENVS/eojev
export TRITON_CACHE_DIR=$EOJEV_MODELS_ROOT/triton_cache PYTHONPATH=$PWD OMP_NUM_THREADS=8
echo "$(hostname) GPU ${CUDA_VISIBLE_DEVICES:-} $(date)"
D1=$EOJEV_DATASETS/converted/probes/d1
status=0
for step in ${STEPS:-parity replay d1}; do
  echo "== $step $(date)"
  case $step in
    parity) python -m train.parity || status=1 ;;
    replay) python -m train.label_replay || status=1 ;;
    d1)
      python -m bench.probes.d1_rows
      for set in d1_count_dior d1_cdvqa_ratio d1_xbd_damage; do
        python -m bench.probes.extract --rows "$D1/$set.jsonl" --out-dir "$EOJEV_RUNS/probes/d1/$set" --chunks 8
      done
      python -m bench.probes.fit "$EOJEV_RUNS/probes/d1/d1_count_dior" "$EOJEV_RUNS/probes/d1/d1_cdvqa_ratio" \
        "$EOJEV_RUNS/probes/d1/d1_xbd_damage" ;;
    rerun)
      pairs=""
      for pr in decision_index/parity_rows:decision_index det_count/det_count_nwpu_vhr10:det_count_nwpu_vhr10 \
                xbd/xbd_damage_3k:xbd_damage vhm/vhm_hnstd:vhm_hnstd; do
        python -m bench.runners.d3 --rows "$EOJEV_DATASETS/converted/${pr%%:*}.jsonl" --out-dir "$EOJEV_RUNS/rerun/${pr##*:}"
        pairs="$pairs $EOJEV_RUNS/d3/${pr##*:}:$EOJEV_RUNS/rerun/${pr##*:}"
      done
      python -m bench.analysis.rerun_check --out "$EOJEV_RUNS/rerun/report.json" $pairs || status=1 ;;
  esac
  echo "   $step finished $(date)"
done
echo "EXIT $status $(date)"
exit $status
