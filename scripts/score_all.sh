#!/usr/bin/env bash
# Score every finished run: writes <run dir>/score_<by>.json next to the results (skips runs without results).
#   bash scripts/score_all.sh            (bootstrap 200; set BOOT=0 for a quick pass)
set -uo pipefail
cd "$(dirname "$0")/.."
source env.sh
source "$EOJEV_CONDA/etc/profile.d/conda.sh"
conda activate $EOJEV_ENVS/eojev
C=$EOJEV_DATASETS/converted; R=$EOJEV_RUNS; A=$C/ablations; BOOT=${BOOT:-200}

score() {  # rows run_dir by
  local rows=$1 dir=$2 by=$3
  shopt -s nullglob; local files=("$dir"/chunk*.jsonl "$dir"/retry/chunk*.jsonl); shopt -u nullglob
  [ ${#files[@]} -eq 0 ] && return
  python -m bench.metrics.score --rows "$rows" --results "${files[@]}" --by "$by" --bootstrap "$BOOT" \
    --out "$dir/score_$by.json" > /dev/null && echo "scored $dir ($by)"
}

for m in d3 qwen3.5-9b d3_blind; do
  score $C/scene/scene_all.jsonl $R/$m/scene_nway benchmark
  score $C/scene/scene_16way.jsonl $R/$m/scene_16way benchmark
  score $C/geobench_vlm/geobench_vlm_single.jsonl $R/$m/geobench_vlm benchmark
  score $C/geobench_vlm/geobench_vlm_single.jsonl $R/$m/geobench_vlm task
  score $C/rsvqa/rsvqa_lr.jsonl $R/$m/rsvqa_lr task_family
  score $C/rsvqa/rsvqa_hr_test2_5k.jsonl $R/$m/rsvqa_hr task_family
  score $C/lhrs_bench/lhrs_bench_circular.jsonl $R/$m/lhrs_bench benchmark
  score $C/bigearthnet/bigearthnet_v2_test_1k.jsonl $R/$m/bigearthnet benchmark
  score $C/vrsbench/vrsbench_vqa_mc_5k.jsonl $R/$m/vrsbench_vqa_mc task_family
  score $C/xbd/xbd_damage_3k.jsonl $R/$m/xbd_damage benchmark
  score $C/xbd/xbd_disaster_type.jsonl $R/$m/xbd_type benchmark
  score $C/dvl/dvl_choice.jsonl $R/$m/dvl task_family
  score $C/fmow/fmow_wilds_ood_2k.jsonl $R/$m/fmow_wilds qid
  score $C/vhm/vhm_hnstd.jsonl $R/$m/vhm_hnstd task_family
  score $C/urbench/urbench_geo.jsonl $R/$m/urbench_geo task_family
  for v in optical sar fused; do
    score $C/earthmind/earthmind_mc_$v.jsonl $R/$m/earthmind_$v benchmark
    score $C/earthmind/earthmind_mc_$v.jsonl $R/$m/earthmind_$v task_family
  done
  score $C/cdvqa/cdvqa_test2_5k.jsonl $R/$m/cdvqa task_family
  score $C/xlrs/xlrs_lite_v0_downscale.jsonl $R/$m/xlrs_v0 benchmark
  score $C/xlrs/xlrs_lite_v0_downscale.jsonl $R/$m/xlrs_v0 task_family
  score $C/xlrs/xlrs_lite_v1_tiles.jsonl $R/$m/xlrs_v1 benchmark
  score $C/xlrs/xlrs_lite_v1_tiles.jsonl $R/$m/xlrs_v1 task_family
  score $C/disasterm3/disasterm3_mc_5k.jsonl $R/$m/disasterm3 task_family
  score $C/disasterm3/disasterm3_mc_5k.jsonl $R/$m/disasterm3 post_modality
  score $C/choice/choice_mc.jsonl $R/$m/choice level1
  score $C/choice/choice_mc.jsonl $R/$m/choice task_family
done

for m in d3 qwen3.5-9b; do  # robustness
  score $A/scene_sub1k_perm3.jsonl $R/robust/$m/scene_perm3 benchmark
  score $A/geobench_sub1k_perm3.jsonl $R/robust/$m/geobench_perm3 benchmark
done
for t in rot90 rot180 rot270 hflip; do
  score $A/scene_sub1k.jsonl $R/robust/d3/scene_$t benchmark
  score $A/geobench_sub1k.jsonl $R/robust/d3/geobench_$t benchmark
done
for t in down2 down4 permavg; do  # ablations
  score $A/scene_sub1k.jsonl $R/ablation/d3/scene_$t benchmark
  score $A/geobench_sub1k.jsonl $R/ablation/d3/geobench_$t benchmark
done
score $A/bigearthnet_state.jsonl $R/ablation/d3/bigearthnet_state benchmark
score $A/scene_sub1k_state.jsonl $R/ablation/d3/scene_state benchmark
for k in 4 8 16; do score $A/scene_manyclass_${k}way.jsonl $R/ablation/d3/scene_${k}way benchmark; done
echo "ALL SCORED $(date)"
