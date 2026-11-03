# Phase-1 SOTA re-measurement: model -> "python env|rows files under converted/".
# Sourced by scripts/specialist_runs.sh.
plan() {
  case $1 in
    geozero)         echo "eojev|scene/scene_all choice/choice_mc geobench_vlm/geobench_vlm_single rsvqa/rsvqa_hr_test2_5k vhm/vhm_hnstd fmow/fmow_wilds_ood_2k" ;;
    rsthinker)       echo "eojev|scene/scene_all vrsbench/vrsbench_vqa_mc_5k" ;;
    scorers-r1)      echo "eojev|lhrs_bench/lhrs_bench_circular" ;;
    geo-r1)          echo "eojev|fmow/fmow_wilds_ood_2k urbench/urbench_geo" ;;
    terrascope)      echo "eo-sa2va|xbd/xbd_damage_3k xbd/xbd_disaster_type xbd/xbd_disaster_type_xbdlabels disasterm3/disasterm3_mc_5k" ;;
    earthmind)       echo "eo-sa2va|earthmind/earthmind_mc_optical earthmind/earthmind_mc_sar bigearthnet/bigearthnet_v2_test_1k" ;;
    earthmind-multi) echo "eo-sa2va|earthmind/earthmind_mc_fused" ;;
    earthdial)       echo "eo-earthdial|rsvqa/rsvqa_lr xbd/xbd_disaster_type xbd/xbd_disaster_type_xbdlabels xbd/xbd_damage_3k" ;;
    geochrono)       echo "eo-geochrono|cdvqa/cdvqa_test2_5k dvl/dvl_choice" ;;
    vhm)             echo "eo-vhm|vhm/vhm_hnstd" ;;
    waybob-xview2)   echo "eojev|xbd/xbd_disaster_type xbd/xbd_disaster_type_xbdlabels" ;;
    teochat)         echo "eo-teochat|xbd/xbd_disaster_type xbd/xbd_disaster_type_xbdlabels xbd/xbd_damage_3k" ;;
    # eval-v1.0 reference models
    uhr-bat2)        echo "eojev|rshr/rshr_v0_downscale rshr/rshr_v1_tiles mme_rw_rs/mme_rw_rs_v0_downscale" ;;
    rs-opsd)         echo "eojev|mme_rw_rs/mme_rw_rs_v0_downscale mme_rw_rs/mme_rw_rs_v1_tiles" ;;
    sky-vlm)         echo "eojev|spatialsky/spatialsky_closed" ;;
    sis-motion)      echo "eo-sismotion|sis_bench/sis_bench" ;;
    *) echo "unknown model $1" >&2; return 1 ;;
  esac
}
# Rows files with a native-format variant (bench/specialists/native.py), run by the "-native" modes.
plan_native() {
  case $1 in
    earthdial|teochat) echo "xbd/xbd_disaster_type xbd/xbd_damage_3k" ;;
    waybob-xview2)     echo "xbd/xbd_disaster_type" ;;
    *) echo "" ;;
  esac
}
