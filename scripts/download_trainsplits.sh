#!/usr/bin/env bash
# Training (and validation) splits of the benchmark sources (every eval image is kept out by the leak check). Test splits are not fetched here. Files land next to the eval downloads
# in $EOJEV_DATASETS/raw/<name>. PART picks one group, so the big ones can run as separate jobs:
#   cdvqa       CDVQA train + val webdataset shards (SECOND pairs; ~66 GB)
#   disasterm3  DisasterM3 Instruct: train_release.json, train_images.zip, box_train_images.zip (~33 GB; no masks)
#   rest        VRSBench train, DVL-Instruct (train.zip), RSVQA-LR / HR train + val question files
#   sbatch --wrap "PART=cdvqa bash scripts/download_trainsplits.sh" ...
set -uo pipefail
source "$(dirname "$0")/../env.sh"
source "$EOJEV_CONDA/etc/profile.d/conda.sh"
conda activate $EOJEV_ENVS/eojev
export HF_HUB_DISABLE_XET=1
F="python $(dirname "$0")/hf_fetch.py"
RAW=$EOJEV_DATASETS/raw
case ${PART:?set PART} in
  cdvqa) $F ljx620/CDVQA cdvqa-wds --allow "train/*" "val/*" manifest.json --rate 2 ;;
  disasterm3)
    $F Kingdrone-Junjue/DisasterM3 disasterm3 --allow "DisasterM3_Instruct/train_release.json" \
      "DisasterM3_Instruct/train_images.zip" "DisasterM3_Instruct/box_train_images.zip" --rate 2 ;;
  rest)
    $F xiang709/VRSBench vrsbench --allow VRSBench_train.json Images_train.zip Annotations_train.zip --rate 2
    $F weihao1115/dvl_suite dvl --allow train.zip --rate 2
    for f in LR_split_train_questions.json LR_split_train_answers.json LR_split_train_images.json \
             LR_split_val_questions.json LR_split_val_answers.json LR_split_val_images.json; do
      [ -s $RAW/rsvqa-lr/$f ] || curl -sSL --retry 3 -o $RAW/rsvqa-lr/$f https://zenodo.org/api/records/6344334/files/$f/content
    done
    for f in USGS_split_train_questions.json USGS_split_train_answers.json USGS_split_train_images.json \
             USGS_split_val_questions.json USGS_split_val_answers.json USGS_split_val_images.json; do
      [ -s $RAW/rsvqa-hr/$f ] || curl -sSL --retry 3 -o $RAW/rsvqa-hr/$f https://zenodo.org/api/records/6344367/files/$f/content
    done
    ls -la $RAW/rsvqa-lr $RAW/rsvqa-hr | grep -E "train|val" ;;
  *) echo "unknown PART $PART"; exit 1 ;;
esac
echo "DONE $PART $(date)"
