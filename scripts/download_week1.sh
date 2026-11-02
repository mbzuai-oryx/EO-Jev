#!/usr/bin/env bash
# Week-1 benchmark data -> $EOJEV_DATASETS/raw. Test splits / eval files only where the repo allows it.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
RAW=$EOJEV_DATASETS/raw
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
raw = os.environ["EOJEV_DATASETS"] + "/raw"
jobs = [
    ("MBZUAI/GeoChat-Bench", "geochat-bench", None),
    ("torchgeo/ucmerced", "ucmerced", None),
    ("isaaccorley/aid", "aid", None),
    ("jonathan-roberts1/WHU-RS19", "whu-rs19", None),
    ("jonathan-roberts1/SIRI-WHU", "siri-whu", None),
    ("timm/eurosat-rgb", "eurosat-rgb", ["data/test-*", "README.md"]),
    ("timm/resisc45", "resisc45", ["data/test-*", "README.md"]),
    ("aialliance/GEOBench-VLM", "geobench-vlm", ["Single.zip", "Single.parquet", "Temporal.zip", "README.md", "preprocess_xbd.py"]),
    ("PumpkinCat/LHRS_Data", "lhrs-bench", ["LHRS-Bench/*"]),
]
for repo, name, allow in jobs:
    path = snapshot_download(repo, repo_type="dataset", local_dir=f"{raw}/{name}", allow_patterns=allow, max_workers=8)
    print("OK", repo, "->", path, flush=True)
PY
# RSVQA (Zenodo, CC-BY-4.0)
mkdir -p $RAW/rsvqa-lr $RAW/rsvqa-hr
for f in Images_LR.zip LR_split_test_questions.json LR_split_test_answers.json LR_split_test_images.json; do
  [ -s $RAW/rsvqa-lr/$f ] || curl -sSL --retry 3 -o $RAW/rsvqa-lr/$f https://zenodo.org/api/records/6344334/files/$f/content
done
echo "OK rsvqa-lr"
for f in USGS_split_test_phili_questions.json USGS_split_test_phili_answers.json USGS_split_test_phili_images.json \
         USGS_split_test_questions.json USGS_split_test_answers.json USGS_split_test_images.json Images.tar; do
  [ -s $RAW/rsvqa-hr/$f ] || curl -sSL --retry 3 -o $RAW/rsvqa-hr/$f https://zenodo.org/api/records/6344367/files/$f/content
done
echo "OK rsvqa-hr"
echo "DONE $(date)"
