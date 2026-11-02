#!/usr/bin/env bash
# RSVQA LR + HR eval files from Zenodo (CC-BY-4.0) -> $EOJEV_DATASETS/raw/rsvqa-{lr,hr}. Retries through Zenodo timeouts.
set -uo pipefail
source "$(dirname "$0")/../env.sh"
RAW=$EOJEV_DATASETS/raw
get() {  # record dir file
  local out=$RAW/$2/$3
  for attempt in 1 2 3 4 5 6; do
    curl -sSL --connect-timeout 60 --retry 5 --retry-all-errors --retry-delay 20 -C - -o "$out" \
      "https://zenodo.org/api/records/$1/files/$3/content" && { echo "OK $2/$3"; return 0; }
    echo "retry $attempt $2/$3"; sleep 30
  done
  echo "FAILED $2/$3"; return 1
}
mkdir -p $RAW/rsvqa-lr $RAW/rsvqa-hr
for f in Images_LR.zip LR_split_test_questions.json LR_split_test_answers.json LR_split_test_images.json; do get 6344334 rsvqa-lr $f; done
for f in USGS_split_test_phili_questions.json USGS_split_test_phili_answers.json USGS_split_test_phili_images.json \
         USGS_split_test_questions.json USGS_split_test_answers.json USGS_split_test_images.json Images.tar; do get 6344367 rsvqa-hr $f; done
echo "DONE $(date)"
