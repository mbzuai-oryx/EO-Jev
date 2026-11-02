#!/usr/bin/env bash
# Phase 0: pull the subject model and its exact base at pinned revisions.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
for repo, rev in [("vllm-sr/d3-flash", os.environ["D3_FLASH_REV"]),
                  ("Qwen/Qwen3.5-9B", os.environ["QWEN35_9B_REV"])]:
    path = snapshot_download(repo, revision=rev, max_workers=8)
    print(repo, rev, "->", path, flush=True)
PY
echo "DONE $(date)"
