#!/usr/bin/env bash
# xBD test + hold splits from the HF mirror (user decision 2026-10-16) -> $EOJEV_DATASETS/raw/xbd. CC-BY-NC-SA-4.0.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download("kshitijrajsharma/xview2-xbd", repo_type="dataset",
                      local_dir=os.environ["EOJEV_DATASETS"] + "/raw/xbd",
                      allow_patterns=["data/test-*", "data/hold-*", "README.md"], max_workers=8)
print("OK", p)
PY
echo "DONE $(date)"
