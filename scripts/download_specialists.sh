#!/usr/bin/env bash
# SOTA re-measurement: open-weight EO specialist checkpoints -> $HF_HUB_CACHE (on /l),
# authors' inference code -> $EOJEV_MODELS_ROOT/third_party/<name> (untrusted code: never run from inside that tree
# except through the model loaders that need it).
set -euo pipefail
source "$(dirname "$0")/../env.sh"
export HF_HUB_DISABLE_XET=1
PY=$EOJEV_ENVS/eojev/bin/python
$PY -I - <<'PY'
from huggingface_hub import snapshot_download
skip = ["*.h5", "*.msgpack", "*.ot", "__pycache__/*", "*.pyc"]
repos = [
    ("hjvsl/GeoZero", ["GeoZero-8B-without-RFT/*", "*.md"]),
    ("minglanga/RSThinker", None), ("PumpkinCat/Qwen2VL-7B-RS-R1", None), ("miniHui/Geo-R1", None),
    ("sy1998/EarthMind-4B", None), ("sy1998/EarthMind4B_multi", None), ("sy1998/TerraScope", None),
    ("akshaydudhane/EarthDial_4B_RGB", None), ("Davidup1/GeoChrono", None), ("Qwen/Qwen3-VL-4B-Instruct", None),
    ("FitzPC/vhm_7B", None), ("openai/clip-vit-large-patch14-336", None),
]
for repo, allow in repos:
    for attempt in range(5):
        try:
            path = snapshot_download(repo, allow_patterns=allow, ignore_patterns=skip, max_workers=4)
            print("OK", repo, "->", path, flush=True)
            break
        except Exception as e:  # noqa: BLE001 - transient network errors
            print("retry", repo, attempt, type(e).__name__, str(e)[:200], flush=True)
    else:
        print("FAILED", repo, flush=True)
PY
TP=$EOJEV_MODELS_ROOT/third_party
mkdir -p "$TP"
for spec in shuyansy/Earth-Observation-VLMs hiyamdebary/EarthDial IntelliSensing/GeoChrono opendatalab/VHM MiliLab/GeoZero; do
  name=${spec#*/}
  [ -d "$TP/$name/.git" ] || git clone --depth 1 "https://github.com/$spec.git" "$TP/$name" || echo "CLONE FAILED $spec"
done
echo "DONE $(date)"
