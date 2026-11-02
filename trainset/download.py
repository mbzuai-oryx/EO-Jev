"""Download the training sources. Training splits only: no eval split is
fetched, apart from small label files used to check a mirror against our official copies.

Hugging Face repos go to the dataset cache ($EOJEV_DATASET_HUB, as for the eval data), pinned to the revisions
checked on 2026-10-22. Direct files go to $EOJEV_DATASETS/raw/train/<source>/. The local snapshot paths are written
to $EOJEV_DATASETS/trainset/downloads.json. AID is already local (raw/aid/AID.zip).

Run as a CPU job (never on the login node):
    MODULE=trainset.download [ARGS="eurosat fmow ..."] sbatch -J eojev-dl slurm/convert_cpu.sbatch
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from bench.common import DATASETS

HUB = os.environ.get("EOJEV_DATASET_HUB", str(DATASETS / "hf_hub"))
RAW_TRAIN = DATASETS / "raw" / "train"
MANIFEST = DATASETS / "trainset" / "downloads.json"

HF = {  # name: (repo, revision, allow_patterns)
    "eurosat": ("timm/eurosat-rgb", "b4e28552cd5f3932b6abc37eb20d3e84901ad728",
                ["data/train-*", "data/validation-*", "README.md"]),
    "fmow": ("danielz01/fMoW", "8a27a7988a20341352b85bf0e65e88a6c1337f1c", ["WILDS/train-*.parquet", "README.md"]),
    "bigearthnet": ("timm/bigearthnet-v2-rgb", "edb997737d3888abe5257c71d8e4fabc3c8ec34d",
                    ["data/train-*.parquet", "README.md"]),
    # DOTA-v2.0 train; the val annotations only to compare the mirror with our official val labels
    "dota": ("torchgeo/dota", "672e63236622f7da6ee37fca44c50ac368b77cab",
             ["dotav1.0_images_train.tar.gz", "dotav2.0_images_train.tar.gz", "dotav2.0_annotations_train.tar.gz",
              "dotav2.0_annotations_val.tar.gz", "samples.csv", "README.md"]),
    "dior": ("torchgeo/dior", "6b3a7a1f1debd0bd27b94671410c026f2e202fcf",
             ["Images_trainval.zip", "Images_test.zip", "Annotations_trainval.zip", "sample_df.csv", "README.md"]),
    "xbd": ("kshitijrajsharma/xview2-xbd", "079adbab22aa09633dfd50ece0dd0137431a9780",
            ["data/train-*", "data/tier3-*", "README.md"]),
    "vhm_list": ("FitzPC/VHM_dataset_sft", "3881324794c42a8d03ae83f5bb1cd410f65bc6f7", ["list_sft.json", "README.md"]),
    "jev": ("SargeDev/jev-distill-corpus-v3", "75111a13a8455032f0435050bfc7e0c9026ecfe8",
            ["train.jsonl", "validation.jsonl", "README.md"]),
    "aokvqa": ("HuggingFaceM4/A-OKVQA", "d1b0efa3a436e9101dfbde3752db7607da696c35",
               ["data/train-*", "data/validation-*", "README.md"]),
}
URLS = {  # name: [(url, file name)]
    # DIOR's official labels and split lists (torchgeo has no test labels and does not separate val)
    "dior_official": [
        ("https://drive.usercontent.google.com/download?id=1KoQzqR20qvIXDf1qsXCHGxD003IPmXMw&export=download&confirm=t",
         "Annotations.zip"),
        ("https://drive.usercontent.google.com/download?id=1vOmzwxpBtwbK5o8xSa9u0IdB4H95MBHw&export=download&confirm=t",
         "ImageSets.zip"),
    ],
}
DEFAULT = list(HF) + list(URLS)


def fetch_hf(name: str) -> str:
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    repo, rev, patterns = HF[name]
    return snapshot_download(repo, repo_type="dataset", revision=rev, allow_patterns=patterns, cache_dir=HUB,
                             max_workers=8)


def fetch_urls(name: str) -> str:
    dest = RAW_TRAIN / name
    dest.mkdir(parents=True, exist_ok=True)
    for url, fname in URLS[name]:
        path = dest / fname
        if path.exists() and path.stat().st_size > 0:
            continue
        part = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:  # noqa: S310 - fixed https URLs
            while chunk := r.read(1 << 20):
                f.write(chunk)
        head = part.read_bytes()[:4]
        if head[:2] != b"PK":  # a Drive HTML page instead of the zip (quota or confirmation page)
            raise RuntimeError(f"{fname}: not a zip (starts with {head!r}); Google Drive refused the download")
        part.rename(path)
    return str(dest)


def main() -> None:
    names = sys.argv[1:] or DEFAULT
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    failed = []
    for name in names:
        t0 = time.time()
        try:
            path = fetch_hf(name) if name in HF else fetch_urls(name)
        except Exception as exc:  # noqa: BLE001 - report every source, then fail at the end
            print(f"FAILED {name}: {type(exc).__name__}: {exc}", flush=True)
            failed.append(name)
            continue
        manifest[name] = {"path": path, "revision": HF[name][1] if name in HF else None,
                          "fetched": time.strftime("%Y-%m-%d %H:%M")}
        MANIFEST.write_text(json.dumps(manifest, indent=1))
        print(f"ok {name}: {path} ({time.time() - t0:.0f}s)", flush=True)
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
