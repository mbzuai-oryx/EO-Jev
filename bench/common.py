"""Shared paths, row I/O and image handling for the eo-jev harness.

A row is one benchmark item: its images plus one or more typed questions about them (see docs/03_benchmark_plan.md §4.2).

    {"id", "group_id", "benchmark", "split", "task_family", "skills": [...],
     "images": [paths relative to $EOJEV_DATASETS/converted], "state": str | dict,
     "questions": {qid: {"type": "choice" | "noul" | "score", "instructions": str, "criteria": ...}},
     "answer": {qid: option key (choice) | bool (noul) | level index (score)},
     "meta": {...}}
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from PIL import Image

def _path(var: str) -> Path:
    """A path set by env.sh (per machine in env.local.sh)."""
    if var not in os.environ:
        raise RuntimeError(f"{var} is not set: `source env.sh` first (paths are configured in env.local.sh)")
    return Path(os.environ[var])


DATASETS = _path("EOJEV_DATASETS")
RAW = DATASETS / "raw"
CONVERTED = DATASETS / "converted"
RUNS = _path("EOJEV_RUNS")
HF_HUB_CACHE = _path("HF_HUB_CACHE")
THIRD_PARTY = _path("EOJEV_MODELS_ROOT") / "third_party"  # specialists' own inference code (download_specialists.sh)
SITE = _path("EOJEV_MODELS_ROOT") / "site"  # source of the results website (benchmarks.html, add_v1.py)
D3_FLASH = HF_HUB_CACHE / "models--vllm-sr--d3-flash/snapshots/521525effa79469d899a570726547dd7f00da3af"

# d3 reads images at 65,536 .. 1,638,400 pixels (decision_config / preprocessor_config).
MIN_PIXELS = 65_536
MAX_PIXELS = 1_638_400

REQUIRED_KEYS = ("id", "group_id", "benchmark", "split", "task_family", "images", "state", "questions", "answer")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    """Write rows; benchmark row files (dicts with "id" and "questions") must have unique ids."""
    rows = list(rows)
    ids = [r["id"] for r in rows if isinstance(r, dict) and "questions" in r]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: {len(ids) - len(set(ids))} duplicate row ids")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def validate_row(row: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in row]
    if missing:
        raise ValueError(f"{row.get('id')}: missing {missing}")
    for qid, q in row["questions"].items():
        if q["type"] not in ("choice", "noul", "score"):
            raise ValueError(f"{row['id']}/{qid}: bad type {q['type']}")
        if qid not in row["answer"]:
            raise ValueError(f"{row['id']}/{qid}: no answer")
        if q["type"] == "choice" and row["answer"][qid] not in q["criteria"]:
            raise ValueError(f"{row['id']}/{qid}: answer {row['answer'][qid]!r} not among the options")


SAR_BENCHMARKS = {"EarthMind-Bench-sar", "EarthMind-Bench-fused", "BigEarthNet.txt-s2s1"}  # every row has a SAR image
SAR_ANSWERS = {"SAR", "SAR image"}  # CHOICE image_modality / VHM image_type rows whose image is SAR


def is_sar(row: dict) -> bool:
    """True when a row's images include SAR. SAR is left out of training and scoring (decided 2026-10-29: SAR may get
    its own decision model). Rows: EarthMind SAR and optical + SAR, BigEarthNet.txt S2 + S1, DisasterM3 items with a
    SAR post-event image, CHOICE / VHM modality questions whose answer is SAR, and GeoMMBench questions about SAR
    imaging (by their text, so approximate)."""
    if row["benchmark"] in SAR_BENCHMARKS or (row.get("meta") or {}).get("post_modality") == "SAR":
        return True
    if row["task_family"] in ("image_modality", "image_type"):
        return bool(SAR_ANSWERS & {str(a) for a in row["answer"].values()})
    if row["benchmark"].startswith("GeoMMBench"):
        return re.search(r"\bSAR\b", json.dumps(row["questions"])) is not None
    return False


def save_png(image: Image.Image, path: Path) -> Path:
    """Write an 8-bit RGB PNG (d3 accepts PNG / JPEG / WebP only), creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(path, format="PNG")
    return path


NATIVE_FORMATS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}


def store_image(data: bytes, dest_stem: Path) -> tuple[str, list[int]]:
    """Keep PNG / JPEG / WebP bytes as they are (RGB or L); re-encode anything else to PNG.
    Returns (path relative to CONVERTED, [width, height])."""
    import io

    image = Image.open(io.BytesIO(data))
    size = list(image.size)
    ext = NATIVE_FORMATS.get(image.format or "")
    if ext and image.mode in ("RGB", "L"):
        path = dest_stem.with_suffix(ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    else:
        path = save_png(image, dest_stem.with_suffix(".png"))
    return str(path.relative_to(CONVERTED)), size


def transform_image(image: Image.Image, name: str | None) -> Image.Image:
    """Robustness / ablation variants applied at run time (docs/03 §7, §8.2)."""
    if not name or name == "none":
        return image
    if name in ("rot90", "rot180", "rot270"):
        return image.rotate(-int(name[3:]), expand=True)  # clockwise
    if name == "hflip":
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if name == "vflip":
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if name.startswith("down"):  # down2 / down4: shrink by that factor (resolution ablation A1)
        factor = int(name[4:])
        return image.resize((max(1, image.width // factor), max(1, image.height // factor)), Image.Resampling.BICUBIC)
    raise ValueError(f"unknown image transform {name!r}")
