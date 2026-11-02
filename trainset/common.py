"""Shared helpers for training rows.

Training rows follow the benchmark row schema (bench/common.py) so the trainer renders prompts exactly as the runners
do. Images are stored under $EOJEV_DATASETS/trainset/images, and row image paths are relative to $EOJEV_DATASETS/
trainset, never to converted/: the eval index walks converted/, and training data must never be mistaken for eval data.

Rules every family follows (format rules):
- Answers are computed from labels. Wording comes from templates; no LLM.
- The gold option's position is uniform, because the option order is shuffled per row.
- The number of options K is drawn from 2 up to the full list, so every option code seen at test time is trained.
- In about 18% of choice rows the gold is replaced by a "none of these" option, which is then the answer. In another
  ~10% "none of these" is present but wrong, so its presence alone never gives the answer away.
- Wrong options mix confusable classes (same group) with random ones.
- Every row records its source, licence, template id and image sha256; group_id is the source scene, which also
  decides the train/dev split.
"""

from __future__ import annotations

import hashlib
import io
import random
from pathlib import Path

from PIL import Image

from bench.common import DATASETS, validate_row

TRAIN = DATASETS / "trainset"
ROWS = TRAIN / "rows"
IMAGES = TRAIN / "images"
NATIVE = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
NONE_WORDINGS = ["none of these", "none of the above", "none of the listed options", "not one of these"]
NONE_RATE = 0.18          # gold replaced by "none of these"
NONE_DISTRACTOR_RATE = 0.10  # "none of these" offered but wrong
DEV_PERCENT = 2


def split_of(group_id: str) -> str:
    """Deterministic train/dev split by source scene: about 2% dev."""
    h = int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16)
    return "dev" if h % 100 < DEV_PERCENT else "train"


def store(data: bytes, rel_stem: str) -> tuple[str, list[int], str]:
    """Keep PNG / JPEG / WebP bytes as they are (as the eval converters do); re-encode anything else to PNG.
    Returns (path relative to TRAIN, [w, h], sha256 of the stored bytes)."""
    image = Image.open(io.BytesIO(data))
    size = list(image.size)
    ext = NATIVE.get(image.format or "")
    if not (ext and image.mode in ("RGB", "L")):
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        data, ext = buf.getvalue(), ".png"
    path = IMAGES / f"{rel_stem}{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size != len(data):
        path.write_bytes(data)
    return str(path.relative_to(TRAIN)), size, hashlib.sha256(data).hexdigest()


def sample_k(rng: random.Random, n_classes: int) -> int:
    """Number of options: the full list a quarter of the time, otherwise a size from 2 up to the full list."""
    if n_classes <= 2 or rng.random() < 0.25:
        return n_classes
    sizes = [k for k in (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50) if k < n_classes]
    return rng.choice(sizes)


def choice_options(rng: random.Random, gold: str, classes: list[str], groups: dict[str, str] | None = None,
                   k: int | None = None, overlaps: dict[str, set[str]] | None = None) -> tuple[list[str], str, dict]:
    """Options for one choice question, shuffled. Returns (options, answer, meta).

    overlaps: classes that are partly true of an image of `gold` (an airport image also shows a runway). They are
    never offered when "none of these" is the answer, so that answer stays unambiguous."""
    k = k or sample_k(rng, len(classes))
    r = rng.random()
    none_answer = r < NONE_RATE and k >= 2
    banned = (overlaps or {}).get(gold, set()) if none_answer else set()
    others = [c for c in classes if c != gold and c not in banned]
    k = min(k, len(others) + 1)
    same = [c for c in others if groups and groups.get(c) == groups.get(gold)]
    rng.shuffle(same)
    n_conf = min(len(same), (k - 1) // 2 + (k - 1) % 2)  # about half the wrong options are confusable
    picked = same[:n_conf]
    rest = [c for c in others if c not in picked]
    picked += rng.sample(rest, min(len(rest), k - 1 - len(picked)))
    meta = {"k": k, "n_confusable": n_conf, "none": None}
    if none_answer:
        none = rng.choice(NONE_WORDINGS)
        options, answer = picked + [none], none
        meta["none"] = "answer"
    elif r < NONE_RATE + NONE_DISTRACTOR_RATE and k >= 3:
        none = rng.choice(NONE_WORDINGS)
        options, answer = [gold] + picked[:-1] + [none], gold
        meta["none"] = "distractor"
    else:
        options, answer = [gold] + picked, gold
    rng.shuffle(options)
    return options, answer, meta


def make_row(*, rid: str, group_id: str, family: str, source: str, task_family: str, images: list[str], state: str,
             questions: dict, answer: dict, meta: dict) -> dict:
    row = {"id": rid, "group_id": group_id, "benchmark": f"trainset/{family}/{source}", "split": split_of(group_id),
           "task_family": task_family, "images": images, "state": state, "questions": questions, "answer": answer,
           "meta": {"family": family, "source": source, **meta}}
    validate_row(row)
    return row


def render(row: dict) -> str:
    """The row as the model sees it (options lettered in order), for review samples."""
    letters = [chr(ord("A") + i) for i in range(26)]
    out = [f"[{row['id']}]  split={row['split']}  images={row['images']}"]
    if row["state"]:
        out.append(f"  state: {row['state']}")
    for qid, q in row["questions"].items():
        out.append(f"  Q ({q['type']}): {q['instructions']}")
        if q["type"] in ("choice", "score"):
            opts = list(q["criteria"]) if isinstance(q["criteria"], dict) else list(q["criteria"])
            for i, o in enumerate(opts):
                tag = letters[i] if i < 26 else f"#{i + 1}"
                desc = q["criteria"].get(o) if isinstance(q["criteria"], dict) else None
                out.append(f"     {tag}. {o}" + (f"  ({desc})" if desc else ""))
        out.append(f"  answer: {row['answer'][qid]}")
    return "\n".join(out)
