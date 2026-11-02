"""C4 RSVQA -> rows, one per active test question (group_id = image).

LR (Sentinel-2 RGB, 256 px; 10,004 active test questions on 100 images): presence and comparison -> `noul`,
rural/urban -> `choice`, count -> `score` over the paper's five bins (0 / 1-10 / 11-100 / 101-1000 / >1000).
HR test-2 (Philadelphia, 15 cm aerial, 512 px): presence and comparison -> `noul` (GeoChat protocol), a seeded
5,000-question subsample; images are extracted from Images.tar on first use.

    python -m bench.converters.rsvqa lr
    python -m bench.converters.rsvqa hr
"""

from __future__ import annotations

import json
import random
import sys
import tarfile
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image

from bench.common import CONVERTED, RAW, save_png, validate_row, write_jsonl

LR_COUNT_BINS = ["0", "between 1 and 10", "between 11 and 100", "between 101 and 1000", "more than 1000"]
FAMILY = {"presence": "presence", "comp": "comparison", "rural_urban": "rural_urban", "count": "count"}
HR_SAMPLE = 5000


def lr_count_bin(n: int) -> int:
    return 0 if n == 0 else 1 if n <= 10 else 2 if n <= 100 else 3 if n <= 1000 else 4


def load(src: Path, prefix: str) -> tuple[list[dict], dict[int, str]]:
    questions = json.load(open(src / f"{prefix}_questions.json"))["questions"]
    answers = json.load(open(src / f"{prefix}_answers.json"))["answers"]
    active = [q for q in questions if q.get("active")]
    gold = {a["question_id"]: a["answer"] for a in answers if a.get("active")}
    return [q for q in active if q["id"] in gold], gold


def make_row(bench: str, q: dict, answer: str, image_rel: str, gsd: float) -> dict | None:
    qtype = q["type"]
    text = q["question"]
    if qtype in ("presence", "comp"):
        question = {"type": "noul", "instructions": text}
        gold = answer == "yes"
    elif qtype == "rural_urban":
        question = {"type": "choice", "instructions": text, "criteria": {"rural": None, "urban": None}}
        gold = answer
    elif qtype == "count" and bench == "RSVQA-LR":
        question = {"type": "score", "instructions": text, "criteria": LR_COUNT_BINS}
        gold = lr_count_bin(int(answer))
    else:
        return None
    row = {
        "id": f"{bench}:{q['id']}",
        "group_id": f"{bench}:img{q['img_id']}",
        "benchmark": bench,
        "split": "test" if bench == "RSVQA-LR" else "test-2 (Philadelphia)",
        "task_family": FAMILY[qtype],
        "skills": [FAMILY[qtype]],
        "images": [image_rel],
        "state": "",
        "questions": {"q": question},
        "answer": {"q": gold},
        "meta": {"raw_answer": answer, "gsd_m": gsd, "img_id": q["img_id"]},
    }
    validate_row(row)
    return row


def lr() -> None:
    src = RAW / "rsvqa-lr"
    questions, gold = load(src, "LR_split_test")
    out_dir = CONVERTED / "rsvqa" / "lr_images"
    images: dict[int, str] = {}
    with zipfile.ZipFile(src / "Images_LR.zip") as zf:
        for img_id in sorted({q["img_id"] for q in questions}):
            image = Image.open(BytesIO(zf.read(f"Images_LR/{img_id}.tif")))
            images[img_id] = str(save_png(image, out_dir / f"{img_id}.png").relative_to(CONVERTED))
    rows = [r for q in questions if (r := make_row("RSVQA-LR", q, gold[q["id"]], images[q["img_id"]], 10.0))]
    n = write_jsonl(CONVERTED / "rsvqa" / "rsvqa_lr.jsonl", rows)
    print("RSVQA-LR:", n, "rows", Counter(r["task_family"] for r in rows))


def hr() -> None:
    src = RAW / "rsvqa-hr"
    questions, gold = load(src, "USGS_split_test_phili")
    questions = [q for q in questions if q["type"] in ("presence", "comp")]
    random.Random(0).shuffle(questions)
    questions = questions[:HR_SAMPLE]
    need = {q["img_id"] for q in questions}
    out_dir = CONVERTED / "rsvqa" / "hr_images"
    images: dict[int, str] = {}
    with tarfile.open(src / "Images.tar") as tf:
        for member in tf:
            stem = Path(member.name).stem
            if member.isfile() and stem.isdigit() and int(stem) in need:
                image = Image.open(BytesIO(tf.extractfile(member).read()))
                images[int(stem)] = str(save_png(image, out_dir / f"{stem}.png").relative_to(CONVERTED))
    missing = need - set(images)
    if missing:
        raise FileNotFoundError(f"{len(missing)} HR images missing from Images.tar, e.g. {sorted(missing)[:5]}")
    rows = [r for q in questions if (r := make_row("RSVQA-HR", q, gold[q["id"]], images[q["img_id"]], 0.15))]
    n = write_jsonl(CONVERTED / "rsvqa" / "rsvqa_hr_test2_5k.jsonl", rows)
    print("RSVQA-HR test-2:", n, "rows", Counter(r["task_family"] for r in rows), "on", len(need), "images")


if __name__ == "__main__":
    {"lr": lr, "hr": hr}[sys.argv[1]]()
