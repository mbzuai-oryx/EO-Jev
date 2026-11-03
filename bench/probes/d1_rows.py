"""Probe rows for diagnostic D1: can a linear readout find count, change-amount and
damage information in d3's decision state and in the vision features, where the zero-shot answers are weak?

- **d1_count_dior:**
  - Data: DIOR images and classes from the counting training rows (trainset fam2, DIOR only; eval images already
    removed).
  - Question: one fixed question, "How many <plural> are there in this image?", with 7 count bins: 0, 1, 2, 3 to 4,
    5 to 9, 10 to 19, 20 or more.
  - Split: train / test 75 / 25 by scene.
- **d1_cdvqa_ratio:**
  - Data: the CDVQA test-2 change-ratio questions, asked exactly as in the zero-shot run (11 bins).
  - Split: 70 / 30 by image pair.
- **d1_xbd_damage:**
  - Train: the xBD damage training crops (trainset fam4, train split, up to 1,000 per level).
  - Test: the xBD-damage benchmark's 3,000 buildings.
  - Question: both use the benchmark's question, so test measures transfer to unseen scenes.

Rows go to $EOJEV_DATASETS/converted/probes/d1/<set>.jsonl. probes/ is excluded from the leak check as non-eval
data. Image paths are absolute because the sets mix the trainset and converted roots. Features come from
bench/probes/extract.py and probes from bench/probes/fit.py (slurm/r0_gpu.sbatch, step d1).
"""

from __future__ import annotations

import collections
import hashlib
import json
import random
from pathlib import Path

from bench.common import CONVERTED, read_jsonl, write_jsonl
from bench.converters.xbd_damage import DAMAGE_Q, LEVELS
from trainset.common import ROWS, TRAIN
from trainset.fam2_9_objects import DIOR_NAMES

OUT = CONVERTED / "probes" / "d1"
COUNT_BINS = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3 to 4"), (5, 9, "5 to 9"), (10, 19, "10 to 19"),
              (20, 10**9, "20 or more")]
XBD_TRAIN_PER_LEVEL = 1000


def split_by(group: str, train_percent: int) -> str:
    return "train" if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100 < train_percent else "test"


def absolute(row: dict, root) -> list[str]:
    return [str(root / p) for p in row["images"]]


def count_rows() -> list[dict]:
    out, seen = [], set()
    for r in read_jsonl(ROWS / "final" / "fam2_counting.jsonl"):
        m = r["meta"]
        if m["source"].lower() != "dior" or (r["images"][0], m["class"]) in seen:
            continue
        seen.add((r["images"][0], m["class"]))
        label = next(text for lo, hi, text in COUNT_BINS if lo <= m["count"] <= hi)
        plural = DIOR_NAMES.get(m["class"], (m["class"], m["class"]))[1]
        out.append({"id": f"d1:count:{r['images'][0]}:{m['class']}", "group_id": r["group_id"], "benchmark": "d1_count_dior",
                    "split": "probe", "task_family": "count", "images": absolute(r, TRAIN), "state": "",
                    "questions": {"q": {"type": "choice", "instructions": f"How many {plural} are there in this image?",
                                        "criteria": {text: None for _, _, text in COUNT_BINS}}},
                    "answer": {"q": label},
                    "meta": {"count": m["count"], "class": m["class"], "probe_split": split_by(r["group_id"], 75)}})
    return out


def cdvqa_rows() -> list[dict]:
    out = []
    for r in read_jsonl(CONVERTED / "cdvqa" / "cdvqa_test2_5k.jsonl"):
        if "ratio" in r["task_family"]:
            out.append({**r, "images": absolute(r, CONVERTED),
                        "meta": {**r["meta"], "probe_split": split_by(r["group_id"], 70)}})
    return out


def xbd_rows() -> list[dict]:
    question = {"q": {"type": "score", "instructions": DAMAGE_Q, "criteria": LEVELS}}
    per_level = collections.defaultdict(list)
    for r in read_jsonl(ROWS / "final" / "fam4_damage.jsonl"):
        if r["task_family"] == "damage_ordinal" and r["split"] == "train":
            per_level[r["answer"]["q"]].append(r)
    out = []
    for level, rows in sorted(per_level.items()):
        random.Random(level).shuffle(rows)
        for r in rows[:XBD_TRAIN_PER_LEVEL]:
            out.append({**r, "images": absolute(r, TRAIN), "state": "", "questions": question,
                        "meta": {"level": level, "event": r["meta"]["event"], "probe_split": "train"}})
    for r in read_jsonl(CONVERTED / "xbd" / "xbd_damage_3k.jsonl"):
        out.append({**r, "images": absolute(r, CONVERTED), "meta": {"event": r["meta"]["event"], "probe_split": "test"}})
    return out


def main(out: Path = OUT) -> None:
    for name, rows in (("d1_count_dior", count_rows()), ("d1_cdvqa_ratio", cdvqa_rows()), ("d1_xbd_damage", xbd_rows())):
        write_jsonl(out / f"{name}.jsonl", rows)
        splits = collections.Counter(r["meta"]["probe_split"] for r in rows)
        labels = collections.Counter(str(r["answer"]["q"]) for r in rows)
        print(name, len(rows), json.dumps(splits), json.dumps(dict(sorted(labels.items()))), flush=True)


if __name__ == "__main__":
    main()
