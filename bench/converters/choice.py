"""C2 CHOICE (NeurIPS'25 D&B) -> rows: the 9,407 multiple-choice questions over 21 tasks (3-4 options).

Skipped: visual grounding (box output) and referring-expression segmentation (mask output), which d3 cannot
produce. Options are parsed from the "A.xxx" lines of each question; task_family is the leaf task, and the
level-1 / level-2 dimensions (perception | reasoning, ...) are kept in meta for the paper's hierarchy.

    python -m bench.converters.choice
"""

from __future__ import annotations

import json
import re
from collections import Counter

from bench.common import CONVERTED, RAW, store_image, validate_row, write_jsonl

SRC = RAW / "choice"
OPT = re.compile(r"^([A-Z])\.\s*(.*)$")
SKIP = {"visual_grounding", "referring_expression_segmentation"}


def parse(question: str) -> tuple[str, list[tuple[str, str]]]:
    lines = question.split("\n")
    first = next(i for i, l in enumerate(lines) if OPT.match(l.strip()))
    stem = "\n".join(lines[:first]).strip()
    opts = []
    for l in lines[first:]:
        m = OPT.match(l.strip())
        if m:
            opts.append((m.group(1), m.group(2).strip()))
        elif l.strip() and opts:  # an option spilling onto a second line
            opts[-1] = (opts[-1][0], opts[-1][1] + " " + l.strip())
    return stem, opts


def main() -> None:
    rows = []
    for f in sorted(SRC.glob("*/*/*/*.json")):
        level1, level2, task = f.parts[-4], f.parts[-3], f.parts[-2]
        if task in SKIP:
            continue
        for it in json.load(open(f)):
            stem, opts = parse(it["question"])
            texts = [t for _, t in opts]
            if len(set(texts)) != len(texts):
                texts = [f"{t} ({l})" for (l, _), t in zip(opts, texts)]
            letters = [l for l, _ in opts]
            gold = texts[letters.index(it["answer"].strip())]
            rel, size = store_image((SRC / it["image_path"]).read_bytes(),
                                    CONVERTED / "choice" / "images" / task / it["id"])
            row = {"id": f"CHOICE:{task}:{it['id']}", "group_id": f"CHOICE:{it['image_path']}", "benchmark": "CHOICE",
                   "split": "test", "task_family": task, "skills": [level2],
                   "images": [rel], "state": "",
                   "questions": {"q": {"type": "choice", "instructions": stem, "criteria": {t: None for t in texts}}},
                   "answer": {"q": gold},
                   "meta": {"level1": level1, "level2": level2, "orig_px": size, "n_options": len(texts)}}
            validate_row(row)
            rows.append(row)
    n = write_jsonl(CONVERTED / "choice" / "choice_mc.jsonl", rows)
    print("CHOICE:", n, "rows;", json.dumps(Counter(r["meta"]["level1"] for r in rows)))


if __name__ == "__main__":
    main()
