"""C5 LHRS-Bench (690 single-choice questions on 108 images) -> rows, with CircularEval.

Each question is asked once per cyclic rotation of its options (2-4 rotations); `group_id` is the question, so
"strict" accuracy (all rotations correct, the paper's strict protocol) and the plain first-order accuracy can
both be computed. Option keys are the option texts ("A. Factory; B. Park" -> "Factory", "Park").

    python -m bench.converters.lhrs_bench
"""

from __future__ import annotations

import json
import re

from bench.common import CONVERTED, RAW, validate_row, write_jsonl

SRC = RAW / "lhrs-bench" / "LHRS-Bench"
QTYPES = {"1": "identity", "2": "color", "3": "orientation", "4": "shape", "5": "quantity", "6": "area",
          "7": "distance", "8": "resolution", "9": "modality", "10": "location", "11": "reasoning"}
OPTION = re.compile(r"\s*([A-Z])\.\s*(.*?)\s*$")


def parse_choices(text: str) -> list[tuple[str, str]]:
    parts = [p for p in text.strip().rstrip(".").split(";") if p.strip()]
    out = []
    for p in parts:
        m = OPTION.match(p)
        if not m:
            raise ValueError(f"cannot parse option {p!r} in {text!r}")
        out.append((m.group(1), m.group(2).rstrip(".")))
    return out


def main() -> None:
    qa = json.load(open(SRC / "qa.json"))
    rows = []
    for entry in qa["data"]:
        image_rel = str((SRC / "imgs_png" / entry["filename"]).relative_to(RAW.parent))
        # images live under raw/; express the path relative to converted/ for the runner
        image_rel = "../raw/" + image_rel.split("raw/", 1)[1]
        for k, pair in enumerate(entry["qa_pairs"]):
            options = parse_choices(pair["choices"])
            letters = [l for l, _ in options]
            texts = [t for _, t in options]
            if len(set(texts)) != len(texts):
                raise ValueError(f"duplicate options in {entry['filename']} #{k}")
            gold = texts[letters.index(pair["answer"].strip().rstrip("."))]
            qid = f"{entry['filename'].split('.')[0]}#{k}"
            for shift in range(len(texts)):
                rotated = texts[shift:] + texts[:shift]
                row = {
                    "id": f"LHRS-Bench:{qid}:o{shift}",
                    "group_id": f"LHRS-Bench:{qid}",
                    "benchmark": "LHRS-Bench",
                    "split": "test",
                    "task_family": "vqa",
                    "skills": [QTYPES[t] for t in pair["type"]],
                    "images": [image_rel],
                    "state": "",
                    "questions": {"q": {"type": "choice", "instructions": pair["question"],
                                        "criteria": {t: None for t in rotated}}},
                    "answer": {"q": gold},
                    "meta": {"rotation": shift, "n_options": len(texts), "original_order": texts},
                }
                validate_row(row)
                rows.append(row)
    n = write_jsonl(CONVERTED / "lhrs_bench" / "lhrs_bench_circular.jsonl", rows)
    print("LHRS-Bench:", n, "rows from", qa["qa_num"], "questions")


if __name__ == "__main__":
    main()
