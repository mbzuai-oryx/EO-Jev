"""C3 VRSBench VQA (test: 37,409 open short answers, GPT-4-judged in the paper) -> closed-form rows.

Conversion, per answer:
  - "yes" / "no" (any question type)            -> `noul`
  - object quantity (integers)                  -> `score`, 8 levels: 0, 1, 2, 3, 4, 5, 6 to 10, more than 10
  - color / category / shape / position / direction / scene type / rural-urban
                                                 -> `choice` over that type's closed vocabulary: its most frequent
                                                    normalized answers (>= 15 occurrences, at most 16), synonyms merged
  - anything else (free text, rare answers)     -> dropped
A seeded, type-stratified sample of 5,000 questions is written. Scores are NOT comparable to the paper's
GPT-judged accuracy; the variant is reported as "VRSBench-VQA-MC".

    python -m bench.converters.vrsbench
"""

from __future__ import annotations

import json
import random
import re
import zipfile
from collections import Counter, defaultdict

from bench.common import CONVERTED, RAW, validate_row, write_jsonl

SRC = RAW / "vrsbench"
OUT = CONVERTED / "vrsbench"
N_TOTAL = 5000
QUOTA = {  # target share of the 5,000 per question kind
    "yes/no": 1500, "object quantity": 1000, "object color": 500, "object category": 500, "object position": 500,
    "scene type": 400, "object shape": 300, "object direction": 150, "rural or urban": 150,
}
COUNT_LEVELS = ["0", "1", "2", "3", "4", "5", "6 to 10", "more than 10"]
SYNONYMS = {
    "grey": "gray", "dark gray": "gray", "light gray": "gray",
    "middle-right": "right", "right-middle": "right", "middle-left": "left", "left-middle": "left",
    "top-middle": "top", "middle-top": "top", "bottom-middle": "bottom", "middle-bottom": "bottom",
    "middle": "center", "centre": "center", "central": "center", "middle-center": "center",
    "horizontal": "horizontal (east-west)", "horizontally": "horizontal (east-west)", "east-west": "horizontal (east-west)",
    "vertical": "vertical (north-south)", "vertically": "vertical (north-south)", "north-south": "vertical (north-south)",
    "diagonally": "diagonal", "round": "circular", "rectangle": "rectangular", "square-shaped": "square",
    "diamond-shaped": "diamond", "tennis courts": "tennis court", "airplane": "plane", "airplanes": "plane",
    "ships": "ship", "small vehicle": "small-vehicle", "small vehicles": "small-vehicle",
    "right side": "right", "left side": "left", "top side": "top", "bottom side": "bottom",
    "bottom right": "bottom-right", "bottom left": "bottom-left", "top right": "top-right", "top left": "top-left",
    "upper-left": "top-left", "upper-right": "top-right", "lower-left": "bottom-left", "lower-right": "bottom-right",
    "greenish": "green", "sandy": "beige", "tan": "beige", "reddish-brown": "brown",
    "agricultural fields": "agricultural", "farmland": "agricultural", "tennis courts": "tennis court",
}
# Fixed, non-overlapping vocabularies where the data-driven one would mix in non-answers ("edge", "vehicle", "dark").
FIXED_VOCAB = {
    "object position": ["top-left", "top", "top-right", "left", "center", "right", "bottom-left", "bottom", "bottom-right"],
    "object color": ["green", "white", "gray", "red", "blue", "brown", "yellow", "black", "orange", "beige"],
}


def normalize(answer: str) -> str:
    a = answer.strip().lower().rstrip(".")
    a = re.sub(r"\s+", " ", a)
    return SYNONYMS.get(a, a)


def count_level(answer: str) -> int | None:
    if not answer.isdigit():
        return None
    n = int(answer)
    return n if n <= 5 else 6 if n <= 10 else 7


def main() -> None:
    items = json.load(open(SRC / "VRSBench_EVAL_vqa.json"))
    # Closed vocabularies per choice type.
    vocab: dict[str, list[str]] = {}
    for qtype in [k for k in QUOTA if k not in ("yes/no", "object quantity")]:
        counts = Counter(normalize(x["ground_truth"]) for x in items if x["type"] == qtype)
        counts.pop("yes", None), counts.pop("no", None)
        vocab[qtype] = FIXED_VOCAB.get(qtype) or [a for a, c in counts.most_common(16) if c >= 15]

    pools: dict[str, list[tuple[dict, dict, object]]] = defaultdict(list)
    for x in items:
        gt = normalize(x["ground_truth"])
        if gt in ("yes", "no"):
            pools["yes/no"].append((x, {"type": "noul", "instructions": x["question"]}, gt == "yes"))
        elif x["type"] == "object quantity":
            level = count_level(gt)
            if level is not None:
                pools["object quantity"].append(
                    (x, {"type": "score", "instructions": x["question"], "criteria": COUNT_LEVELS}, level))
        elif x["type"] in vocab and gt in vocab[x["type"]]:
            pools[x["type"]].append((x, None, gt))

    rng = random.Random(0)
    picked = []
    for kind, quota in QUOTA.items():
        pool = pools[kind]
        picked += [(kind, p) for p in rng.sample(pool, min(quota, len(pool)))]

    rows, need = [], set()
    for kind, (x, question, gold) in picked:
        if question is None:  # closed-vocabulary choice; shuffle the vocabulary per item
            options = vocab[kind][:]
            random.Random(f"vrs:{x['question_id']}:{x['image_id']}").shuffle(options)
            question = {"type": "choice", "instructions": x["question"], "criteria": {o: None for o in options}}
        need.add(x["image_id"])
        row = {
            "id": f"VRSBench-VQA:{x['image_id']}:{x['question_id']}:{x['type']}",
            "group_id": f"VRSBench:{x['image_id']}",
            "benchmark": "VRSBench-VQA-MC",
            "split": "test",
            "task_family": "vqa_" + kind.replace(" ", "_").replace("/", ""),
            "skills": [x["type"]],
            "images": [f"vrsbench/images/{x['image_id']}"],
            "state": "",
            "questions": {"q": question},
            "answer": {"q": gold},
            "meta": {"vrs_type": x["type"], "raw_answer": x["ground_truth"], "kind": kind},
        }
        validate_row(row)
        rows.append(row)

    with zipfile.ZipFile(SRC / "Images_val.zip") as zf:
        for name in sorted(need):
            dest = OUT / "images" / name
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(f"Images_val/{name}"))
    n = write_jsonl(OUT / "vrsbench_vqa_mc_5k.jsonl", rows)
    print("VRSBench-VQA-MC:", n, "rows on", len(need), "images", Counter(r["meta"]["kind"] for r in rows))
    print("vocabularies:", json.dumps(vocab))


if __name__ == "__main__":
    main()
