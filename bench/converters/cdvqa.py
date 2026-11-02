"""C8 CDVQA test-2 (31,036 questions on SECOND bi-temporal pairs, 512 px, 0.15 m) -> rows (seeded 5k sample).

Source: the webdataset mirror ljx620/CDVQA (each sample = question + its image pair; official answer space).
  change_or_not / increase_or_not / decrease_or_not          -> `noul`
  largest_change / smallest_change / change_to_what          -> 6-class `choice`
  change_ratio / change_ratio_types (11 bins: 0, 0-10%, ...) -> 11-option `choice` (bin order kept in meta for
                                                                within-one-bin accuracy; `score` allows <= 10 levels)
Sampling is stratified by question type, proportional to the test-2 distribution.

    python -m bench.converters.cdvqa
"""

from __future__ import annotations

import json
import random
import re
import tarfile
from collections import Counter, defaultdict

from bench.common import CONVERTED, RAW, store_image, validate_row, write_jsonl

SRC = RAW / "cdvqa-wds" / "test2"
OUT = CONVERTED / "cdvqa"
N = 5000
CLASSES = {"buildings": "buildings", "low_vegetation": "low vegetation", "trees": "trees",
           "NVG_surface": "non-vegetated ground surface", "water": "water", "playgrounds": "playgrounds"}
BINS = ["0", "0_to_10", "10_to_20", "20_to_30", "30_to_40", "40_to_50", "50_to_60", "60_to_70", "70_to_80",
        "80_to_90", "90_to_100"]
BIN_TEXT = {b: ("0%" if b == "0" else b.replace("_to_", " to ") + "%") for b in BINS}
STATE = "Image 1 is the earlier (pre-change) image and Image 2 the later (post-change) image of the same area."


def main() -> None:
    samples = []  # (tar path, key, meta, question, answer)
    for tar_path in sorted(SRC.glob("test2-*.tar")):
        with tarfile.open(tar_path) as tf:
            for m in tf.getmembers():
                if m.name.endswith(".json"):
                    rec = json.loads(tf.extractfile(m).read())
                    q = rec["conversations"][0]["value"]
                    q = re.sub(r"^(Image \d: <image>\n)+", "", q).strip()
                    samples.append((tar_path, m.name[: -len(".json")], rec["meta"], q, rec["conversations"][1]["value"]))
    by_type = defaultdict(list)
    for s in samples:
        by_type[s[2]["question_type"]].append(s)
    rng = random.Random(0)
    picked = []
    for qtype, ss in sorted(by_type.items()):
        picked += rng.sample(ss, min(len(ss), round(N * len(ss) / len(samples))))

    by_tar = defaultdict(list)
    for s in picked:
        by_tar[s[0]].append(s)
    rows = []
    for tar_path, ss in by_tar.items():
        with tarfile.open(tar_path) as tf:
            for _, key, meta, question, answer in ss:
                images = [store_image(tf.extractfile(f"{key}.{i}.img").read(),
                                      OUT / "images" / f"{meta['image_id'].rsplit('.', 1)[0]}_{i}")[0] for i in (0, 1)]
                qtype = meta["question_type"]
                if qtype in ("change_or_not", "increase_or_not", "decrease_or_not"):
                    q, gold = {"type": "noul", "instructions": question}, answer == "yes"
                elif qtype in ("largest_change", "smallest_change", "change_to_what"):
                    opts = list(CLASSES.values())
                    random.Random(f"cdvqa:{key}").shuffle(opts)
                    q, gold = {"type": "choice", "instructions": question, "criteria": {o: None for o in opts}}, CLASSES[answer]
                else:
                    q = {"type": "choice", "instructions": question, "criteria": {BIN_TEXT[b]: None for b in BINS}}
                    gold = BIN_TEXT[answer]
                row = {"id": f"CDVQA:{meta['question_id']}", "group_id": f"CDVQA:{meta['image_id']}",
                       "benchmark": "CDVQA", "split": "test2", "task_family": qtype, "skills": ["change"],
                       "images": images, "state": STATE, "questions": {"q": q}, "answer": {"q": gold},
                       "meta": {"raw_answer": answer, "gsd_m": 0.15,
                                **({"ordinal_levels": [BIN_TEXT[b] for b in BINS]} if "ratio" in qtype else {})}}
                validate_row(row)
                rows.append(row)
    n = write_jsonl(OUT / "cdvqa_test2_5k.jsonl", rows)
    print("CDVQA test-2:", n, "rows of", len(samples), json.dumps(Counter(r["task_family"] for r in rows)))


if __name__ == "__main__":
    main()
