"""C27 TerraScope-Bench (CVPR'26; arXiv 2603.19039) multiple-choice questions -> rows. AS-PUBLISHED.

Source: HF sy1998/TerraScope-Bench @ c00191b737bc01044e3de93fb4f18c5f33b83564 (Apache-2.0):
`data/TerraScope_bench.parquet`, 3,549 rows with one embedded 256 x 256 RGB PNG each (2,006 file names; Sentinel-2
optical patches at 10 m), the question, `candidates` (JSON, options keyed A..D in published order), the answer
letter, `task_type` and the generating `ground_truth` (JSON). Tasks: boundary_detection 855 (2 options),
comparative_ranking 855 (2), coverage_percentage 855 (4), absolute_area 855 (4), distance_measurement 129 (4).
The paper's sixth task, Building Change Estimation (288 multi-temporal items; 3,837 in the paper), is not in the
HF release, so the paper's "overall" (macro over six tasks) cannot be reproduced: cite per task and the 5-task mean.

As published (cite TerraScope 72.2 (5-task) / EarthMind-4B 46.4 directly):
  - all 3,549 items in parquet order; images are the published PNG bytes (256 x 256 = 65,536 px = MIN_PIXELS, so
    none is resized), stored once per published file name;
  - `instructions` = the published question verbatim; one `choice` over the candidate texts in published key order
    (A, B[, C, D]), texts verbatim (no letter prefixes in the source); task_family = task_type.
Not identical (recorded in meta): the paper's "unified prompt template" is not released (only its guidance sentence
"Please respond with only the option letter (A, B, C, or D) corresponding to your answer." is quoted in App. F.2);
our runners append their own lettered options and "Answer with the option's letter from the given choices.". The
paper scores free text with the regex \\b[A-D]\\b (first match); we score the chosen option. absolute_area and
distance_measurement questions do not state the 10 m pixel size (as published).

    python -m bench.converters.terrascope_bench
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from bench.common import CONVERTED, DATASETS, MAX_PIXELS, MIN_PIXELS, store_image, validate_row, write_jsonl
from bench.converters.perception_utils import sample_text, summary

REVISION = "sy1998/TerraScope-Bench@c00191b737bc01044e3de93fb4f18c5f33b83564"
SRC = (DATASETS / "hf_hub" / "datasets--sy1998--TerraScope-Bench" / "snapshots"
       / "c00191b737bc01044e3de93fb4f18c5f33b83564" / "data" / "TerraScope_bench.parquet")
OUT = CONVERTED / "terrascope_bench"
GUIDANCE = "Please respond with only the option letter (A, B, C, or D) corresponding to your answer."


def main() -> None:
    table = pq.read_table(SRC).to_pylist()
    rows, stored, digests = [], {}, {}
    for k, rec in enumerate(table):
        name = rec["image"]["path"]
        data = rec["image"]["bytes"]
        digest = hashlib.sha1(data).hexdigest()
        if name in digests and digests[name] != digest:
            raise ValueError(f"{name}: two different images under one file name")
        digests[name] = digest
        if name not in stored:
            stored[name] = store_image(data, OUT / "images" / Path(name).stem)
        rel, size = stored[name]
        if not MIN_PIXELS <= size[0] * size[1] <= MAX_PIXELS:
            raise ValueError(f"{name}: {size} outside the pixel limits; images are kept as published")
        cand = json.loads(rec["candidates"])
        keys = list(cand)
        if keys != list("ABCD"[: len(keys)]):
            raise ValueError(f"row {k}: candidate keys {keys}")
        options = [str(cand[x]).strip() for x in keys]
        if len(set(options)) != len(options):
            raise ValueError(f"row {k}: duplicate options {options}")
        row = {
            "id": f"TerraScope-Bench:{k}",
            "group_id": f"TerraScope-Bench:{name}",
            "benchmark": "TerraScope-Bench",
            "split": "test",
            "task_family": rec["task_type"],
            "skills": ["quantitative", "lulc"],
            "images": [rel],
            "state": "",
            "questions": {"q": {"type": "choice", "instructions": rec["question"].strip(),
                                "criteria": {o: None for o in options}}},
            "answer": {"q": str(cand[rec["answer"].strip()]).strip()},
            "meta": {"source": REVISION, "source_row": k, "image": name, "orig_px": size, "resize_factor": 1.0,
                     "published_candidates": cand, "published_answer_letter": rec["answer"].strip(),
                     "ground_truth": json.loads(rec["ground_truth"]), "as_published": True,
                     "published_answer_guidance": GUIDANCE, "sensor": "Sentinel-2 RGB", "gsd_m": 10.0},
        }
        validate_row(row)
        rows.append(row)
    write_jsonl(OUT / "terrascope_bench.jsonl", rows)
    print(summary("TerraScope-Bench", rows))
    print("  options per task:", dict(Counter((r["task_family"], len(r["questions"]["q"]["criteria"])) for r in rows)))
    print(f"  images: {len(stored)} file names ({len(set(digests.values()))} distinct contents); resized: 0; dropped: 0")
    print(sample_text(rows, 3, seed=1))


if __name__ == "__main__":
    main()
