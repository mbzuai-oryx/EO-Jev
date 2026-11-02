"""C26 GeoMMBench (CVPR'26 Highlight; expert multi-sensor geoscience knowledge) four-option MCQ -> rows. ADAPTED.

Source: HF AR-X/GeoMMBench @ 298d660bbbb37fd528a0122af6375e4aa0d7bdd3 (no licence declared on the card):
`data/test.parquet` (1,016 rows, index 37..1052) and `data/validation.parquet` (37 rows, index 0..36); columns
question, A, B, C, D, answer (letter), index, image (PNG bytes, one image per question; many are charts or diagrams,
and several questions share one image).

Adapted (docs/06 §8: we run VHM-7B ourselves on this version):
  - options SHUFFLED per item with `random.Random(row_id)` (row_id = "GeoMMBench:<split>:<index>"): in the published
    order always answering "B" scores 40.3% on test (A 253, B 409, C 230, D 124). Option texts are kept verbatim
    (none carries a letter prefix; texts like "Image A", "Method B" or "(a)" are content and stay); "None of the
    above"-type options (28 test items) are shuffled with the rest, which keeps their meaning. meta keeps the
    published option order, the published answer letter and the permutation;
  - images are stored once per distinct content (sha1): 619 test + 37 val files. 14 test questions (9 distinct
    images) above MAX_PIXELS are downscaled (Lanczos) and 18 test questions (14 distinct images) + 2 val images below
    MIN_PIXELS are upscaled (bicubic) into d3's pixel range, re-encoded as PNG; the factor and original size are in
    meta; all other images keep their published PNG bytes.
  - instructions = the published question text; the runners add the lettered options.
Val (37) goes to a separate file.

    python -m bench.converters.geommbench
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter

import pyarrow.parquet as pq

from bench.common import CONVERTED, DATASETS, validate_row, write_jsonl
from bench.converters.perception_utils import sample_text, store_fitted, summary

SNAP = DATASETS / "hf_hub" / "datasets--AR-X--GeoMMBench" / "snapshots" / "298d660bbbb37fd528a0122af6375e4aa0d7bdd3"
OUT = CONVERTED / "geommbench"
FILES = {"test": "test.parquet", "val": "validation.parquet"}


def convert(split: str) -> tuple[list[dict], Counter]:
    table = pq.read_table(SNAP / "data" / FILES[split]).to_pylist()
    rows, stats = [], Counter()
    seen = {}
    for rec in table:
        row_id = f"GeoMMBench:{split}:{rec['index']}"
        published = [str(rec[k]).strip() for k in "ABCD"]
        if len(set(published)) != 4:
            raise ValueError(f"{row_id}: duplicate options {published}")
        gold = published["ABCD".index(rec["answer"].strip())]
        perm = list(range(4))
        random.Random(row_id).shuffle(perm)
        options = [published[i] for i in perm]
        data = rec["image"]["bytes"]
        digest = hashlib.sha1(data).hexdigest()[:16]
        if digest not in seen:
            seen[digest] = store_fitted(data, OUT / "images" / digest)
            info = seen[digest]
            stats["images"] += 1
            if info["resize_factor"] < 1:
                stats["images_downscaled"] += 1
            elif info["resize_factor"] > 1:
                stats["images_upscaled"] += 1
        info = seen[digest]
        row = {
            "id": row_id,
            "group_id": f"GeoMMBench:img:{digest}",
            "benchmark": "GeoMMBench",
            "split": "test" if split == "test" else "val",
            "task_family": "geommbench_mcq",
            "skills": ["expert_knowledge"],
            "images": [info["path"]],
            "state": "",
            "questions": {"q": {"type": "choice", "instructions": rec["question"].strip(),
                                "criteria": {o: None for o in options}}},
            "answer": {"q": gold},
            "meta": {"source": "AR-X/GeoMMBench@298d660bbbb37fd528a0122af6375e4aa0d7bdd3", "source_split": split,
                     "source_index": rec["index"], "published_options": published,
                     "published_answer_letter": rec["answer"].strip(),
                     "shuffled_answer_letter": "ABCD"[options.index(gold)],
                     "permutation": perm, "shuffle_seed": f"random.Random({row_id!r})",
                     "orig_px": info["orig_px"], "px": info["px"], "resize_factor": info["resize_factor"],
                     "image_sha1_16": digest, "adapted": "options shuffled; out-of-range images resized"},
        }
        validate_row(row)
        rows.append(row)
    return rows, stats


def main() -> None:
    for split, fname in (("test", "geommbench_test.jsonl"), ("val", "geommbench_val.jsonl")):
        rows, stats = convert(split)
        write_jsonl(OUT / fname, rows)
        print(summary(f"GeoMMBench {split}", rows))
        print(f"  images: {dict(stats)}; dropped: 0")
        print("  published answer letters:", dict(sorted(Counter(r["meta"]["published_answer_letter"] for r in rows).items())))
        if split == "test":
            print(sample_text(rows, 3))


if __name__ == "__main__":
    main()
