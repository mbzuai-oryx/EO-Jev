"""C15 (part) UrBench geo-localization dimension (AAAI'25; bczhou/UrBench test split) -> rows.

  city-retrieval (1,852)      which city? (street-view or satellite views)
  image-retrieval (723)       satellite image -> which of four street views matches (options are images)
  camera-localization (956)   which satellite quadrant the street view was shot from
  orientation (580)           which marked direction a panorama faces
All are 4-option MCQs over up to 6 images. d3 places every image before the text, so each "<image>"
placeholder in the question / options is rewritten to "Image k" (k = order of the image columns). Street-view-only
items (view_type "street-view") are not overhead imagery: slice them out with --by meta view_type when reporting.

    python -m bench.converters.urbench
"""

from __future__ import annotations

import ast
import json
from collections import Counter

import pyarrow.parquet as pq

from bench.common import CONVERTED, RAW, store_image, validate_row, write_jsonl

SRC = RAW / "urbench"
OUT = CONVERTED / "urbench"
TASKS = ["city-retrieval", "image-retrieval", "camera-localization", "orientation"]


def main() -> None:
    rows = []
    for task in TASKS:
        for f in sorted((SRC / task).glob("test-*.parquet")):
            for r in pq.read_table(f).to_pylist():
                blobs = [r[f"image_{k}"] for k in range(1, 7) if r.get(f"image_{k}")]
                images = [store_image(b["bytes"], OUT / "images" / task / f"{r['id']}_{k}")[0]
                          for k, b in enumerate(blobs, 1)]
                counter = iter(range(1, len(images) + 1))
                question = r["question"]
                while "<image>" in question:
                    question = question.replace("<image>", f"(Image {next(counter)})", 1)
                choices = [str(c) for c in ast.literal_eval(r["choices"])]
                texts = []
                for c in choices:
                    while "<image>" in c:
                        c = c.replace("<image>", f"Image {next(counter)}", 1)
                    texts.append(c.strip())
                duplicated = len(set(texts)) != len(texts)
                if duplicated:  # source bug (e.g. "Córdoba" listed twice): keep both, told apart by letter
                    texts = [f"{t} ({chr(ord('A') + i)})" for i, t in enumerate(texts)]
                gold = texts[ord(r["answer"].strip()) - ord("A")]
                row = {"id": f"UrBench:{r['id']}", "group_id": f"UrBench:{r['id']}", "benchmark": "UrBench-geo",
                       "split": "test", "task_family": task.replace("-", "_"), "skills": ["geolocation", r["view_type"]],
                       "images": images, "state": "",
                       "questions": {"q": {"type": "choice", "instructions": question.strip(),
                                           "criteria": {t: None for t in texts}}},
                       "answer": {"q": gold},
                       "meta": {"view_type": r["view_type"], "source": r["source"], "n_images": len(images),
                                "duplicate_options_in_source": duplicated}}
                validate_row(row)
                rows.append(row)
    n = write_jsonl(OUT / "urbench_geo.jsonl", rows)
    print("UrBench-geo:", n, "rows;", json.dumps(Counter((r["task_family"], r["meta"]["view_type"]) for r in rows).most_common()))


if __name__ == "__main__":
    main()
