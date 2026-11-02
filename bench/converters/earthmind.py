"""C11 EarthMind-Bench (sy1998/EarthMind-Bench) multiple-choice tasks -> rows, in three input variants.

MC tasks (1,810): hallucination detection 295, object counting 842, object existence 199, scene classification
258, spatial relationship 216 (navigation, caption, disaster and urban answers are free text: skipped).
Every item has a co-registered optical and SAR image. Variants, as in the paper's evaluation:
  optical  (RGB image only)      sar  (SAR image only)      fused  (Image 1 optical + Image 2 SAR)
Paper MC averages, optical / SAR / fused: GPT-4o 64.1 / 47.8 / 61.1; EarthMind-4B 61.7 / 61.3 / 70.0.

    python -m bench.converters.earthmind
"""

from __future__ import annotations

import json
import re
from collections import Counter

from bench.common import CONVERTED, RAW, store_image, validate_row, write_jsonl

SRC = RAW / "earthmind-bench"
OUT = CONVERTED / "earthmind"
MC = ["hallucination_detection_all_unmatched.json", "object_counting_all_unmatched.json",
      "object_existence_all_unmatched.json", "scene_classification_unmatched.json",
      "spatial_relationship_all_unmatched.json"]
OPT = re.compile(r"^\s*([A-Z])\s*\.\s*(.*)$")
STATES = {"optical": "", "sar": "The image is a SAR (synthetic aperture radar) image.",
          "fused": "Image 1 is an optical image and Image 2 a SAR (synthetic aperture radar) image of the same area."}


def main() -> None:
    variants = {v: [] for v in STATES}
    for fname in MC:
        for i, x in enumerate(json.load(open(SRC / "json" / fname))):
            opts = [OPT.match(c) for c in x["candidate"]]
            if not all(opts):
                raise ValueError(f"unparsed options {x['candidate']}")
            letters = [m.group(1) for m in opts]
            texts = [m.group(2).strip() for m in opts]
            if len(set(texts)) != len(texts):
                texts = [f"{t} ({l})" for l, t in zip(letters, texts)]
            gold = texts[letters.index(x["answer"].strip().rstrip("."))]
            stem = x["file_name"].rsplit(".", 1)[0]
            rgb = store_image((SRC / "img/test/rgb/img" / f"{stem}.png").read_bytes(), OUT / "images" / "rgb" / stem)[0]
            sar = store_image((SRC / "img/test/sar/img" / f"{stem}.png").read_bytes(), OUT / "images" / "sar" / stem)[0]
            family = x["task_type"].replace(" ", "_")
            for variant, images in (("optical", [rgb]), ("sar", [sar]), ("fused", [rgb, sar])):
                row = {"id": f"EarthMind:{family}:{i}:{variant}", "group_id": f"EarthMind:{stem}",
                       "benchmark": f"EarthMind-Bench-{variant}", "split": "test", "task_family": family,
                       "skills": [family, variant], "images": images, "state": STATES[variant],
                       "questions": {"q": {"type": "choice", "instructions": x["question"],
                                           "criteria": {t: None for t in texts}}},
                       "answer": {"q": gold}, "meta": {"variant": variant, "source": stem}}
                validate_row(row)
                variants[variant].append(row)
    for variant, rows in variants.items():
        for i, r in enumerate(rows):  # two source files can share a task name: make ids unique
            r["id"] += f"#{i}"
        write_jsonl(OUT / f"earthmind_mc_{variant}.jsonl", rows)
    print("EarthMind-Bench:", {v: len(r) for v, r in variants.items()}, json.dumps(Counter(r["task_family"] for r in variants["optical"])))


if __name__ == "__main__":
    main()
