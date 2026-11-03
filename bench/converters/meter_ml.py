"""C17 METER-ML scene classification, VHM evaluation split (565 images, 6 classes) -> rows. AS-PUBLISHED.

Source: HF FitzPC/VHM_eval_dataset @ 0a9d907a1bc4dc8431248420705cf9c9205333cf (raw/vhm-eval):
`json/cls_METER_ML.json` (565 items) and `datasets_eval.zip` -> `datasets_eval/cls_METER_ML/<n>.jpg` (720 x 720 NAIP
RGB JPEGs). This is the official METER-ML v1.0 test split (Zenodo 6911013) minus its Negative and multi-facility
samples: natural gas processing plants 107, wastewater treatment plants 106, landfills 95, oil refineries and
petroleum terminals 94, concentrated animal feeding operations 92, coal mines 71.

As published (cite RSCoVLM-7B 75.93 directly):
  - all 565 items, the published JPEG bytes unchanged (all are within MIN_PIXELS..MAX_PIXELS, so no resize);
  - `instructions` is VHM's prompt verbatim ("Classify the image within one of the given classes:<6 names>.Answer
    with one word or short phrase."); every item has the same prompt;
  - one `choice` over the 6 class names in the prompt's fixed order (not shuffled), task_family "scene_cls" so the
    specialist runners answer with the class name (bench/specialists/prompting.py "names" mode), as VHM does.
Not identical (runner-level, recorded here and in meta): the runners append their own option list / answer suffix,
so the class list appears twice in the text a model sees; VHM's own runs also prefix the task tag "{CLS} " (VHM
model only, RSEvalKit model_eval_mp.py convt_qa) and score a free-text answer by substring match
(`pred.replace(" ", "").lower() in answer.replace(" ", "").lower()`), whereas we score the chosen option.

    python -m bench.converters.meter_ml
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from bench.common import CONVERTED, MAX_PIXELS, MIN_PIXELS, RAW, store_image, validate_row, write_jsonl

SRC = RAW / "vhm-eval"
OUT = CONVERTED / "meter_ml"
REVISION = "FitzPC/VHM_eval_dataset@0a9d907a1bc4dc8431248420705cf9c9205333cf"
PROMPT_RE = re.compile(r"^Classify the image within one of the given classes:(.*)\.Answer with one word or short phrase\.$")


def main() -> None:
    items = json.load(open(SRC / "json" / "cls_METER_ML.json"))
    prompts = {x["conversations"][0]["value"] for x in items}
    if len(prompts) != 1:
        raise ValueError(f"expected one prompt, got {len(prompts)}")
    prompt = prompts.pop()
    classes = [c.strip() for c in PROMPT_RE.match(prompt).group(1).split(",")]
    if len(classes) != 6:
        raise ValueError(f"expected 6 classes, got {classes}")

    rows, sizes = [], Counter()
    with zipfile.ZipFile(SRC / "datasets_eval.zip") as zf:
        for k, x in enumerate(items):
            label = x["conversations"][1]["value"].strip()
            if label not in classes:
                raise ValueError(f"{x['image']}: label {label!r} not in the prompt's classes")
            stem = Path(x["image"]).stem
            rel, size = store_image(zf.read(f"datasets_eval/{x['image_path']}/{x['image']}"), OUT / "images" / stem)
            if not MIN_PIXELS <= size[0] * size[1] <= MAX_PIXELS:
                raise ValueError(f"{x['image']}: {size} outside the pixel limits; this converter keeps images as published")
            sizes[tuple(size)] += 1
            row = {
                "id": f"METER-ML:{stem}",
                "group_id": f"METER-ML:{stem}",
                "benchmark": "METER-ML",
                "split": "test",
                "task_family": "scene_cls",
                "skills": ["lulc", "facility"],
                "images": [rel],
                "state": "",
                "questions": {"scene": {"type": "choice", "instructions": prompt,
                                        "criteria": {c: None for c in classes}}},
                "answer": {"scene": label},
                "meta": {"source": REVISION, "source_json": "json/cls_METER_ML.json", "source_index": k,
                         "image": x["image"], "orig_px": size, "resize_factor": 1.0, "as_published": True,
                         "published_prompt": prompt, "class_order": "VHM prompt order (fixed)",
                         "vhm_task_tag": "{CLS}", "sensor": "NAIP RGB", "gsd_m": 1.0},
            }
            validate_row(row)
            rows.append(row)
    n = write_jsonl(OUT / "meter_ml_vhm.jsonl", rows)
    print(f"METER-ML (VHM split): {n} rows, {n} choice questions (task_family scene_cls), {len(rows)} images")
    print("image sizes:", dict(sizes), "| resized: 0 | dropped: 0")
    print("answers:", Counter(r["answer"]["scene"] for r in rows))


if __name__ == "__main__":
    main()
