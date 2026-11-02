"""C1 GEOBench-VLM "single" split (3,211 five-option MCQs, 17 tasks) -> rows.

Uses the first of the five paraphrased prompts as the question (all five are kept in meta for the paraphrase-
consistency test); options are the benchmark's own five, in its order (the benchmark already randomizes the
answer position). Images are BMP in the source, re-encoded to PNG. Duplicate option strings (44 items) are
merged, which is recorded in meta.

    python -m bench.converters.geobench_vlm
"""

from __future__ import annotations

import io
import json

import pyarrow.parquet as pq
from PIL import Image

from bench.common import CONVERTED, RAW, save_png, validate_row, write_jsonl
from bench.converters.xbd_mirror import images_by_original_name

OUT = CONVERTED / "geobench_vlm"

FAMILY = {  # task -> (task_family, skills)
    "Aircraft Type Classification": ("fine_grained", ["object_cls", "fine_grained"]),
    "Ship Type Classification": ("fine_grained", ["object_cls", "fine_grained"]),
    "Crop Type Classification": ("lulc", ["lulc", "fine_grained"]),
    "Land Use Classification": ("lulc", ["lulc"]),
    "Scene Classification": ("scene_cls", ["lulc"]),
    "Disaster Type Classification": ("disaster", ["disaster"]),
    "Fire Risk Assessment": ("ordinal_assessment", ["disaster", "ordinal"]),
    "Tree Health Assessment": ("ordinal_assessment", ["vegetation", "ordinal"]),
    "Sptatial Relation Classification": ("spatial_relation", ["spatial_relation"]),
}


def main() -> None:
    table = pq.read_table(RAW / "geobench-vlm" / "Single.parquet").to_pylist()
    # 210 Disaster Type items reference xBD images the benchmark does not redistribute ("xBD_<scene>_post_disaster.png");
    # fill them from the xBD mirror, as the benchmark's preprocess_xbd.py does from an xBD download.
    xbd_names = {rec["image_name"].removeprefix("xBD_") for rec in table if rec["image"] is None}
    xbd = images_by_original_name(xbd_names) if xbd_names else {}
    rows = []
    for rec in table:
        if rec["image"] is None:
            rec["image"] = {"bytes": xbd[rec["image_name"].removeprefix("xBD_")]}
        task = rec["task"]
        family, skills = FAMILY.get(task, ("counting", ["counting"]) if "Counting" in task else ("other", []))
        options = list(dict.fromkeys(rec["options_list"]))  # keep order, drop repeats
        qid = f"{rec['question_id']}"
        image = Image.open(io.BytesIO(rec["image"]["bytes"]))
        rel = str(save_png(image, OUT / "images" / f"{qid}.png").relative_to(CONVERTED))
        row = {
            "id": f"GEOBench-VLM:{qid}",
            "group_id": f"GEOBench-VLM:{rec['image_name']}",
            "benchmark": "GEOBench-VLM",
            "split": "single",
            "task_family": family,
            "skills": skills,
            "images": [rel],
            "state": "",
            "questions": {"q": {"type": "choice", "instructions": rec["prompts"][0],
                                "criteria": {o: None for o in options}}},
            "answer": {"q": rec["ground_truth"]},
            "meta": {"task": task, "orig_px": list(image.size), "n_options": len(options),
                     "merged_duplicate_options": len(options) != len(rec["options_list"]),
                     "prompts": rec["prompts"], "cls_description": rec["cls_description"] or None,
                     "source_image": rec["image_path"]},
        }
        validate_row(row)
        rows.append(row)
    n = write_jsonl(OUT / "geobench_vlm_single.jsonl", rows)
    print("GEOBench-VLM single:", n, "rows")


if __name__ == "__main__":
    main()
