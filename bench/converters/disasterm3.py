"""C10 DisasterM3 bench (NeurIPS'25 D&B) closed-form tasks -> rows (seeded ~5k sample, stratified).

Six closed-form tasks (12,968 items; reports, restoration advice and segmentation are generation-only):
  Building Damage Counting (4,982), Road Damage Counting (2,178), Disaster Type Recognition (420),
  Relational Reasoning (1,018; one post-event image with the red / blue boxes drawn)  -> 5-option `choice`
  Disaster Bearing Bodies Recognition (2,363) and Disaster Scene Recognition (2,007), multi-select over
  8 options -> one `noul` per option (exact-set match downstream).
Pre/post pairs go in as two images; the state says which is which and whether the post image is SAR.

    python -m bench.converters.disasterm3
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict

from bench.common import CONVERTED, RAW, store_image, validate_row, write_jsonl

SRC = RAW / "disasterm3" / "DisasterM3_Bench"
OUT = CONVERTED / "disasterm3"
QUOTA = {"Building Damage Counting": 1500, "Road Damage Counting": 800, "Disaster Type Recognition": 420,
         "Relational Reasoning": 700, "Disaster Bearing Bodies Recognition": 900, "Disaster Scene Recognition": 700}
MULTI = {"Disaster Bearing Bodies Recognition", "Disaster Scene Recognition"}


def path(p: str) -> str:
    return p.replace("\\", "/")


def main() -> None:
    items = [x for x in json.load(open(SRC / "benchmark_release.json")) if x["task"] in QUOTA]
    strata = defaultdict(list)
    for i, x in enumerate(items):
        x["_i"] = i
        strata[(x["task"], x.get("post_image_type") or x.get("image_type"))].append(x)
    rng = random.Random(0)
    picked = []
    for task, quota in QUOTA.items():
        groups = {k: v for k, v in strata.items() if k[0] == task}
        total = sum(len(v) for v in groups.values())
        for key, xs in sorted(groups.items()):
            picked += rng.sample(xs, min(len(xs), round(quota * len(xs) / total)))

    rows, skipped = [], []
    for x in picked:
        task = x["task"]
        options = [str(o) for o in x["options_list"]]
        letters = [l.strip() for l in str(x["ground_truth_option"]).strip().rstrip(".").split(",")]
        golds = {options[ord(l) - ord("A")] for l in letters}
        modality = x.get("post_image_type") or x.get("image_type")
        if "pre_image_path" in x:
            srcs = [path(x["pre_image_path"]), path(x["post_image_path"])]
            state = f"Image 1 was taken before the disaster (optical); Image 2 after it ({modality})."
        else:
            srcs = [path(x["image_path"])]
            state = f"A {modality} image taken after the disaster."
        try:
            images = [store_image((SRC / s).read_bytes(), OUT / "images" / s.rsplit(".", 1)[0])[0] for s in srcs]
        except Exception:  # noqa: BLE001 - the release has a zero-byte image (tuscaloosa_tornado_00000278_pre)
            skipped.append(x["_i"])
            continue
        if task in MULTI:
            questions = {f"opt{i}": {"type": "noul", "instructions": f"{x['prompts']} Is \"{o}\" one of them?"}
                         for i, o in enumerate(options)}
            answer = {f"opt{i}": o in golds for i, o in enumerate(options)}
        else:
            if len(set(options)) != len(options) or len(golds) != 1:
                continue
            questions = {"q": {"type": "choice", "instructions": x["prompts"], "criteria": {o: None for o in options}}}
            answer = {"q": golds.pop()}
        family = task.lower().replace(" ", "_") + ("_multi" if task in MULTI else "")
        row = {"id": f"DisasterM3:{x['_i']}", "group_id": f"DisasterM3:{srcs[-1]}", "benchmark": "DisasterM3",
               "split": "bench", "task_family": family, "skills": ["disaster"] + (["sar"] if modality == "SAR" else []),
               "images": images, "state": state, "questions": questions, "answer": answer,
               "meta": {"task": task, "post_modality": modality, "multi_select": task in MULTI,
                        "cls_description": x.get("cls_description")}}
        validate_row(row)
        rows.append(row)
    n = write_jsonl(OUT / "disasterm3_mc_5k.jsonl", rows)
    print("DisasterM3:", n, "rows;", json.dumps(Counter(r["task_family"] for r in rows)),
          json.dumps(Counter(r["meta"]["post_modality"] for r in rows)), "| skipped (unreadable image):", skipped)


if __name__ == "__main__":
    main()
