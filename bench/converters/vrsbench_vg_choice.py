"""C3b VRSBench visual grounding (referring) as 4-way box choice -> rows. ADAPTED (box-choice).

Source: VRSBench (NeurIPS'24 D&B; HF xiang709/VRSBench, local raw/vrsbench): `VRSBench_EVAL_referring.json`
(16,159 referring items over 9,318 val images: 6,736 `unique`, 9,423 non-unique; boxes `{<x1><y1><x2><y2>}` in
0-100 plus the oriented `obj_corner` in 0-1), `Annotations_val.zip` (per-image JSON with every described object of
that image) and `Images_val.zip` (512 x 512 PNGs, DOTA-v2 / DIOR crops).

Adapted (docs/06 §8: not comparable to Acc@0.5; we run RSThinker ourselves on this version):
  - a seeded sample of 2,000: 1,000 unique + 1,000 non-unique items (meta.is_unique), each half stratified by
    object class (proportional, largest remainder; seed 0);
  - gold box = the extent of `obj_corner` clipped to the image (pixel-exact; the published 0-100 `ground_truth`,
    which is the same extent rounded down, is kept in meta);
  - box choice exactly as C16 (bench/converters/dior_rsvg_choice.py): K = 4 lettered boxes (A-D; red / yellow /
    cyan / magenta, black outline, letter tag outside the box); distractors in order 1. other described objects in
    the same image, same class first, 2. boxes of described objects of similar area (0.5-2x) from other VRSBench
    images, same class first, at their own coordinates, 3. similar-size random proposals (fallback); every
    distractor has IoU < 0.3 with the gold box and with the other candidates; gold letter seeded by row id;
    chance 1/4. VRSBench describes 1-7 objects per image (not every object), so most distractors come from 2.;
  - question: instructions = 'Referring expression: "<question>"\\nWhich box shows it?', `choice` over
    "box A" .. "box D";
  - images keep 512 x 512 (no resize); overlays at native resolution, saved as JPEG (quality 95, 4:4:4).

    python -m bench.converters.vrsbench_vg_choice
"""

from __future__ import annotations

import io
import json
import random
import re
import zipfile
from collections import Counter, defaultdict

from PIL import Image

from bench.common import CONVERTED, RAW, validate_row, write_jsonl
from bench.converters.perception_utils import (BOX_COLOURS, draw_candidates, pick_candidates, sample_text, stratified,
                                               summary)

SRC = RAW / "vrsbench"
OUT = CONVERTED / "grounding"
N_PER_FLAG = 1000
K = 4
STATE = ("Four candidate boxes are drawn on the image, each in a different colour and labelled with a letter "
         "(A, B, C or D) placed next to the box.")


def extent(corner: list[float], w: int, h: int) -> list[float]:
    xs, ys = corner[0::2], corner[1::2]
    clip = lambda v, hi: min(max(v, 0.0), 1.0) * hi  # noqa: E731
    return [round(clip(min(xs), w), 1), round(clip(min(ys), h), 1), round(clip(max(xs), w), 1), round(clip(max(ys), h), 1)]


def area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def main() -> None:
    items = json.load(open(SRC / "VRSBench_EVAL_referring.json"))
    with zipfile.ZipFile(SRC / "Annotations_val.zip") as zf:
        ann = {m.split("/")[-1][:-5] + ".png": json.loads(zf.read(m)) for m in zf.namelist()
               if m.startswith("Annotations_val/") and m.endswith(".json")}
    size = 512  # every VRSBench image is 512 x 512 (asserted below when drawing)
    flat = [(img, o["obj_cls"], extent(o["obj_corner"], size, size)) for img, a in sorted(ann.items()) for o in a["objects"]]
    by_class: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    for img, c, b in flat:
        by_class[c].append((img, b))
    all_boxes = [(img, b) for img, _, b in flat]

    for it in items:
        it["_flag"] = "unique" if it["unique"] else "non-unique"
    rng0 = random.Random(0)
    picked = []
    for flag in ("unique", "non-unique"):
        picked += stratified([it for it in items if it["_flag"] == flag], "obj_cls", N_PER_FLAG, rng0)
    picked.sort(key=lambda it: it["question_id"])

    rows, sources, drawn = [], Counter(), 0
    with zipfile.ZipFile(SRC / "Images_val.zip") as zf:
        for it in picked:
            row_id = f"VRSBench-VG-choice:{it['question_id']}"
            rng = random.Random(row_id)
            with Image.open(io.BytesIO(zf.read(f"Images_val/{it['image_id']}"))) as im:
                image = im.convert("RGB")
            w, h = image.size
            if (w, h) != (size, size):
                raise ValueError(f"{it['image_id']}: unexpected size {(w, h)}")
            gold = extent(it["obj_corner"], w, h)
            others = [(o["obj_cls"], extent(o["obj_corner"], w, h)) for o in ann[it["image_id"]]["objects"]
                      if o["obj_corner"] != it["obj_corner"]]
            ga = area(gold)

            def similar(pool):
                return [b for img, b in pool if img != it["image_id"] and 0.5 <= area(b) / max(ga, 1.0) <= 2.0]

            pools = [
                ("same_image_same_class", [b for c, b in others if c == it["obj_cls"]]),
                ("same_image_other_class", [b for c, b in others if c != it["obj_cls"]]),
                ("other_image_same_class", similar(by_class[it["obj_cls"]])),
                ("other_image_any_class", similar(all_boxes)),
            ]
            cands = pick_candidates(gold, pools, (w, h), rng, k=K)
            gold_pos = rng.randrange(K)
            order = cands[1:gold_pos + 1] + [cands[0]] + cands[gold_pos + 1:]
            letters = "ABCD"
            dest = OUT / "images" / "vrsbench_vg" / f"{it['question_id']}.jpg"
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                draw_candidates(image, [b for _, b in order]).save(dest, format="JPEG", quality=95, subsampling=0)
                drawn += 1
            for i, (src, _) in enumerate(order):
                if i != gold_pos:
                    sources[src] += 1
            options = [f"box {letters[i]}" for i in range(K)]
            row = {
                "id": row_id,
                "group_id": f"VRSBench:{it['image_id']}",
                "benchmark": "VRSBench-VG-choice",
                "split": "test",
                "task_family": "grounding_box_choice_" + ("unique" if it["unique"] else "nonunique"),
                "skills": ["grounding", "referring_expression"],
                "images": [str(dest.relative_to(CONVERTED))],
                "state": STATE,
                "questions": {"q": {"type": "choice",
                                    "instructions": f'Referring expression: "{it["question"].strip()}"\nWhich box shows it?',
                                    "criteria": {o: None for o in options}}},
                "answer": {"q": options[gold_pos]},
                "meta": {"source": "VRSBench_EVAL_referring.json", "question_id": it["question_id"],
                         "image": it["image_id"], "class": it["obj_cls"], "is_unique": it["unique"],
                         "size_group": it["size_group"], "published_ground_truth": it["ground_truth"],
                         "obj_corner": it["obj_corner"], "gold_box_xyxy": gold, "gold_letter": letters[gold_pos],
                         "candidates": [{"letter": letters[i], "source": s, "box_xyxy": b, "colour": BOX_COLOURS[i][0]}
                                        for i, (s, b) in enumerate(order)],
                         "k": K, "chance": 1 / K, "max_iou": 0.3, "orig_px": [w, h], "resize_factor": 1.0,
                         "adapted": "box-choice (4 drawn candidates), 2k subset (1k unique + 1k non-unique)"},
            }
            validate_row(row)
            rows.append(row)
    write_jsonl(OUT / "vrsbench_vg_choice.jsonl", rows)
    print(summary("VRSBench-VG-choice", rows))
    print("  unique flag:", dict(Counter(r["meta"]["is_unique"] for r in rows)))
    print("  classes:", dict(Counter(r["meta"]["class"] for r in rows).most_common()))
    print("  distractor sources:", dict(sources))
    print(f"  images drawn this run: {drawn}; resized: 0; dropped: 0")
    print(sample_text(rows, 3))


if __name__ == "__main__":
    main()
