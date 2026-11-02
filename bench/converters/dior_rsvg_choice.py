"""C16 DIOR-RSVG visual grounding as 4-way box choice (official test split) -> rows. ADAPTED (box-choice).

Source: DIOR-RSVG (Zhan et al., TGRS 2023), the authors' Google Drive release linked from
github.com/ZhanYang-nwpu/RSVG-pytorch (research use): `Annotations.zip` (17,402 XMLs, 38,320 described objects)
and `test.txt` (7,500 indices). raw/dior-rsvg/test_pairs_sorted_order.json holds the 7,500 official test pairs
(index, image, box_xyxy in px, description, class) over 6,102 images, reconstructed with the official
data_loader's order (sorted XML paths, one entry per object). Images: raw/dior-rsvg/JPEGImages (800 x 800; 474 from
the official zip, the rest from the HF torchgeo/dior re-host @ 6b3a7a1f, see the manifest §11).
NOT used: raw/vhm-eval VG_DOIR_RSVG_test, which mixes train/val pairs into its 7,500 items (only 1,463 are test).

Adapted (docs/06 §8: not comparable to Acc@0.5; we run RSThinker ourselves on this version):
  - a seeded, class-stratified sample of 2,000 of the 7,500 pairs (proportional allocation per class, largest
    remainder; seed 0);
  - box choice: K = 4 lettered candidate boxes (A-D) drawn on the image in red / yellow / cyan / magenta with a black
    outline, each letter in a filled tag outside its box. The gold box plus 3 distractors, taken in this order:
      1. other described objects in the same image, same class first, then other classes (RSVG XMLs);
      2. boxes of other DIOR-RSVG objects from other images with a similar area (0.5-2x the gold box), same class
         first, then any class, at their own coordinates;
      3. similar-size random proposals (fallback; never needed when 2. has candidates).
    Every distractor has IoU < 0.3 with the gold box and with the other candidates. The gold letter is drawn with
    `random.Random(row_id)`; distractor order is seeded too. Chance = 1/4 (meta.chance).
  - question: state names the overlay; instructions = 'Referring expression: "<description>"\\nWhich box shows it?';
    `choice` over "box A" .. "box D".
  - images keep their 800 x 800 size (no resize); overlays are drawn at native resolution and saved as JPEG
    (quality 95, 4:4:4), one image per item.
Full DIOR annotations (all objects per image) are not used: no local copy covers the DIOR test half of these images,
and mixing sources would make distractor difficulty depend on the image's DIOR split.

    python -m bench.converters.dior_rsvg_choice
"""

from __future__ import annotations

import json
import random
import re
import zipfile
from collections import Counter, defaultdict

from PIL import Image

from bench.common import CONVERTED, RAW, validate_row, write_jsonl
from bench.converters.perception_utils import (BOX_COLOURS, draw_candidates, pick_candidates, sample_text, stratified,
                                               summary)

SRC = RAW / "dior-rsvg"
OUT = CONVERTED / "grounding"
N = 2000
K = 4
STATE = ("Four candidate boxes are drawn on the image, each in a different colour and labelled with a letter "
         "(A, B, C or D) placed next to the box.")
OBJ_RE = re.compile(r"<name>(.*?)</name>.*?<xmin>(\d+)</xmin>\s*<ymin>(\d+)</ymin>\s*<xmax>(\d+)</xmax>\s*"
                    r"<ymax>(\d+)</ymax>.*?<description>(.*?)</description>", re.S)


def load_objects() -> tuple[dict[str, dict], list[tuple[str, str, list[int]]]]:
    """Per image: size and described objects; plus a flat list (image, class, box) of every described object."""
    per_image, flat = {}, []
    with zipfile.ZipFile(SRC / "Annotations.zip") as zf:
        for m in sorted(zf.namelist()):
            if not m.endswith(".xml"):
                continue
            x = zf.read(m).decode()
            img = re.search(r"<filename>(.*?)</filename>", x).group(1)
            size = tuple(map(int, re.search(r"<width>(\d+)</width>\s*<height>(\d+)</height>", x).groups()))
            objs = [(o[0], [int(v) for v in o[1:5]]) for o in OBJ_RE.findall(x)]
            per_image[img] = {"size": size, "objects": objs}
            flat += [(img, c, b) for c, b in objs]
    return per_image, flat


def area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def main() -> None:
    pairs = json.load(open(SRC / "test_pairs_sorted_order.json"))
    per_image, flat = load_objects()
    official = set(json.load(open(SRC / "_official_rsvg_images_present.json")))
    by_class: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    for img, c, b in flat:
        by_class[c].append((img, b))
    all_boxes = [(img, b) for img, _, b in flat]

    picked = stratified(pairs, "class", N, random.Random(0))
    picked.sort(key=lambda p: p["index"])
    rows, sources, drawn = [], Counter(), 0
    for p in picked:
        row_id = f"DIOR-RSVG-choice:{p['index']}"
        rng = random.Random(row_id)
        gold = [float(v) for v in p["box_xyxy"]]
        ann = per_image[p["image"]]
        with Image.open(SRC / "JPEGImages" / p["image"]) as im:
            image = im.convert("RGB")
        w, h = image.size
        same_img = [(c, [float(v) for v in b]) for c, b in ann["objects"] if [float(v) for v in b] != gold]
        ga = area(gold)

        def similar(pool):
            return [[float(v) for v in b] for img, b in pool
                    if img != p["image"] and 0.5 <= area(b) / max(ga, 1.0) <= 2.0 and b[2] <= w and b[3] <= h]

        pools = [
            ("same_image_same_class", [b for c, b in same_img if c == p["class"]]),
            ("same_image_other_class", [b for c, b in same_img if c != p["class"]]),
            ("other_image_same_class", similar(by_class[p["class"]])),
            ("other_image_any_class", similar(all_boxes)),
        ]
        cands = pick_candidates(gold, pools, (w, h), rng, k=K)
        gold_pos = rng.randrange(K)
        distractors = cands[1:]
        order = distractors[:gold_pos] + [cands[0]] + distractors[gold_pos:]
        letters = "ABCD"
        dest = OUT / "images" / "dior_rsvg" / f"{p['index']}.jpg"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            draw_candidates(image, [b for _, b in order]).save(dest, format="JPEG", quality=95, subsampling=0)
            drawn += 1
        for src, _ in order[:gold_pos] + order[gold_pos + 1:]:
            sources[src] += 1
        options = [f"box {letters[i]}" for i in range(K)]
        row = {
            "id": row_id,
            "group_id": f"DIOR-RSVG:{p['image']}",
            "benchmark": "DIOR-RSVG-choice",
            "split": "test",
            "task_family": "grounding_box_choice",
            "skills": ["grounding", "referring_expression"],
            "images": [str(dest.relative_to(CONVERTED))],
            "state": STATE,
            "questions": {"q": {"type": "choice",
                                "instructions": f'Referring expression: "{p["description"].strip()}"\nWhich box shows it?',
                                "criteria": {o: None for o in options}}},
            "answer": {"q": options[gold_pos]},
            "meta": {"source": "DIOR-RSVG official test split (test.txt index, sorted-XML order)",
                     "rsvg_index": p["index"], "image": p["image"], "class": p["class"],
                     "description": p["description"], "gold_box_xyxy": p["box_xyxy"], "gold_letter": letters[gold_pos],
                     "candidates": [{"letter": letters[i], "source": s, "box_xyxy": b, "colour": BOX_COLOURS[i][0]}
                                    for i, (s, b) in enumerate(order)],
                     "k": K, "chance": 1 / K, "max_iou": 0.3, "orig_px": [w, h], "resize_factor": 1.0,
                     "image_source": "official RSVG zip" if p["image"] in official else "torchgeo/dior mirror",
                     "xml_size": list(ann["size"]),
                     "adapted": "box-choice (4 drawn candidates), 2k class-stratified subset"},
        }
        if list(ann["size"]) != [w, h]:
            row["meta"]["size_mismatch"] = f"XML says {ann['size']}, image is {(w, h)}"
        validate_row(row)
        rows.append(row)
    write_jsonl(OUT / "dior_rsvg_choice.jsonl", rows)
    print(summary("DIOR-RSVG-choice", rows))
    print("  classes:", dict(sorted(Counter(r["meta"]["class"] for r in rows).items())))
    print("  distractor sources:", dict(sources))
    print(f"  images drawn this run: {drawn}; resized: 0; dropped: 0;",
          "size mismatches:", sum("size_mismatch" in r["meta"] for r in rows))
    print(sample_text(rows, 3))


if __name__ == "__main__":
    main()
