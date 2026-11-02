"""C9 xBD (HF mirror, test split: 933 scenes, 10 events) -> two row files.

Building damage: buildings are connected components of the pre-event footprint mask (>= 100 px); a building's
level is the majority post-event label inside it (1 no damage .. 4 destroyed on the mirror's mask -> 0..3).
A class-balanced, seeded 3,000-building sample (750 per level) is asked one 4-level `score` question over two
images, the pre- and post-event crops (2x the building's box, at least 96 px), with the building outlined in red.

Disaster type: every scene, pre + post 1024 px images, one 7-way `choice` (event -> type).

    python -m bench.converters.xbd_damage
"""

from __future__ import annotations

import io
import random
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from bench.common import CONVERTED, save_png, validate_row, write_jsonl
from bench.converters.xbd_mirror import scenes

OUT = CONVERTED / "xbd"
PER_LEVEL = 750
MIN_AREA = 100
LEVELS = ["no damage", "minor damage", "major damage", "destroyed"]
# xBD Joint Damage Scale definitions (Gupta et al. 2019), kept for the label-wording ablation (A7).
LEVEL_DEFINITIONS = [
    "no damage: undisturbed, no sign of water, structural or shingle damage, or burn marks",
    "minor damage: partially burnt, water surrounding the structure, volcanic flow nearby, roof elements missing, or visible cracks",
    "major damage: partial wall or roof collapse, encroaching volcanic flow, or surrounded by water or mud",
    "destroyed: scorched, completely collapsed, partially or completely covered with water or mud, or no longer present",
]
EVENT_TYPE = {
    "socal-fire": "wildfire", "santa-rosa-wildfire": "wildfire", "woolsey-fire": "wildfire",
    "pinery-bushfire": "wildfire", "portugal-wildfire": "wildfire",
    "hurricane-florence": "hurricane", "hurricane-harvey": "hurricane", "hurricane-michael": "hurricane",
    "hurricane-matthew": "hurricane",
    "midwest-flooding": "flood", "nepal-flooding": "flood",
    "palu-tsunami": "tsunami", "sunda-tsunami": "tsunami",
    "mexico-earthquake": "earthquake",
    "guatemala-volcano": "volcanic eruption", "lower-puna-volcano": "volcanic eruption",
    "joplin-tornado": "tornado", "moore-tornado": "tornado", "tuscaloosa-tornado": "tornado",
}
DISASTER_TYPES = ["wildfire", "hurricane", "flood", "tsunami", "earthquake", "volcanic eruption", "tornado"]
# xBD's own disaster_type labels (the label JSONs' metadata; also used by EarthDial and other xBD-trained models):
# hurricanes are "wind" or "flooding" depending on the dominant damage (Harvey / Florence: flooding).
# xbd_disaster_type_xbdlabels.jsonl uses these (bench/converters/xbd_type_xbdlabels.py).
EVENT_TYPE_XBD = {
    "socal-fire": "fire", "santa-rosa-wildfire": "fire", "woolsey-fire": "fire", "pinery-bushfire": "fire",
    "portugal-wildfire": "fire",
    "hurricane-florence": "flooding", "hurricane-harvey": "flooding", "midwest-flooding": "flooding",
    "nepal-flooding": "flooding",
    "hurricane-michael": "wind", "hurricane-matthew": "wind", "joplin-tornado": "wind", "moore-tornado": "wind",
    "tuscaloosa-tornado": "wind",
    "palu-tsunami": "tsunami", "sunda-tsunami": "tsunami",
    "mexico-earthquake": "earthquake",
    "guatemala-volcano": "volcano", "lower-puna-volcano": "volcano",
}
DISASTER_TYPES_XBD = ["fire", "flooding", "wind", "earthquake", "tsunami", "volcano"]
DAMAGE_Q = ("Image 1 was taken before and Image 2 after a disaster; both show the same place. "
            "How damaged is the building outlined in red?")
TYPE_Q = "Image 1 was taken before and Image 2 after a natural disaster; both show the same place. What type of disaster occurred?"


def buildings(pre_mask: np.ndarray, post_mask: np.ndarray) -> list[tuple[tuple[slice, slice], int]]:
    labels, n = ndimage.label(pre_mask > 0)
    out = []
    for i, box in enumerate(ndimage.find_objects(labels), 1):
        inside = labels[box] == i
        if inside.sum() < MIN_AREA:
            continue
        levels = post_mask[box][inside]
        levels = levels[(levels > 0) & (levels < 5)]
        if levels.size:
            out.append((box, int(np.bincount(levels).argmax()) - 1))
    return out


def crop(image: Image.Image, box: tuple[slice, slice]) -> Image.Image:
    (y0, y1), (x0, x1) = (box[0].start, box[0].stop), (box[1].start, box[1].stop)
    side = max(96, 2 * max(y1 - y0, x1 - x0))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    top = min(max(0, cy - side // 2), image.height - side)
    left = min(max(0, cx - side // 2), image.width - side)
    out = image.crop((left, top, left + side, top + side)).copy()
    ImageDraw.Draw(out).rectangle((x0 - left - 2, y0 - top - 2, x1 - left + 1, y1 - top + 1), outline=(255, 0, 0), width=2)
    return out


def main() -> None:
    def png(blob: dict) -> Image.Image:
        return Image.open(io.BytesIO(blob["bytes"]))

    # Pass 1: every building's level, plus the scene-level disaster-type rows.
    candidates: dict[int, list[tuple[str, tuple[slice, slice]]]] = defaultdict(list)
    type_rows = []
    for row in scenes(columns=["image_name", "t1_image", "t2_image", "t1_mask", "t2_mask"], splits=("test",)):
        scene = row["scene"]
        for box, level in buildings(np.array(png(row["t1_mask"])), np.array(png(row["t2_mask"]))):
            candidates[level].append((scene, box))
        pre = save_png(png(row["t1_image"]), OUT / "scenes" / f"{scene}_pre.png")
        post = save_png(png(row["t2_image"]), OUT / "scenes" / f"{scene}_post.png")
        event = scene.rsplit("_", 1)[0]
        options = DISASTER_TYPES[:]
        random.Random(scene).shuffle(options)
        r = {"id": f"xBD-type:{scene}", "group_id": f"xBD:{scene}", "benchmark": "xBD-disaster-type", "split": "test",
             "task_family": "disaster", "skills": ["disaster", "temporal"],
             "images": [str(pre.relative_to(CONVERTED)), str(post.relative_to(CONVERTED))], "state": "",
             "questions": {"q": {"type": "choice", "instructions": TYPE_Q, "criteria": {o: None for o in options}}},
             "answer": {"q": EVENT_TYPE[event]}, "meta": {"event": event, "gsd_m": "<=0.8"}}
        validate_row(r)
        type_rows.append(r)

    # Balanced sample, then pass 2 to cut the crops.
    rng = random.Random(0)
    picked: dict[str, list[tuple[tuple[slice, slice], int]]] = defaultdict(list)
    for level in range(4):
        for scene, box in rng.sample(candidates[level], PER_LEVEL):
            picked[scene].append((box, level))
    damage_rows = []
    for row in scenes(columns=["image_name", "t1_image", "t2_image"], splits=("test",)):
        scene = row["scene"]
        if scene not in picked:
            continue
        pre_img, post_img = png(row["t1_image"]).convert("RGB"), png(row["t2_image"]).convert("RGB")
        for box, level in picked[scene]:
            key = f"{scene}_{box[0].start}_{box[1].start}"
            pre = save_png(crop(pre_img, box), OUT / "buildings" / f"{key}_pre.png")
            post = save_png(crop(post_img, box), OUT / "buildings" / f"{key}_post.png")
            r = {"id": f"xBD-damage:{key}", "group_id": f"xBD:{scene}", "benchmark": "xBD-damage", "split": "test",
                 "task_family": "damage_ordinal", "skills": ["disaster", "temporal", "ordinal"],
                 "images": [str(pre.relative_to(CONVERTED)), str(post.relative_to(CONVERTED))], "state": "",
                 "questions": {"q": {"type": "score", "instructions": DAMAGE_Q, "criteria": LEVELS}},
                 "answer": {"q": level},
                 "meta": {"event": scene.rsplit("_", 1)[0], "box_yx": [box[0].start, box[0].stop, box[1].start, box[1].stop],
                          "building_px": (box[0].stop - box[0].start) * (box[1].stop - box[1].start),
                          "level_definitions": LEVEL_DEFINITIONS}}
            validate_row(r)
            damage_rows.append(r)

    write_jsonl(OUT / "xbd_damage_3k.jsonl", damage_rows)
    write_jsonl(OUT / "xbd_disaster_type.jsonl", type_rows)
    print("xBD damage:", len(damage_rows), Counter(r["answer"]["q"] for r in damage_rows))
    print("xBD disaster type:", len(type_rows), Counter(r["answer"]["q"] for r in type_rows))
    print("building candidates per level:", {k: len(v) for k, v in sorted(candidates.items())})


if __name__ == "__main__":
    main()
