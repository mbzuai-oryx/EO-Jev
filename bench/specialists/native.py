"""Native-format variants: a specialist asked in its own training/eval format (its task tokens, label set, image
layout), with its labels mapped back to our options so the result is scored on our rows like any other run.

Used for in-domain models on our xBD rows, via bench/runners/vlm.py --native:
- EarthDial (its released xBD test files, akshaydudhane/EarthDial-Dataset, Change_detection/xBD):
  disaster type = xBD_img_cls_testset_1 (our 933 test scenes, pre + post), damage = xBD_reg_cls_testset_1 (pre + post
  full tiles, the building as <box>[[x0, y0, x1, y1, 0]]</box> in percent of the image, x first; its eval appends
  "Answer in single word." to this question). Questions are used verbatim, quotes and literal "\\n" included.
- WayBob Qwen3-VL-8B xView2 LoRA (WayBob/Disaster_Recognition_RemoteSense_EN_CN_JA, sharegpt json): post image only,
  at most 262,144 pixels (training image_max_pixels), two turns as in training: "Has a disaster occurred in this
  image?" with its post-image reply, then "What type of disaster occurred in this image?"; answers "A <type> disaster
  occurred in this image."
- TEOChat (TEOChatlas eval jsons, jirvin16/TEOChatlas eval/xBD_*.json): 256-px tiles of the 1024-px scenes (4 x 4
  grid), pre + post. Disaster type is asked per tile ("What disaster has occurred here? Choose one from: ...",
  answers "A hurricane." etc., the same six types as ours) for all 16 tiles of the scene; option probabilities are
  averaged over the tiles and written answers vote. Damage: the grid tile holding the building's centre, the building
  as "[x_min, y_min, x_max, y_max]" = round(px / 256 * 100) inside that tile, clipped to it.
EarthDial and WayBob use xBD's own disaster labels, where Hurricanes Harvey and Florence are "flood" and Matthew and Michael "wind";
our rows call all four "hurricane". wind -> hurricane is exact (no tornado in our test split); flood -> flood, so the
216 Harvey / Florence scenes cannot be right and the best reachable accuracy on our labels is 717 / 933 = 76.8%.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bench.specialists.prompting import Item, parse_answer, render


@dataclass
class Native:
    item: Item               # what the model sees (mode "names" over its own labels)
    ours: Item               # our rendering of the same question (to_answer / to_answer_generated)
    to_ours: list[int]       # native label index -> index into ours.labels
    image_paths: list[str]   # under converted/
    max_pixels: int | None = None
    # several views of the same question (e.g. the 16 tiles of a scene), each a crop box (left, top, right, bottom)
    # applied to every image; answers are combined over views (mean probabilities / majority vote of written answers)
    crops: list[tuple[int, int, int, int] | None] | None = None
    keywords: list[str] | None = None  # fallback parse of free text: one keyword per native label


def _scene(row: dict) -> str:
    return row["group_id"].split(":", 1)[1]


def _type_map(ours: Item, native_to_type: dict[str, str]) -> list[int]:
    return [ours.options.index(native_to_type[n]) for n in native_to_type]


def _earthdial(row: dict, qid: str, ours: Item) -> Native | None:
    if row["benchmark"] == "xBD-disaster-type":
        to_type = {"volcano": "volcanic eruption", "fire": "wildfire", "earthquake": "earthquake", "flood": "flood",
                   "tsunami": "tsunami", "wind": "hurricane"}
        text = ('"[changedet] [hr_rgb_temp_0.5] <image> <image> \\n Analyze the images to identify the type of '
                'disaster that occurred. Options:  volcano, fire, earthquake, flood, tsunami, wind."')
        item = Item(row["id"], qid, "choice", "names", text, list(to_type), native=True)
        return Native(item, ours, _type_map(ours, to_type), row["images"])
    if row["benchmark"] == "xBD-damage":
        y0, y1, x0, x1 = row["meta"]["box_yx"]
        size = 1024  # xBD tiles (bench/converters/xbd_damage.py: box_yx is in full-tile pixels)
        box = ", ".join(str(round(v * 100 / size)) for v in (x0, y0, x1, y1))
        labels = ["no-damage", "destroyed", "minor-damage", "major-damage"]  # order as listed in its question
        text = (f'"[identify] [hr_rgb_temp_0.5] <image> <image> \\n Compare the building at <box>[[{box}, 0]]</box> '
                "between the pre-disaster and post-disaster images. In the post-disaster image, classify the level "
                'of damage. Classes: no-damage, destroyed, minor-damage, major-damage."Answer in single word.')
        item = Item(row["id"], qid, "score", "names", text, labels, native=True)
        to_ours = [ours.options.index(lab.replace("-", " ")) for lab in labels]
        scene = _scene(row)
        return Native(item, ours, to_ours, [f"xbd/scenes/{scene}_pre.png", f"xbd/scenes/{scene}_post.png"])
    return None


def _waybob(row: dict, qid: str, ours: Item) -> Native | None:
    if row["benchmark"] != "xBD-disaster-type":
        return None
    to_type = {"fire": "wildfire", "flooding": "flood", "wind damage": "hurricane", "volcano": "volcanic eruption",
               "tsunami": "tsunami", "earthquake": "earthquake"}
    answers = [f"A {t} disaster occurred in this image." for t in to_type]
    item = Item(row["id"], qid, "choice", "names", "What type of disaster occurred in this image?", answers,
                native=True, turns=[("Has a disaster occurred in this image?",
                                     "Yes, a disaster has occurred in this image. This is a post-disaster image.")])
    return Native(item, ours, _type_map(ours, to_type), [row["images"][1]], max_pixels=262144,
                  keywords=["fire", "flood", "wind", "volcan", "tsunami", "earthquake"])


TILE = 256  # TEOChatlas xBD tiles


def _teochat(row: dict, qid: str, ours: Item) -> Native | None:
    head = "These are two satellite images capturing the same location at different times: <video> \n"
    scene = _scene(row)
    if row["benchmark"] == "xBD-disaster-type":
        to_type = {"A volcanic eruption.": "volcanic eruption", "A hurricane.": "hurricane",
                   "An earthquake.": "earthquake", "A flood.": "flood", "A tsunami.": "tsunami",
                   "A wildfire.": "wildfire"}
        text = head + ("What disaster has occurred here? Choose one from: a volcanic eruption, a hurricane, "
                       "an earthquake, a flood, a tsunami, a wildfire.")
        item = Item(row["id"], qid, "choice", "names", text, list(to_type), native=True)
        crops = [(c * TILE, r * TILE, (c + 1) * TILE, (r + 1) * TILE) for r in range(4) for c in range(4)]
        return Native(item, ours, _type_map(ours, to_type), row["images"], crops=crops,
                      keywords=["volcan", "hurricane", "earthquake", "flood", "tsunami", "wildfire"])
    if row["benchmark"] == "xBD-damage":
        y0, y1, x0, x1 = row["meta"]["box_yx"]
        r, c = min(3, (y0 + y1) // 2 // TILE), min(3, (x0 + x1) // 2 // TILE)
        left, top = c * TILE, r * TILE
        clip = lambda v, o: min(TILE, max(0, v - o))  # noqa: E731
        box = [round(clip(x0, left) / TILE * 100), round(clip(y0, top) / TILE * 100),
               round(clip(x1, left) / TILE * 100), round(clip(y1, top) / TILE * 100)]
        labels = ["No damage.", "Minor damage.", "Major damage.", "Destroyed."]
        text = head + (f"Classify the level of damage experienced by the building at location [{', '.join(map(str, box))}] "
                       "in the second image. Choose from: No damage, Minor Damage, Major Damage, Destroyed.")
        item = Item(row["id"], qid, "score", "names", text, labels, native=True)
        to_ours = [ours.options.index(lab.rstrip(".").lower()) for lab in labels]
        return Native(item, ours, to_ours, [f"xbd/scenes/{scene}_pre.png", f"xbd/scenes/{scene}_post.png"],
                      crops=[(left, top, left + TILE, top + TILE)],
                      keywords=["no damage", "minor", "major", "destroyed"])
    return None


RENDERERS = {"earthdial": _earthdial, "waybob-xview2": _waybob, "teochat": _teochat}


def render_native(model: str, row: dict, qid: str) -> Native | None:
    """The model's own format for this question, or None if it has none (the row is then skipped)."""
    fn = RENDERERS.get(model)
    return fn(row, qid, render(row, qid)) if fn else None


def parse_native(nat: Native, text: str) -> int | None:
    """Index into our labels from the model's free-text answer, or None."""
    i = parse_answer(nat.item, text)
    if i is None and nat.keywords:
        hits = [k for k, kw in enumerate(nat.keywords) if re.search(kw, text, re.I)]
        i = hits[0] if len(hits) == 1 else None
    return None if i is None else nat.to_ours[i]


def to_ours_probs(nat: Native, probs: list[float]) -> list[float]:
    """Probabilities over the model's labels -> over our labels (labels with no native counterpart get 0)."""
    out = [0.0] * len(nat.ours.labels)
    for i, p in enumerate(probs):
        out[nat.to_ours[i]] += p
    return out
