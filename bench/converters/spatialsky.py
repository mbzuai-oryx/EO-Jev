"""C31 SpatialSky-Bench (arXiv 2511.13269, CVPR'26; 13 tasks x 100 items on UAVScenes frames) -> closed-form rows.

Source: GitHub `linglingxiansen/SpatialSKy` @ 90ab3b840a0206c55161babf68ba32bef5f988e6 (README badge Apache-2.0, no
LICENSE file): `benchmark/*.json` (raw/spatialsky-bench/repo/); the camera frames are UAVScenes (HF `sijieaaa/UAVScenes`
@ 04a224fd14b0b8fdfd2633e68f20471b85c611a5, CC-BY-SA-4.0), 2448x2048 JPEGs (5.0 MP), in raw/spatialsky-bench/UAVScenes/.

ADAPTED (closed subset + resolution + reverse-pointing marker); report it as "not the published protocol".
The 6 tasks the official metric scores by option / category match (metric/choose.py) are kept, 600 items:
  color    (100) 4 lettered options                                     -> `choice` over the option texts
  spatial  (100) 4-6 lettered options                                   -> `choice` over the option texts
  counting (100) "How many ...?", answers 1-10, exactly 10 of each      -> 10-way `choice` "1" .. "10"
  distance (100) 73 "Which is farther from me / closer to me, X or Y?"  -> 2-way `choice` [X, Y] (named order);
                 27 "What is the farthest / closest object ...?"        -> 19-way `choice` over the published
                                                                           category list (as published)
  height   (100) 74 "Which is lower / higher, X or Y?" -> 2-way; 26 "What is the highest / lowest object ...?"
                 -> 19-way, as for distance (meta.form = "pair" | "superlative")
  reverse  (100) "What object is at (x, y)?" with native pixel coordinates -> 19-way `choice` over the published
                 category list. A red ring (radius 12 px, 3 px, white-edged) is drawn AFTER resizing, centred on
                 the point scaled to the shown image; the question's "(x, y)" becomes the scaled coordinates
                 followed by "[the centre of the red circle]", and the state names the image size.
Images: resized exactly as the official inference sees them (parallel_inference.py: Qwen2.5-VL processor with
max_pixels = 1,605,632 and min_pixels = 200,704, i.e. qwen-vl-utils smart_resize with factor 28): 2448x2048 ->
1372x1148 (Lanczos, JPEG q95; marked reverse images PNG). Original size and factors are in meta.
Kept: question wording (the boilerplate "You are a vision-language assistant", the answer-format instructions and
the inline option / category lists are removed: the runners add lettered options), option texts and order.
Dropped (not closed-form; documented): pointing, freespace, function (points / free text), bbox (boxes), landing
(JSON safety report), single_image_caption, multi_image_caption (captions) - 700 items.

    python -m bench.converters.spatialsky
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from bench.common import CONVERTED, RAW, validate_row, write_jsonl
from bench.converters.uav_common import (answer_balance, atomic_save, rel, resize, show_samples)

SRC = RAW / "spatialsky-bench"
BENCH = SRC / "repo" / "benchmark"
OUT = CONVERTED / "spatialsky"
COMMIT = "90ab3b840a0206c55161babf68ba32bef5f988e6"
MAX_PX, MIN_PX, FACTOR = 1_605_632, 256 * 28 * 28, 28  # parallel_inference.py
TASKS = ["color", "spatial", "counting", "distance", "height", "reverse"]
DROPPED = ["pointing", "freespace", "function", "bbox", "landing", "single_image_caption", "multi_image_caption"]
HEAD = "<image>\nYou are a vision-language assistant.\n"
MC_HEAD = "Answer the following multiple choice question by selecting the correct option:"
PAIR = re.compile(r"^Which is (farther from me|closer to me|lower|higher), (.+?) or (.+?)\?$")
CATS = re.compile(r"\nChoose from the follow object categories: (.*?)\. Return only the object category\. "
                  r"Do not include extra text or reasoning\.$", re.S)
POINT = re.compile(r"\((\d+),\s*(\d+)\)")


def smart_resize(h: int, w: int) -> tuple[int, int]:
    """qwen-vl-utils smart_resize (factor 28, the processor's min / max pixels)."""
    hb, wb = max(FACTOR, round(h / FACTOR) * FACTOR), max(FACTOR, round(w / FACTOR) * FACTOR)
    if hb * wb > MAX_PX:
        beta = math.sqrt(h * w / MAX_PX)
        hb, wb = math.floor(h / beta / FACTOR) * FACTOR, math.floor(w / beta / FACTOR) * FACTOR
    elif hb * wb < MIN_PX:
        beta = math.sqrt(MIN_PX / (h * w))
        hb, wb = math.ceil(h * beta / FACTOR) * FACTOR, math.ceil(w * beta / FACTOR) * FACTOR
    return hb, wb


def ring(image: Image.Image, x: float, y: float, r: int = 12, width: int = 3) -> Image.Image:
    d = ImageDraw.Draw(image)
    d.ellipse([x - r - width, y - r - width, x + r + width, y + r + width], outline="white", width=1)
    d.ellipse([x - r - width + 1, y - r - width + 1, x + r + width - 1, y + r + width - 1], outline=(255, 0, 0),
              width=width)
    d.ellipse([x - r + 1, y - r + 1, x + r - 1, y + r - 1], outline="white", width=1)
    return image


def load_image(path: str) -> tuple[Image.Image, list[int], list[int]]:
    with Image.open(SRC / path) as im:
        w, h = im.size
        hb, wb = smart_resize(h, w)
        return resize(im.convert("RGB"), (wb, hb)), [w, h], [wb, hb]


def plain_image(path: str) -> dict:
    dest = OUT / "images" / Path(path).relative_to("UAVScenes/interval5_CAM_LIDAR").with_suffix(".jpg")
    with Image.open(SRC / path) as im:
        w, h = im.size
    hb, wb = smart_resize(h, w)
    if not dest.exists():
        atomic_save(load_image(path)[0], dest)
    return {"path": rel(dest), "orig_px": [w, h], "px": [wb, hb], "scale": [round(wb / w, 6), round(hb / h, 6)]}


def mc_question(q: str) -> tuple[str, list[str], list[str]]:
    body = q.removeprefix(HEAD).replace("\nReturn only the option in the format: \\boxed{option}. Do not include "
                                        "extra text or reasoning.", "")
    body = body.removeprefix(MC_HEAD).strip()
    if "Options:" in body:
        stem, _, opts = body.partition("Options:")
    else:  # color #16 has no "Options:" header (and a truncated question, kept as published)
        stem, opts = body[:body.index("\n\nA. ")], body[body.index("\n\nA. "):]
    parsed = re.findall(r"^([A-F])\. (.*)$", opts, re.M)
    return stem.strip(), [l for l, _ in parsed], [t.strip() for _, t in parsed]


def main() -> None:
    rows, stats = [], Counter()
    for task in TASKS:
        for k, x in enumerate(json.load(open(BENCH / f"{task}.json"))):
            path = x["image"][0] if isinstance(x["image"], list) else x["image"]
            q, gt = x["question"], x["ground_truth"]
            meta = {"task": task, "index": k, "source_image": path, "published_question": q, "published_gt": gt}
            state = ""
            if task in ("color", "spatial"):
                stem, letters, texts = mc_question(q)
                gold_letter = re.fullmatch(r"\\boxed\{([A-F])\}", gt).group(1)
                if len(set(texts)) != len(texts) or letters != list("ABCDEF")[:len(letters)]:
                    raise ValueError(f"{task} #{k}: options {letters} {texts}")
                options, gold = texts, texts[letters.index(gold_letter)]
                meta["gold_letter"] = gold_letter
            elif task == "counting":
                m = re.fullmatch(r"Given an image, answer this question: (.*?)\. Put your answer in this format: "
                                 r"<//boxed>\{number\}\.", q.removeprefix(HEAD), re.S)
                stem = m.group(1).strip()
                options = [str(i) for i in range(1, 11)]
                gold = re.fullmatch(r"<//boxed>\{(\d+)\}", gt).group(1)
            else:
                body = q.removeprefix(HEAD)
                cm = CATS.search(body)
                cats = cm.group(1).split(", ")
                stem = body[:cm.start()].strip()
                gold = gt.strip()
                if task in ("distance", "height") and (pm := PAIR.match(stem)):
                    options = [pm.group(2), pm.group(3)]
                    meta["form"] = "pair"
                else:
                    options = cats
                    if task != "reverse":
                        meta["form"] = "superlative"
                stats[f"{task}_{meta.get('form', 'point')}"] += 1
            if task == "reverse":
                pts = POINT.findall(stem)
                if len(pts) != 1:
                    raise ValueError(f"reverse #{k}: {stem}")
                image, orig, size = load_image(path)
                sx, sy = size[0] / orig[0], size[1] / orig[1]
                px, py = int(pts[0][0]), int(pts[0][1])
                xs, ys = round(px * sx), round(py * sy)
                dest = OUT / "images_marked" / f"reverse_{k:03d}.png"
                if not dest.exists():
                    atomic_save(ring(image, px * sx, py * sy), dest)
                stem = POINT.sub(f"({xs}, {ys}) [the centre of the red circle]", stem)
                state = (f"Coordinates (x, y) are pixel positions in the image as shown ({size[0]} x {size[1]} px); "
                         "the point is marked by a red circle.")
                info = {"path": rel(dest), "orig_px": orig, "px": size, "scale": [round(sx, 6), round(sy, 6)],
                        "point_native": [px, py], "point_shown": [xs, ys]}
            else:
                info = plain_image(path)
            if gold not in options:
                raise ValueError(f"{task} #{k}: gold {gold!r} not in {options}")
            meta["image"] = info
            row = {
                "id": f"SpatialSky:{task}:{k:03d}",
                "group_id": f"SpatialSky:{path}",
                "benchmark": "SpatialSky-Bench",
                "split": "test",
                "task_family": task,
                "skills": ["uav", "spatial"],
                "images": [info["path"]],
                "state": state,
                "questions": {"q": {"type": "choice", "instructions": stem, "criteria": {o: None for o in options}}},
                "answer": {"q": gold},
                "meta": meta,
            }
            validate_row(row)
            rows.append(row)
    n = write_jsonl(OUT / "spatialsky_closed.jsonl", rows)
    print(f"SpatialSky-Bench: {n} rows -> {OUT / 'spatialsky_closed.jsonl'}")
    print("task families:", dict(Counter(r["task_family"] for r in rows)), "| forms:", dict(stats))
    print("options per question:", dict(Counter((r["task_family"], len(r["questions"]["q"]["criteria"])) for r in rows)))
    print("images:", len({r["images"][0] for r in rows}), "sizes:",
          dict(Counter((tuple(r["meta"]["image"]["orig_px"]), tuple(r["meta"]["image"]["px"])) for r in rows)))
    print("dropped tasks (not closed-form):", {t: len(json.load(open(BENCH / f"{t}.json"))) for t in DROPPED})
    print("gold position per family:", json.dumps(answer_balance(rows)))
    show_samples(rows)


if __name__ == "__main__":
    main()
