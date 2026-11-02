"""C19 MME-RealWorld, Remote Sensing subset (3,738 five-option MCQ on 1,265 images of 0.7-86 MP) -> rows.

Source: HF `yifanzhang114/MME-RealWorld` @ 741cb8831ac86085bd54f678d13ca193e2334114 (Apache-2.0): the RS questions
of `MME_RealWorld.json` (raw/mme-realworld-rs/MME_RealWorld_RS.json) and the PNGs of the `remote_sensing.tar.gz`
parts (raw/mme-realworld-rs/remote_sensing/). Categories: position 1,257, color 1,255, count 1,226 (all
"Perception"). Every item has five options; E is always the reject option ("This image doesn't feature the
color / count / position.", 3 wordings) and is correct in only 14 items.

ADAPTED (resolution only); report it as "not the published protocol". Two input variants, built as in xlrs.py:
  v0 "downscale": the whole image resized (Lanczos) to <= 1,638,400 px (1,257 of 1,265 images are larger).
  v1 "tiles":     the v0 overview plus its four quadrants, each resized to <= 1,638,400 px (5 images); the state
                  names the layout (xlrs.py's wording).
Kept as published: all 3,738 items, the question text, the option texts and their order A-E, and option E
(meta.reject_option / reject_letter name it, so accuracy can be reported with and without it: drop the 14 E-gold
items and renormalise over A-D). Option letters "(A)" are stripped (the runners add letters). 24 items repeat an
option text (e.g. "(B) 5" and "(D) 5"; one has two empty options): there each option text gets its published
letter appended, "5 (B)", as xlrs.py does, so the keys stay distinct; meta.duplicate_options flags them.
Not kept: the published prompt template ("The choices are listed below: ... The best answer is:"); our runners use
their common lettered-option prompt.

    MODULE=bench.converters.mme_rw_rs ARGS="--workers 7" sbatch slurm/convert_cpu.sbatch   (~15 min CPU job)
    python -m bench.converters.mme_rw_rs                (resumable: images whose five tiles exist are not redone)
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import re
import time
from collections import Counter
from pathlib import Path

from PIL import Image

from bench.common import CONVERTED, MAX_PIXELS, MIN_PIXELS, RAW, save_png, validate_row, write_jsonl

Image.MAX_IMAGE_PIXELS = None
SRC = RAW / "mme-realworld-rs"
OUT = CONVERTED / "mme_rw_rs"
OPTION = re.compile(r"^\(([A-Z])\)\s*(.*)$", re.S)
REJECT = re.compile(r"^Th(is|e) image (doesn't|does not) feature the (color|count|number|position)\.$")
LAYOUT = ("Each source image is given as 5 images: the whole image, then its top-left, top-right, "
          "bottom-left and bottom-right quarters in more detail.")  # xlrs.py's v1 wording


# ---- overview + quadrant tiles (shared with rshr.py) -------------------------------------------------------------

def fit_size(w: int, h: int) -> tuple[int, int]:
    """Size after xlrs.fit: Lanczos downscale to <= MAX_PIXELS keeping the aspect ratio; never upscaled."""
    if w * h <= MAX_PIXELS:
        return w, h
    s = math.sqrt(MAX_PIXELS / (w * h))
    return max(1, int(w * s)), max(1, int(h * s))


def fit(image: Image.Image) -> Image.Image:
    """xlrs.fit; images under MIN_PIXELS are upscaled (bicubic) to it (none are expected for these UHR sets)."""
    w, h = fit_size(*image.size)
    if (w, h) != image.size:
        return image.resize((w, h), Image.Resampling.LANCZOS)
    if w * h < MIN_PIXELS:
        s = math.sqrt(MIN_PIXELS / (w * h))
        return image.resize((math.ceil(w * s), math.ceil(h * s)), Image.Resampling.BICUBIC)
    return image


def quarter_boxes(w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Top-left, top-right, bottom-left, bottom-right (xlrs.py's order and split)."""
    return [(0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2), (0, h // 2, w // 2, h), (w // 2, h // 2, w, h)]


def tile_sizes(w: int, h: int) -> dict:
    """Sizes of the five tiles of a w x h source, without decoding it."""
    quads = [fit_size(x1 - x0, y1 - y0) for x0, y0, x1, y1 in quarter_boxes(w, h)]
    over = fit_size(w, h)
    return {"overview_px": list(over), "quarter_px": [list(q) for q in quads],
            "resize_factor": round(over[0] / w, 5), "quarter_resize_factor": round(quads[0][0] / (w // 2), 5)}


def tile_files(out: Path, key: str) -> list[Path]:
    """The overview, then the four quadrants."""
    return [out / "images" / f"{key}_full.png"] + [out / "images" / f"{key}_q{i}.png" for i in range(4)]


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in ("I;16", "I;16B", "I;16L", "I"):  # 16-bit grey: stretch to 8 bit
        lo, hi = image.getextrema()
        image = image.point(lambda v: (v - lo) * 255.0 / max(1, hi - lo)).convert("L")
    return image.convert("RGB")


def save_atomic(image: Image.Image, path: Path) -> None:
    """save_png to a temporary name, then rename: a killed job never leaves a truncated tile behind."""
    tmp = path.with_name(path.name + ".part")
    save_png(image, tmp)
    tmp.replace(path)


def rendered(files: list[Path]) -> bool:
    return all(f.exists() for f in files)


def render_tiles(image: Image.Image, files: list[Path]) -> dict:
    """Write overview + quadrants of one decoded image; returns its size info. Decodes nothing else."""
    w, h = image.size
    info = {"orig_px": [w, h], "mode": image.mode, "format": image.format}
    image = to_rgb(image)
    over = fit(image)
    save_atomic(over, files[0])
    sizes = [list(over.size)]
    for f, box in zip(files[1:], quarter_boxes(w, h)):
        tile = fit(image.crop(box))
        save_atomic(tile, f)
        sizes.append(list(tile.size))
    expect = tile_sizes(w, h)
    if sizes != [expect["overview_px"], *expect["quarter_px"]]:
        info["upscaled"] = sizes  # only when a tile fell under MIN_PIXELS
    return info


def rel(p: Path) -> str:
    return str(p.relative_to(CONVERTED))


# ---- MME-RealWorld-RS --------------------------------------------------------------------------------------------

def image_key(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", path.removeprefix("remote_sensing/").rsplit(".", 1)[0])


def _render_one(task: tuple[str, str]) -> tuple[str, dict, float]:
    path, key = task
    t0 = time.time()
    with Image.open(SRC / path) as im:
        im.load()
        info = render_tiles(im, tile_files(OUT, key))
    return path, info, time.time() - t0


def render_images(paths: list[str], workers: int) -> dict[str, dict]:
    """Render every image whose five tiles are not all on disk yet; returns {path: info} for all images."""
    infos, todo = {}, []
    for p in paths:
        if rendered(tile_files(OUT, image_key(p))):
            with Image.open(SRC / p) as im:  # header only
                infos[p] = {"orig_px": list(im.size), "mode": im.mode, "format": im.format}
        else:
            todo.append((p, image_key(p)))
    print(f"images: {len(paths)} referenced, {len(infos)} already rendered, {len(todo)} to render", flush=True)
    t0 = time.time()
    if not todo:
        return infos
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=25) as pool:
        for i, (p, info, dt) in enumerate(pool.imap_unordered(_render_one, todo), 1):
            infos[p] = info
            print(f"[{i}/{len(todo)}] {p} {info['orig_px']} {dt:.1f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    return infos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    items = json.load(open(SRC / "MME_RealWorld_RS.json"))
    paths = sorted({x["Image"] for x in items})
    keys = {image_key(p) for p in paths}
    assert len(keys) == len(paths), "image keys collide"
    infos = render_images(paths, args.workers)

    rows_v0, rows_v1, dups = [], [], 0
    for x in items:
        parsed = [OPTION.match(o.strip()) for o in x["Answer choices"]]
        if not all(parsed):
            raise ValueError(f"{x['Question_id']}: unparsed options {x['Answer choices']}")
        letters = [m.group(1) for m in parsed]
        texts = [m.group(2).strip() for m in parsed]
        if letters != list("ABCDE") or not REJECT.match(texts[4]):
            raise ValueError(f"{x['Question_id']}: unexpected options {x['Answer choices']}")
        dup = len(set(texts)) != len(texts) or not all(texts)
        if dup:  # keep every option distinct, as xlrs.py does
            texts = [f"{t} ({l})".strip() for l, t in zip(letters, texts)]
            dups += 1
        gold = texts[letters.index(x["Ground truth"])]
        qid = x["Question_id"].removeprefix("perception/remote_sensing/")
        key = image_key(x["Image"])
        files = tile_files(OUT, key)
        info = infos[x["Image"]]
        base = {
            "group_id": f"MMERW-RS:{key}",
            "benchmark": "MME-RealWorld-RS",
            "split": "test",
            "task_family": x["Category"],
            "skills": ["perception", x["Category"]],
            "questions": {"q": {"type": "choice", "instructions": x["Text"].strip(),
                                "criteria": {t: None for t in texts}}},
            "answer": {"q": gold},
            "meta": {"question_id": x["Question_id"], "category": x["Category"], "task": x["Task"],
                     "source": x["Image"], "orig_px": info["orig_px"], **tile_sizes(*info["orig_px"]),
                     "gold_letter": x["Ground truth"], "reject_option": texts[4], "reject_letter": "E",
                     "duplicate_options": dup, "published_options": x["Answer choices"]},
        }
        if "upscaled" in info:
            base["meta"]["upscaled_px"] = info["upscaled"]
        v0 = {**base, "id": f"MMERW-RS:{qid}:v0", "images": [rel(files[0])], "state": "",
              "meta": {**base["meta"], "variant": "downscale"}}
        v1 = {**base, "id": f"MMERW-RS:{qid}:v1", "images": [rel(f) for f in files], "state": LAYOUT,
              "meta": {**base["meta"], "variant": "overview+quadrants"}}
        for v in (v0, v1):
            validate_row(v)
        rows_v0.append(v0)
        rows_v1.append(v1)
    write_jsonl(OUT / "mme_rw_rs_v0_downscale.jsonl", rows_v0)
    write_jsonl(OUT / "mme_rw_rs_v1_tiles.jsonl", rows_v1)

    resized = sum(info["orig_px"][0] * info["orig_px"][1] > MAX_PIXELS for info in infos.values())
    print("MME-RealWorld-RS:", len(rows_v0), "rows (1 choice question each) per variant; dropped 0")
    print("  task_family:", dict(Counter(r["task_family"] for r in rows_v0)))
    print("  gold letter:", dict(sorted(Counter(r["meta"]["gold_letter"] for r in rows_v0).items())),
          "| items with repeated option texts (letter appended):", dups)
    print(f"  images: {len(infos)} sources -> {len(infos) * 5} PNGs; {resized} downscaled for v0, "
          f"{sum('upscaled' in i for i in infos.values())} with an upscaled tile; source modes",
          dict(Counter(i["mode"] for i in infos.values())))
    px = sorted(i["orig_px"][0] * i["orig_px"][1] for i in infos.values())
    print(f"  source px: min {px[0]:,} median {px[len(px) // 2]:,} max {px[-1]:,}")

    from bench.specialists.prompting import render  # the text every model sees
    for rows, rid in ((rows_v0, "MMERW-RS:color/0001:v0"), (rows_v1, "MMERW-RS:count/0444:v1"),
                      (rows_v0, "MMERW-RS:position/1220:v0")):  # plain, repeated options (v1), E is gold
        row = next(r for r in rows if r["id"] == rid)
        print(f"\n--- {rid} ({row['task_family']}), gold {row['meta']['gold_letter']}; images: {row['images']}")
        print(render(row, "q").text)


if __name__ == "__main__":
    main()
