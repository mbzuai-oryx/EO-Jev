"""C22 Geo3DVQA (WACV'26, arXiv 2512.07276) test short-answer QAs -> rows, with the queried regions / points drawn.

Source: HF mt129/Geo3DVQA @ 52a912499d8c2322f625e70c133c052356b9136d, qas/svf_short-qas_answers_test.jsonl
(10,232 QAs, phase "mixed_hr0.3", over 333 GeoNRW test tiles from 9 cities). Imagery: GeoNRW RGB orthophotos
(1 m, 1000 x 1000 px; the jp2 is RGB + NIR, NIR dropped), streamed from HF torchgeo/geonrw @ e9a2500b into
raw/geonrw-geo3dvqa-test/<city>/<x>_<y>_rgb.jp2. Tiles are within MIN / MAX_PIXELS: no resize (factor 1.0).

Conversion per category (published question text kept; the paper's metric in brackets):
  sun_exposure 999, highest_region 666, urban_density 585 (4 regions), regional_svf_variability 662 (3 regions),
  openness_assessment 999 (4 regions)                    -> `choice` over "Region A".. as published [accuracy]
  hard_ranking 999 (3 regions)                            -> 6-way `choice` over the published permutations
                                                             "Region A, Region B, Region C", ... [exact order]
  visibility_range 999, sky_visibility 999 (4 points)     -> `choice`; option "Point (x%, y%)" / "(x%, y%)"
                                                             becomes "Point A (x%, y%)" so it names its marker
  hard_pixel 999 (mean SVF of one region, 1 decimal)      -> `score`, 10 levels "0.1 or less", "0.2", .., "1.0"
                                                             [paper: |pred - gt| <= 0.05 on the 1-decimal answer,
                                                             i.e. exact level; gt is never below 0.4]
  height_average 666 (mean height of one region, 10 m)    -> `score`, 10 levels (HEIGHT_LEVELS)          [paper:
                                                             exact for 0 m, +-10 m up to 30 m, +-30% above]
  landcover_type 993 (all classes present, 11 options)    -> one `noul` per class, "_multi" [Jaccard >= 0.8]
  land_use 666 (top-3 classes, 11 options)                -> one `noul` per class, "_multi" [exact set]
Overlays: every region / point the question gives as text percentages is drawn on the image (after any resize):
boxes or dots in a fixed colour per label (A red, B cyan, C yellow, D magenta) with a filled letter tag; a
single queried region (hard_pixel, height_average) is a red box. The text coordinates stay in the question, and
one sentence is appended saying what is drawn. Removed from the text: the inline "Please choose from:" option
list (the runner prints the options) and the answer-format sentences ("Answer format: X.X", "Provide the exact
result rounded to 1 decimal place.", "Please answer in 10-meter increments. Answer format: X m", "Format your
answer as: Region A, Region B, Region C", "Answer format: type1, type2, type3"). hard_ranking's published
questions contain unfilled template slots ("{region1}, {region2}, {region3}"): kept verbatim.

Bins and the paper's tolerance: meta["paper_ok_levels"] lists the level indices whose representative value
(meta["level_values"]) passes the paper's numeric tolerance for that item's gold value, so the paper metric is
recomputed from the predicted level; exact-level accuracy is the stricter primary score. Height levels and
representatives: 10 m or less (10), 20, 30, 40, 50-60 (55), 70-80 (75), 90-110 (100), 120-150 (135), 160-210 (185),
220 m or more (280); every gold value's own level passes the paper tolerance.

ADAPTED (not as-published): numeric answers binned (score), multi-label as per-class yes/no, regions / points drawn
on the image, inline option lists and answer-format lines removed. All 10,232 test QAs are kept.
Images are JPEG q95 4:4:4 (one clean tile per tile for the class questions, one overlay image per drawn question).
Resumable: images already written are skipped.

    python -m bench.converters.geo3dvqa
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from multiprocessing import Pool

from PIL import Image, ImageDraw, ImageFont

from bench.common import CONVERTED, DATASETS, MAX_PIXELS, MIN_PIXELS, RAW, validate_row, write_jsonl

REV = "52a912499d8c2322f625e70c133c052356b9136d"
QAS = DATASETS / "hf_hub" / "datasets--mt129--Geo3DVQA" / "snapshots" / REV / "qas" / "svf_short-qas_answers_test.jsonl"
TILES = RAW / "geonrw-geo3dvqa-test"
OUT = CONVERTED / "geo3dvqa"
WORKERS = 8

COLORS = {"A": (255, 0, 0), "B": (0, 255, 255), "C": (255, 255, 0), "D": (255, 0, 255), "R": (255, 0, 0)}
LABELS = ["others", "forest", "water", "agricultural", "residential", "grassland", "railways", "roads", "commercial",
          "bare_soil", "buildings"]
SVF_LEVELS = ["0.1 or less"] + [f"{v / 10:.1f}" for v in range(2, 11)]
SVF_VALUES = [0.1] + [v / 10 for v in range(2, 11)]
HEIGHT_LEVELS = ["10 m or less", "20 m", "30 m", "40 m", "50 to 60 m", "70 to 80 m", "90 to 110 m", "120 to 150 m",
                 "160 to 210 m", "220 m or more"]
HEIGHT_BOUNDS = [(0, 10), (20, 20), (30, 30), (40, 40), (50, 60), (70, 80), (90, 110), (120, 150), (160, 210), (220, 10**9)]
HEIGHT_VALUES = [10, 20, 30, 40, 55, 75, 100, 135, 185, 280]

BOX = re.compile(r"^(?:- )?(?:Region )?([A-D]): \[xmin=(\d+(?:\.\d+)?)%, ymin=(\d+(?:\.\d+)?)%, "
                 r"xmax=(\d+(?:\.\d+)?)%, ymax=(\d+(?:\.\d+)?)%\]$", re.M)
OPEN_OPTION = re.compile(r"^Region ([A-D]): \[(\d+(?:\.\d+)?)%, (\d+(?:\.\d+)?)%, (\d+(?:\.\d+)?)%, (\d+(?:\.\d+)?)%\]$")
POINT_OPTION = re.compile(r"^(?:Point )?\((\d+(?:\.\d+)?)%, (\d+(?:\.\d+)?)%\)$")
REGION_OPTION = re.compile(r"^Region [A-D]$")
ONE_BOX = re.compile(r"\[(\d+(?:\.\d+)?)%, (\d+(?:\.\d+)?)%, (\d+(?:\.\d+)?)%, (\d+(?:\.\d+)?)%\]")
STRIP = [  # answer-format sentences (the runner reads the answer as a letter / level / yes-no)
    " Provide the exact result rounded to 1 decimal place.", " Answer format: X.X",
    "Please answer in 10-meter increments. Answer format: X m", "Format your answer as: Region A, Region B, Region C",
    "\nAnswer format: type1, type2, type3",
]
REGION_CATS = {"sun_exposure", "highest_region", "urban_density", "regional_svf_variability", "hard_ranking"}
SKILLS = {"sun_exposure": "svf", "hard_pixel": "svf", "hard_ranking": "svf", "regional_svf_variability": "svf",
          "sky_visibility": "svf", "visibility_range": "3d", "openness_assessment": "3d", "urban_density": "3d",
          "height_average": "height", "highest_region": "height", "land_use": "landcover", "landcover_type": "landcover"}


def split_options(text: str) -> tuple[str, list[str]]:
    """Remove the 'Please choose from:' block; return (text without it, its option lines)."""
    lines = text.split("\n")
    if "Please choose from:" not in lines:
        return text, []
    i = lines.index("Please choose from:")
    j = i + 1
    while j < len(lines) and (REGION_OPTION.match(lines[j]) or OPEN_OPTION.match(lines[j]) or POINT_OPTION.match(lines[j])):
        j += 1
    return "\n".join(lines[:i] + lines[j:]), lines[i + 1:j]


def clean(text: str) -> str:
    for s in STRIP:
        text = text.replace(s, "")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if re.search(r"Answer format|Format your answer|Please choose from", text):
        raise ValueError(f"answer-format text left: {text!r}")
    return text


def height_level(metres: int) -> int:
    return next(i for i, (lo, hi) in enumerate(HEIGHT_BOUNDS) if lo <= metres <= hi)


def height_ok(rep: float, gold: int) -> bool:
    if gold == 0:
        return rep == 0
    return abs(rep - gold) <= (10 if gold <= 30 else 0.3 * gold)


def convert(r: dict) -> tuple[dict, dict | None]:
    """One QA -> (row, overlay spec or None)."""
    cat, text, qid = r["category"], r["text"], r["question_id"]
    tile = r["image"].removesuffix("_rgb.jp2")  # <city>/<x>_<y>
    body, option_lines = split_options(text)
    overlay, note, meta = None, "", {}
    questions: dict
    answer: dict
    if cat in REGION_CATS or cat == "openness_assessment":
        if cat == "openness_assessment":
            parsed = [OPEN_OPTION.match(o) for o in option_lines]
            boxes = {m.group(1): [float(m.group(k)) for k in range(2, 6)] for m in parsed}
            options = option_lines
        else:
            boxes = {m.group(1): [float(m.group(k)) for k in range(2, 6)] for m in BOX.finditer(text)}
            options = r.get("choices") or option_lines
            if r.get("choices") and option_lines and option_lines != r["choices"]:
                raise ValueError(f"{qid}: choices {r['choices']} != listed {option_lines}")
        labels = sorted(boxes)
        if not options or labels != ["A", "B", "C", "D"][: len(labels)] or len(labels) < 3:
            raise ValueError(f"{qid}: regions {labels} / options {options}")
        overlay = {"boxes": boxes}
        note = f"Regions {', '.join(labels[:-1])} and {labels[-1]} are drawn on the image as boxes labelled with their letters."
        questions = {"q": {"type": "choice", "instructions": clean(body) + "\n" + note, "criteria": {o: None for o in options}}}
        # openness_assessment options carry their box ("Region B: [0%, 28%, 21%, 49%]"); its answer is "Region B".
        gold = [o for o in options if o == r["answer"] or o.startswith(r["answer"] + ":")]
        if len(gold) != 1:
            raise ValueError(f"{qid}: answer {r['answer']!r} not in {options}")
        answer = {"q": gold[0]}
        meta["regions_pct"] = boxes
    elif cat in ("visibility_range", "sky_visibility"):
        points = [[float(v) for v in POINT_OPTION.match(o).groups()] for o in option_lines]
        labels = "ABCD"[: len(points)]
        if len(points) != 4 or r["answer"] not in option_lines or len(set(option_lines)) != 4:
            raise ValueError(f"{qid}: points {option_lines} / answer {r['answer']}")
        options = [f"Point {l} {o.removeprefix('Point ')}" for l, o in zip(labels, option_lines)]  # source digits kept
        overlay = {"points": dict(zip(labels, points))}
        note = "The candidate points are marked on the image as dots labelled A, B, C and D (in the order of the options)."
        questions = {"q": {"type": "choice", "instructions": clean(body) + "\n" + note, "criteria": {o: None for o in options}}}
        answer = {"q": options[option_lines.index(r["answer"])]}
        meta["points_pct"] = dict(zip(labels, points))
        meta["published_options"] = option_lines
    elif cat in ("hard_pixel", "height_average"):
        box = [float(v) for v in ONE_BOX.search(text).groups()]
        overlay = {"boxes": {"R": box}}
        note = "The region is outlined in red on the image."
        if cat == "hard_pixel":
            gold = float(r["answer"])
            levels, values = SVF_LEVELS, SVF_VALUES
            level = SVF_LEVELS.index(f"{gold:.1f}") if gold >= 0.2 else 0
            ok = [i for i, v in enumerate(values) if abs(v - gold) <= 0.05 + 1e-9]
            meta["paper_tolerance"] = "|pred - gt| <= 0.05 (answers to 1 decimal)"
        else:
            gold = int(r["answer"].removesuffix(" m"))
            levels, values = HEIGHT_LEVELS, HEIGHT_VALUES
            level = height_level(gold)
            ok = [i for i, v in enumerate(values) if height_ok(v, gold)]
            meta["paper_tolerance"] = "exact for 0 m; +-10 m for gt <= 30 m; +-30% above"
        if level not in ok:
            raise ValueError(f"{qid}: gold level {level} fails the paper tolerance for {gold}")
        questions = {"q": {"type": "score", "instructions": clean(body) + "\n" + note, "criteria": levels}}
        answer = {"q": level}
        meta.update(gold_value=gold, level_values=values, paper_ok_levels=ok, region_pct=box)
    elif cat in ("land_use", "landcover_type"):
        golds = [a.strip() for a in r["answer"].split(",")]
        if not set(golds) <= set(LABELS) or ", ".join(LABELS) not in text:
            raise ValueError(f"{qid}: labels {golds}")
        stem = clean(body)
        questions = {lab: {"type": "noul", "instructions": f"{stem}\nIs \"{lab}\" among the correct answers?"} for lab in LABELS}
        answer = {lab: lab in golds for lab in LABELS}
        meta["gold_set"] = golds
        meta["paper_metric"] = "exact set match" if cat == "land_use" else "Jaccard(pred, gold) >= 0.8"
    else:
        raise ValueError(f"{qid}: unknown category {cat}")
    stem_path = f"images/{tile}/q{qid}.jpg" if overlay else f"images/{tile}/tile.jpg"
    multi = cat in ("land_use", "landcover_type")
    row = {
        "id": f"Geo3DVQA:{qid}",
        "group_id": f"Geo3DVQA:{tile}",
        "benchmark": "Geo3DVQA",
        "split": "test",
        "task_family": cat + ("_multi" if multi else ""),
        "skills": [SKILLS[cat]],
        "images": [f"geo3dvqa/{stem_path}"],
        "state": "",
        "questions": questions,
        "answer": answer,
        "meta": {"question_id": qid, "category": cat, "phase": r["phase"], "source_image": r["image"],
                 "published_text": text, "published_answer": r["answer"], "published_choices": r.get("choices"),
                 "overlay": bool(overlay), "orig_px": [1000, 1000], "resize": 1.0, **meta},
    }
    validate_row(row)
    return row, ({"tile": tile, "dest": stem_path, **overlay} if overlay else None)


def font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def tag(draw: ImageDraw.ImageDraw, xy: tuple[float, float], label: str, color, size: int, w: int, h: int,
        placed: list[tuple[int, int, int, int]]) -> None:
    """A filled letter tag with its top-left at xy, kept inside the image and clear of the tags already placed."""
    f = font(size)
    l, t, r, b = draw.textbbox((0, 0), label, font=f)
    tw, th = r - l + 8, b - t + 8
    x = min(max(0, xy[0]), w - tw)
    y = min(max(0, xy[1]), h - th)
    for _ in range(8):  # step right (or left at the edge) past any overlapping tag
        hit = [p for p in placed if x < p[2] + 2 and p[0] < x + tw + 2 and y < p[3] + 2 and p[1] < y + th + 2]
        if not hit:
            break
        x = hit[0][2] + 3 if hit[0][2] + 3 + tw <= w else hit[0][0] - tw - 3
    placed.append((x, y, x + tw, y + th))
    draw.rectangle((x, y, x + tw, y + th), fill=color, outline=(0, 0, 0), width=2)
    draw.text((x + 4 - l, y + 4 - t), label, fill=(0, 0, 0), font=f)


def render(image: Image.Image, spec: dict) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    size = max(16, round(0.028 * w))
    px = lambda v, n: min(n - 1, max(0, round(v / 100 * n)))  # noqa: E731
    for label, (x0, y0, x1, y1) in spec.get("boxes", {}).items():
        box = (px(x0, w), px(y0, h), px(x1, w), px(y1, h))
        draw.rectangle(box, outline=(0, 0, 0), width=5)
        draw.rectangle((box[0] + 1, box[1] + 1, box[2] - 1, box[3] - 1), outline=COLORS[label], width=3)
    placed: list[tuple[int, int, int, int]] = []
    for label, (x0, y0, x1, y1) in spec.get("boxes", {}).items():  # tags last, so boxes never cover them
        if label != "R":
            top = px(y0, h) - size - 10
            tag(draw, (px(x0, w), top if top >= 0 else px(y0, h) + 3), label, COLORS[label], size, w, h, placed)
    rad = max(6, round(0.009 * w))
    for label, (x, y) in spec.get("points", {}).items():
        cx, cy = px(x, w), px(y, h)
        draw.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=COLORS[label], outline=(0, 0, 0), width=2)
        tag(draw, (cx + rad + 2, cy - rad - size - 6), label, COLORS[label], size, w, h, placed)
    return out


def save_jpeg(image: Image.Image, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.jpg")
    image.save(tmp, format="JPEG", quality=95, subsampling=0)
    tmp.replace(path)  # atomic: an interrupted run never leaves a half-written image


def render_tile(job: tuple[str, list[dict]]) -> Counter:
    tile, specs = job
    done = Counter()
    base = OUT / "images" / tile / "tile.jpg"
    todo = [s for s in specs if not (OUT / s["dest"]).exists()]
    if base.exists() and not todo:
        done["skipped"] += len(specs) + 1
        return done
    with Image.open(TILES / f"{tile}_rgb.jp2") as im:
        image = im.convert("RGB")  # RGB + NIR -> RGB
    if image.size != (1000, 1000) or not MIN_PIXELS <= image.width * image.height <= MAX_PIXELS:
        raise ValueError(f"{tile}: unexpected size {image.size}")
    if not base.exists():
        save_jpeg(image, base)
        done["written"] += 1
    for s in todo:
        save_jpeg(render(image, s), OUT / s["dest"])
        done["written"] += 1
    done["skipped"] += len(specs) - len(todo)
    return done


def main() -> None:
    records = [json.loads(line) for line in open(QAS)]
    rows, specs = [], defaultdict(list)
    for r in records:
        row, spec = convert(r)
        rows.append(row)
        specs[r["image"].removesuffix("_rgb.jp2")]  # every tile gets its clean image
        if spec:
            specs[spec["tile"]].append(spec)
    with Pool(WORKERS) as pool:
        counts = sum(pool.imap_unordered(render_tile, sorted(specs.items()), chunksize=4), Counter())
    n = write_jsonl(OUT / "geo3dvqa.jsonl", rows)
    print("Geo3DVQA:", n, "rows ->", OUT / "geo3dvqa.jsonl")
    print("  questions by type:", dict(Counter(q["type"] for r in rows for q in r["questions"].values())))
    print("  task_family:", json.dumps(dict(sorted(Counter(r["task_family"] for r in rows).items()))))
    print("  images:", len(specs), "tiles +", sum(len(v) for v in specs.values()), "overlays;", dict(counts),
          "| resized: 0 (all 1000x1000) | dropped: 0")
    for cat in sorted({r["meta"]["category"] for r in rows}):
        rs = [r for r in rows if r["meta"]["category"] == cat]
        if rs[0]["questions"].get("q", {}).get("type") == "choice":
            dist = Counter(list(r["questions"]["q"]["criteria"]).index(r["answer"]["q"]) for r in rs)
            print(f"  {cat}: gold option index", dict(sorted(dist.items())))
        elif rs[0]["questions"].get("q", {}).get("type") == "score":
            print(f"  {cat}: gold level", dict(sorted(Counter(r["answer"]["q"] for r in rs).items())))
        else:
            print(f"  {cat}: positives per class", {k: sum(r["answer"][k] for r in rs) for k in LABELS})


if __name__ == "__main__":
    main()
