"""Shared helpers for the eval-v1.0 "perception" converters (meter_ml, rs_neg, geommbench, terrascope_bench,
dior_rsvg_choice, vrsbench_vg_choice, det_count): image fitting to d3's pixel range, candidate-box overlays, and the
end-of-run summary / sample printout required by the converter spec. Not a converter itself.
"""

from __future__ import annotations

import io
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from bench.common import CONVERTED, MAX_PIXELS, MIN_PIXELS, NATIVE_FORMATS, save_png

Image.MAX_IMAGE_PIXELS = None
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ---------------------------------------------------------------------------------------------------- images
def fit(image: Image.Image) -> tuple[Image.Image, float]:
    """Resize into [MIN_PIXELS, MAX_PIXELS] keeping the aspect ratio: Lanczos down, bicubic up.
    Returns (image, factor) where factor = new side / old side (1.0 when unchanged)."""
    w, h = image.size
    px = w * h
    if px > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / px)
        size = (max(1, math.floor(w * s)), max(1, math.floor(h * s)))
        return image.resize(size, Image.Resampling.LANCZOS), size[0] / w
    if px < MIN_PIXELS:
        s = math.sqrt(MIN_PIXELS / px)
        size = (math.ceil(w * s), math.ceil(h * s))
        return image.resize(size, Image.Resampling.BICUBIC), size[0] / w
    return image, 1.0


def store_fitted(data: bytes, dest_stem: Path) -> dict:
    """Write an image under converted/: native PNG / JPEG / WebP bytes unchanged when they are RGB / L and already
    within the pixel limits, else a resized (see fit) RGB PNG. Resumable: an existing output is reused.
    Returns {"path": rel, "orig_px": [w, h], "px": [w, h], "resize_factor": f, "reencoded": bool}."""
    for ext in (".png", ".jpg", ".webp"):
        done = dest_stem.with_suffix(ext)
        if done.exists():
            with Image.open(io.BytesIO(data)) as src, Image.open(done) as out:
                return {"path": str(done.relative_to(CONVERTED)), "orig_px": list(src.size), "px": list(out.size),
                        "resize_factor": round(out.size[0] / src.size[0], 6), "reencoded": ext != NATIVE_FORMATS.get(src.format or "")}
    image = Image.open(io.BytesIO(data))
    orig = list(image.size)
    ext = NATIVE_FORMATS.get(image.format or "")
    if ext and image.mode in ("RGB", "L") and MIN_PIXELS <= orig[0] * orig[1] <= MAX_PIXELS:
        path = dest_stem.with_suffix(ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"path": str(path.relative_to(CONVERTED)), "orig_px": orig, "px": orig, "resize_factor": 1.0,
                "reencoded": False}
    image = image.convert("RGB")
    fitted, factor = fit(image)
    path = save_png(fitted, dest_stem.with_suffix(".png"))
    return {"path": str(path.relative_to(CONVERTED)), "orig_px": orig, "px": list(fitted.size),
            "resize_factor": round(factor, 6), "reencoded": True}


# ---------------------------------------------------------------------------------------------------- box choice
# High-contrast, mutually distinct colours for candidate boxes A, B, C, D (red, yellow, cyan, magenta); every box
# and label also gets a black outline so it stays visible on bright and dark ground.
BOX_COLOURS = [("red", (255, 32, 32)), ("yellow", (255, 230, 0)), ("cyan", (0, 230, 255)), ("magenta", (255, 0, 255))]


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def draw_candidates(image: Image.Image, boxes: list[list[float]]) -> Image.Image:
    """Draw boxes (xyxy, pixels of `image`) in BOX_COLOURS order with their letter in a filled tag OUTSIDE the box
    (above its top-left corner; below the box when there is no room above; inside the image either way)."""
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    w, h = image.size
    side = min(w, h)
    lw = max(2, round(side / 300))          # 3 px on 800 px images, 2 px on 512 px
    fsize = max(14, round(side / 32))       # 25 px on 800, 16 px on 512
    font = _font(fsize)
    for i, box in enumerate(boxes):
        colour = BOX_COLOURS[i][1]
        x0, y0, x1, y1 = [round(v) for v in box]
        x1, y1 = max(x1, x0 + 2), max(y1, y0 + 2)
        draw.rectangle([x0 - lw, y0 - lw, x1 + lw, y1 + lw], outline=(0, 0, 0), width=1)
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=lw)
        letter = LETTERS[i]
        tb = draw.textbbox((0, 0), letter, font=font)
        tw, th = tb[2] - tb[0] + 6, tb[3] - tb[1] + 6
        ty = y0 - lw - th - 1 if y0 - lw - th - 1 >= 0 else min(h - th, y1 + lw + 1)
        tx = min(max(0, x0 - lw), w - tw)
        draw.rectangle([tx, ty, tx + tw, ty + th], fill=colour, outline=(0, 0, 0), width=1)
        draw.text((tx + 3 - tb[0], ty + 3 - tb[1]), letter, fill=(0, 0, 0), font=font)
    return image


def pick_candidates(gold: list[float], pools: list[tuple[str, list[list[float]]]], size: tuple[int, int],
                    rng: random.Random, k: int = 4, max_iou: float = 0.3, min_side: float = 4.0
                    ) -> list[tuple[str, list[float]]]:
    """Gold box + (k-1) distractors. `pools` are (source name, boxes) in priority order; each pool is shuffled with
    `rng` and taken greedily. Every chosen box has IoU < max_iou with the gold box and with every other chosen box.
    When the pools run out, similar-size proposals are drawn: the gold box's width / height scaled by 0.7-1.4 at a
    random position inside the image (source "proposal")."""
    chosen: list[tuple[str, list[float]]] = []

    def ok(b: list[float]) -> bool:
        if b[2] - b[0] < min_side or b[3] - b[1] < min_side:
            return False
        return iou(b, gold) < max_iou and all(iou(b, c) < max_iou for _, c in chosen)

    for name, boxes in pools:
        boxes = list(boxes)
        rng.shuffle(boxes)
        for b in boxes:
            if len(chosen) == k - 1:
                break
            if ok(b):
                chosen.append((name, b))
    w, h = size
    gw, gh = gold[2] - gold[0], gold[3] - gold[1]
    tries = 0
    while len(chosen) < k - 1:
        tries += 1
        if tries > 5000:
            raise RuntimeError(f"cannot place {k - 1} distractors around {gold} in {size}")
        s = 0.7 + 0.7 * rng.random() if tries < 4000 else 0.3 + 0.4 * rng.random()
        bw = min(w - 1, max(min_side, gw * s * (0.85 + 0.3 * rng.random())))
        bh = min(h - 1, max(min_side, gh * s * (0.85 + 0.3 * rng.random())))
        x0 = rng.random() * (w - bw)
        y0 = rng.random() * (h - bh)
        b = [round(x0, 1), round(y0, 1), round(x0 + bw, 1), round(y0 + bh, 1)]
        if ok(b):
            chosen.append(("proposal", b))
    return [("gold", gold)] + chosen


def stratified(items: list[dict], key: str, n: int, rng: random.Random) -> list[dict]:
    """Proportional allocation by `key` (largest remainder), then a seeded sample within each stratum."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        groups[it[key]].append(it)
    total = len(items)
    quota = {g: n * len(v) / total for g, v in groups.items()}
    alloc = {g: int(q) for g, q in quota.items()}
    for g in sorted(groups, key=lambda g: (-(quota[g] - alloc[g]), g))[: n - sum(alloc.values())]:
        alloc[g] += 1
    picked = []
    for g in sorted(groups):
        picked += rng.sample(groups[g], alloc[g])
    return picked


# ---------------------------------------------------------------------------------------------------- reporting
def render_text(row: dict, qid: str) -> str:
    """The question as a lettered-option model would see it (state, instructions, A. B. ... options)."""
    q = row["questions"][qid]
    head = row.get("state", "").strip()
    parts = [head] if head else []
    parts.append(q["instructions"].strip())
    if q["type"] == "noul":
        parts.append("(yes / no)")
    else:
        opts = list(q["criteria"])
        parts += [f"{LETTERS[i] if i < 26 else '#' + str(i)}. {o}" for i, o in enumerate(opts)]
    return "\n".join(parts)


def sample_text(rows: list[dict], n: int = 3, seed: int = 0) -> str:
    out = []
    for row in random.Random(seed).sample(rows, min(n, len(rows))):
        for qid in list(row["questions"])[:2]:
            q = row["questions"][qid]
            ans = row["answer"][qid]
            if q["type"] == "choice":
                ans = f"{LETTERS[list(q['criteria']).index(ans)] if list(q['criteria']).index(ans) < 26 else ''} ({ans})"
            out.append(f"--- {row['id']} / {qid}  [{row['task_family']}]\nimages: {row['images']}\n"
                       f"{render_text(row, qid)}\n=> answer: {ans}")
    return "\n".join(out)


def summary(name: str, rows: list[dict]) -> str:
    qtypes, fams, letters = Counter(), Counter(), Counter()
    for r in rows:
        for qid, q in r["questions"].items():
            qtypes[q["type"]] += 1
            fams[r["task_family"]] += 1
            if q["type"] == "choice":
                i = list(q["criteria"]).index(r["answer"][qid])
                letters[LETTERS[i] if i < 26 else f"#{i}"] += 1
            elif q["type"] == "noul":
                letters["yes" if r["answer"][qid] else "no"] += 1
    images = {p for r in rows for p in r["images"]}
    return (f"{name}: {len(rows)} rows, questions by type {dict(qtypes)}, by task_family {dict(fams)}, "
            f"{len(images)} distinct images\n  answer positions: {dict(sorted(letters.items()))}")
