"""C13 XLRS-Bench-lite (3,080 questions on 1,434 ultra-high-resolution images, up to 10,000 x 10,000 px) -> rows.

Two input variants (report separately; neither is the paper's native-resolution protocol):
  V0 "downscale": each image resized (Lanczos) to <= 1,638,400 px, the d3 input cap.
  V1 "tiles":     the V0 overview plus the four quadrants, each resized to <= 1,638,400 px (5 images; ~3.4x
                  the pixel density of V0 for 10k images). The state names the layout.
Single-answer items (2,983) are one `choice` over the options' texts; multi-answer items (97) become one `noul`
per option. Regional questions refer to red circles drawn on the image, which survive resizing.
Change-detection items carry two images (paths joined by ':'); V1 tiles each of them.

    python -m bench.converters.xlrs
"""

from __future__ import annotations

import glob
import io
import math
import re
from collections import Counter

import pyarrow as pa
from PIL import Image

from bench.common import CONVERTED, MAX_PIXELS, RAW, save_png, validate_row, write_jsonl

Image.MAX_IMAGE_PIXELS = None
SRC = RAW / "xlrs-lite" / "train"
OUT = CONVERTED / "xlrs"
OPTION = re.compile(r"^\(([A-Z])\)\s*(.*)$", re.S)


def fit(image: Image.Image) -> Image.Image:
    if image.width * image.height <= MAX_PIXELS:
        return image
    s = math.sqrt(MAX_PIXELS / (image.width * image.height))
    return image.resize((max(1, int(image.width * s)), max(1, int(image.height * s))), Image.Resampling.LANCZOS)


def render(key: str, data: bytes) -> tuple[str, list[str], list[int]]:
    """Write the overview and the four quadrants once per source image; return their relative paths."""
    over = OUT / "images" / f"{key}_full.png"
    quads = [OUT / "images" / f"{key}_q{i}.png" for i in range(4)]
    image = Image.open(io.BytesIO(data))
    size = list(image.size)
    if not (over.exists() and all(q.exists() for q in quads)):
        image = image.convert("RGB")
        save_png(fit(image), over)
        w, h = image.size
        boxes = [(0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2), (0, h // 2, w // 2, h), (w // 2, h // 2, w, h)]
        for q, box in zip(quads, boxes):
            save_png(fit(image.crop(box)), q)
    rel = lambda p: str(p.relative_to(CONVERTED))  # noqa: E731
    return rel(over), [rel(q) for q in quads], size


def main() -> None:
    rows_v0, rows_v1 = [], []
    for f in sorted(glob.glob(str(SRC / "*.arrow"))):
        with pa.memory_map(f) as src:
            table = pa.ipc.open_stream(src).read_all()
        for r in table.to_pylist():
            parsed = [OPTION.match(o.strip()) for o in r["multi-choice options"]]
            if not all(parsed):
                raise ValueError(f"unparsed options: {r['multi-choice options']}")
            letters = [m.group(1) for m in parsed]
            texts = [m.group(2).strip() for m in parsed]
            if len(set(texts)) != len(texts):
                texts = [f"{t} ({l})" for l, t in zip(letters, texts)]
            golds = {texts[letters.index(c)] for c in r["answer"].strip() if c in letters}
            sources = r["path"].split(":")
            if len(r["image"]) != len(sources):
                raise ValueError(f"{r['path']}: {len(r['image'])} images for {len(sources)} paths")
            overviews, tiles, sizes = [], [], []
            for p, im in zip(sources, r["image"]):
                key = re.sub(r"[^A-Za-z0-9_.-]", "_", p)
                o, q, s = render(key, im["bytes"])
                overviews.append(o)
                tiles += [o] + q
                sizes.append(s)
            if len(golds) == 1:
                questions = {"q": {"type": "choice", "instructions": r["question"], "criteria": {t: None for t in texts}}}
                answer = {"q": golds.pop()}
                multi = False
            else:
                questions = {f"opt{i}": {"type": "noul", "instructions": f"{r['question']} Is \"{t}\" one of the correct options?"}
                             for i, t in enumerate(texts)}
                answer = {f"opt{i}": t in golds for i, t in enumerate(texts)}
                multi = True
            category, _, sub = r["category"].partition("/")
            base = {
                "group_id": f"XLRS:{r['path']}",
                "benchmark": "XLRS-Bench-lite",
                "split": "lite",
                "task_family": category.strip().lower().replace(" ", "_") + ("_multi" if multi else ""),
                "skills": [sub.strip()],
                "questions": questions,
                "answer": answer,
                "meta": {"category": r["category"], "orig_px": sizes, "multi_select": multi, "source": r["path"]},
            }
            n_src = len(sources)
            v0 = {**base, "id": f"XLRS:{r['index']}:v0", "images": overviews,
                  "state": "" if n_src == 1 else "Image 1 is the earlier and Image 2 the later image of the same area."}
            layout = ("Each source image is given as 5 images: the whole image, then its top-left, top-right, "
                      "bottom-left and bottom-right quarters in more detail.")
            if n_src > 1:
                layout += " Images 1-5 show the earlier image and images 6-10 the later image of the same area."
            v1 = {**base, "id": f"XLRS:{r['index']}:v1", "images": tiles, "state": layout}
            v0["meta"] = {**base["meta"], "variant": "downscale"}
            v1["meta"] = {**base["meta"], "variant": "overview+quadrants"}
            for v in (v0, v1):
                validate_row(v)
            rows_v0.append(v0)
            rows_v1.append(v1)
    for i, (a, b) in enumerate(zip(rows_v0, rows_v1)):  # the source `index` repeats across shards: make ids unique
        a["id"] += f"#{i}"
        b["id"] += f"#{i}"
    write_jsonl(OUT / "xlrs_lite_v0_downscale.jsonl", rows_v0)
    write_jsonl(OUT / "xlrs_lite_v1_tiles.jsonl", rows_v1)
    print("XLRS-lite:", len(rows_v0), "questions per variant;", Counter(r["task_family"] for r in rows_v0))


if __name__ == "__main__":
    main()
