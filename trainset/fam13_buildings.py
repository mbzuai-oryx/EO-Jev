"""Family 13, building counting: ~3,000 counting questions on crops of xBD pre-disaster images,
counted from xBD's building footprints.

Why: families 2 count DOTA / DIOR classes only, which have no buildings. Without this family the model fell back
to small counts on objects it was never taught to count: EarthMind "houses" 40.7 -> 24.7 (d3-flash -> fine-tuned).

- Source: xBD train + tier3 (the HF mirror family 4 uses), pre-disaster images and footprint masks; scenes named in
  any eval row are blocked as in family 4.
- Buildings: connected components of the footprint mask (>= 100 px), found as C9 / family 4 find them. Touching
  footprints merge into one component, so a crop is skipped when a component in it is over MERGED_X times the
  scene's median footprint area.
- Crops: squares of CROP_SIDES px at random positions (1024 = the whole tile), TRIES per scene. A building counts
  when at least 70% of its box is inside the crop; a crop with any building 30-70% inside is skipped as ambiguous.
  (A first build with 256-512 px crops, 4 tries and an 80 / 20 rule gave few crops above 10 buildings.)
- Counts balanced by COUNT_BINS (share of questions per count bin, with a few empty crops), no xBD event above
  EVENT_CAP of a bin.
- Wording: family 2's counting templates with "buildings", "houses" or "structures"; options as family 2
  (count_options, K from the counts' p99); meta.count gives SORD targets.

    MODULE=trainset.fam13_buildings [ARGS="--sample 10"] sbatch -J eojev-fam13 slurm/convert_cpu.sbatch
"""

from __future__ import annotations

import argparse
import collections
import io
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from scipy import ndimage

from bench.common import write_jsonl
from bench.converters.xbd_damage import MIN_AREA
from trainset.common import ROWS, make_row, render, store
from trainset.fam2_9_objects import CNT_T, count_options
from trainset.fam4_damage import MAN, blocked_scenes, event_balanced, png

FAMILY = "fam13"
TARGET = 3000
CROP_SIDES = (256, 384, 512, 768, 1024)  # 1024: the whole tile
TRIES = 8
MERGED_X = 4.0
COUNT_BINS = [(0, 0, 0.05), (1, 1, 0.10), (2, 2, 0.10), (3, 5, 0.22), (6, 10, 0.22), (11, 20, 0.19), (21, 10**9, 0.12)]
NOUNS = [("buildings", 0.55), ("houses", 0.30), ("structures", 0.15)]


def components(mask: np.ndarray) -> tuple[list[tuple[int, int, int, int]], list[int]]:
    """Footprint components (y0, y1, x0, x1) of at least MIN_AREA pixels, and their pixel areas."""
    labels, _ = ndimage.label(mask > 0)
    areas = np.bincount(labels.ravel())
    boxes, out_areas = [], []
    for i, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is not None and areas[i] >= MIN_AREA:
            boxes.append((sl[0].start, sl[0].stop, sl[1].start, sl[1].stop))
            out_areas.append(int(areas[i]))
    return boxes, out_areas


def window_count(boxes, areas, big: float, y: int, x: int, side: int) -> int | None:
    """Buildings counted in the window, or None when it is ambiguous (a building 20-80% inside, or a merged one)."""
    n = 0
    for (y0, y1, x0, x1), a in zip(boxes, areas):
        oy = max(0, min(y1, y + side) - max(y0, y))
        ox = max(0, min(x1, x + side) - max(x0, x))
        frac = oy * ox / ((y1 - y0) * (x1 - x0))
        if frac >= 0.7:
            if a > big:
                return None
            n += 1
        elif frac > 0.3:
            return None
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0, help="only this many rows, for review")
    args = p.parse_args()
    n_target = args.sample or TARGET
    blocked = blocked_scenes()
    snap = Path(MAN["xbd"]["path"])
    files = sorted(f for f in (snap / "data").glob("*.parquet") if f.name.startswith(("train-", "tier3-")))
    rng = random.Random("fam13")

    # pass 1: candidate crops from the pre-disaster footprint masks
    cands, where = [], {}
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            for i, r in enumerate(pf.read_row_group(g, columns=["image_name", "t1_mask"]).to_pylist()):
                scene = r["image_name"].rsplit("/", 1)[-1]
                if scene in blocked:
                    continue
                mask = np.array(png(r["t1_mask"]))
                boxes, areas = components(mask)
                big = MERGED_X * float(np.median(areas)) if areas else 0.0
                h, w = mask.shape[:2]
                where[scene] = (f, g, i)
                seen = set()  # the whole-tile crop has one position: draw each window once
                for _ in range(TRIES):
                    side = rng.choice(CROP_SIDES)
                    y, x = rng.randrange(0, h - side + 1), rng.randrange(0, w - side + 1)
                    if (y, x, side) in seen:
                        continue
                    seen.add((y, x, side))
                    n = window_count(boxes, areas, big, y, x, side)
                    if n is not None:
                        cands.append((scene, y, x, side, n))
    counts = sorted(c[4] for c in cands)
    k = int(min(60, max(10, counts[int(0.99 * (len(counts) - 1))]))) if counts else 10
    print(f"  candidate crops {len(cands)} from {len(where)} scenes; K = {k}", flush=True)

    picked = []
    for lo, hi, share in COUNT_BINS:
        pool = [c for c in cands if lo <= c[4] <= hi]
        picked += event_balanced(rng, pool, round(n_target * share), lambda c: c[0])[:round(n_target * share)]
    by_scene = collections.defaultdict(list)
    for c in picked:
        by_scene[c[0]].append(c)

    # pass 2: crop the pre-disaster images
    by_group = collections.defaultdict(list)
    for scene in by_scene:
        f, g, i = where[scene]
        by_group[(f, g)].append((scene, i))
    rows = []
    for (f, g), items in sorted(by_group.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        t = pq.ParquetFile(f).read_row_group(g, columns=["t1_image"]).to_pylist()
        for scene, i in items:
            image = png(t[i]["t1_image"]).convert("RGB")
            for _, y, x, side, n in by_scene[scene]:
                buf = io.BytesIO()
                image.crop((x, y, x + side, y + side)).save(buf, format="PNG")
                rel, size, sha = store(buf.getvalue(), f"xbd/count/{scene}_{y}_{x}_{side}")
                noun = rng.choices([w for w, _ in NOUNS], [p for _, p in NOUNS])[0]
                tid, text = rng.choice(CNT_T)
                opts, ans, form = count_options(rng, n, k)
                rows.append(make_row(
                    rid=f"tr:{FAMILY}:xBD:{scene}:{y}:{x}:{side}", group_id=f"xBD:{scene}", family=FAMILY,
                    source="xBD", task_family="count", images=[rel], state="",
                    questions={"q": {"type": "choice", "instructions": text.format(p=noun),
                                     "criteria": {o: None for o in opts}}},
                    answer={"q": ans},
                    meta={"template": tid, "class": "building", "noun": noun, "count": n, "k": k,
                          "options_form": form, "crop": [y, x, side], "event": scene.rsplit("_", 1)[0],
                          "image_sha256": sha, "orig_px": size, "licence": "xBD: CC BY-NC-SA 4.0"}))
    suffix = ".sample" if args.sample else ""
    write_jsonl(ROWS / f"fam13_buildings{suffix}.jsonl", rows)
    with open(ROWS / f"fam13_buildings{suffix}.txt", "w") as fh:
        for r in (rows if args.sample else rng.sample(rows, min(30, len(rows)))):
            fh.write(render(r) + "\n\n")
    hist = collections.Counter(next(f"{lo}-{hi}" for lo, hi, _ in COUNT_BINS if lo <= r["meta"]["count"] <= hi)
                               for r in rows)
    print(f"fam13_buildings: {len(rows)} rows; splits {dict(collections.Counter(r['split'] for r in rows))}; "
          f"count bins {dict(hist)}; nouns {dict(collections.Counter(r['meta']['noun'] for r in rows))}")


if __name__ == "__main__":
    main()
