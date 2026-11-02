"""Families 2 (counting, ~15k rows) and 9 (abstention, ~8k rows) from detection labels.

Sources, after removing eval material:
- DOTA-v2.0 train (torchgeo mirror, checked: its val labels are byte-identical to the official ones). Val is never
  used, because it is C18 and the VRSBench test set. Images are cut into non-overlapping 800 px tiles exactly as C18
  does: tiles with a side under 256 px are dropped, objects are assigned by polygon centroid, and difficult objects
  are counted.
- DIOR train + test (torchgeo images, official HBB labels), minus all of DIOR val (= the VRSBench test set) and
  every image used by a DIOR-RSVG val or test pair (C16; pairs rebuilt in the official loader's order).

Counting (family 2): one question per row, one class per question.
- Option layouts: halfway between 60% full / 40% short ascending lists (first-option bias on other benchmarks) and
  30 / 30 / 20 / 20 (DOTA counting fell). Averaging the weights of models trained with those two, which acts like
  this mix, gave the best counting of the three:
  - 45%: "0".."K" (+ "more than K"), K per dataset and class = p99 of training counts, clamped to [10, 100], as in C18;
  - 15%: 4–6 count values around the count, in shuffled order;
  - 10%: count ranges such as "0", "1 to 2", "3 to 5", ..., "more than 20";
  - 30%: the 4–6 values in ascending order, the gold at a uniformly random position.
- 15% of questions ask about a class absent from the image (answer "0"), picked as below.
- Count values are balanced by bin: COUNT_BINS sets each bin's share of the questions, with classes round-robin
  inside a bin. Balancing classes only made 36% of answers "1" (and 19% "0"); on objects
  it was never taught to count (EarthMind houses / paths) the model then fell back to "1" (true 3 -> "1" 49 times
  of 146, d3-flash 19).
- Class names: 40% the dataset's raw label (as C18 asks), else plain English plurals, a synonym in SYN_RATE of them
  (ship -> boat / vessel, small vehicle -> car, ...).
- Templates differ from C18's five. meta.count holds the exact count for soft ordinal (SORD) targets.

Abstention (family 9), all answers computed from boxes:
- presence: one present and one absent class per row (yes / no at 1:1). In HARD_POS_RATE of rows the present class
  is the tile's hardest one (fewest objects, then smallest boxes): without it the model missed small objects
  that are there (EarthMind hallucination detection, accuracy on "yes" items 96 -> 87). Names as for counting, plus
  EarthMind's "Does the image contain ...?" wording.
- position: "where is the X?" over top left / top right / bottom left / bottom right / centre, plus
  "there is no X in this image". It is asked only about a class with exactly one object, whose centre is clearly inside one region.
  In DECEPTIVE_SHARE of rows X is absent and the abstain option is the answer.
- relative position: two single-object classes clearly left/right or above/below each other, plus
  "one of them is not in this image". DECEPTIVE_SHARE of rows swap one class for an absent one.
- Absent classes (here and for zero counts) are picked as hallucination benchmarks pick their negatives (POPE, VHM's
  deceptive splits), a third each: a class seen elsewhere in the same scene, one that often co-occurs with the
  tile's classes ("adversarial"), or a frequent one ("popular"). Drawing them from the same scene, else any class
  at random, lost VHM's adversarial / popular deceptive questions (abspos 74 -> 59).

    MODULE=trainset.fam2_9_objects [ARGS="--sample 10"] sbatch -J eojev-fam29 slurm/convert_cpu.sbatch
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import itertools
import json
import random
import re
import tarfile
import zipfile
from pathlib import Path

from PIL import Image

from bench.common import RAW, write_jsonl
from trainset.common import ROWS, TRAIN, make_row, render, store, choice_options  # noqa: F401

Image.MAX_IMAGE_PIXELS = None
MAN = json.loads((TRAIN / "downloads.json").read_text())
DOTA_DIR = RAW / "train" / "dota"
TILE, MIN_SIDE, K_MIN, K_MAX = 800, 256, 10, 100
TARGET = {"count_dota": 9000, "count_dior": 6000, "presence": 6000, "position": 2500, "relative": 1500}  # F: pres 4000
ABSENT_COUNT_RATE = 0.15  # share of counting questions about an absent class (answer "0")
COUNT_BINS = [(1, 1, 0.15), (2, 2, 0.13), (3, 5, 0.22), (6, 10, 0.15), (11, 20, 0.12), (21, 10**9, 0.08)]  # lo, hi, share
SYN_RATE, HARD_POS_RATE = 0.30, 0.50
SYNONYMS = {  # raw label -> other (singular, plural) names that are still true of every object of the class
    "ship": [("boat", "boats"), ("vessel", "vessels")], "small-vehicle": [("car", "cars")], "vehicle": [("car", "cars")],
    "plane": [("airplane", "airplanes")], "airplane": [("plane", "planes")],
    "swimming-pool": [("pool", "pools")], "storage-tank": [("tank", "tanks")], "storagetank": [("tank", "tanks")],
}
DECEPTIVE_SHARE = 0.4  # absent-object share of position / relative-position rows (A-D: 1/3, too "yes"; E: 0.5, too "no")
LAYOUT_RATES = {"full": 0.45, "ranges": 0.10, "shuffled": 0.15, "short": 0.30}
# count ranges: the lower bound, then each bin's upper bound; the last bin is open ("more than 20" / "21 or more")
RANGE_SCHEMES = [(0, 0, 2, 5, 10, 20), (0, 4, 9, 19, 49), (0, 0, 3, 9, 24), (0, 1, 4, 9, 14, 29)]
RAW_NAME_RATE = 0.40

DOTA_NAMES = {  # raw label -> (singular, plural)
    "plane": ("plane", "planes"), "ship": ("ship", "ships"), "storage-tank": ("storage tank", "storage tanks"),
    "baseball-diamond": ("baseball diamond", "baseball diamonds"), "tennis-court": ("tennis court", "tennis courts"),
    "basketball-court": ("basketball court", "basketball courts"),
    "ground-track-field": ("ground track field", "ground track fields"), "harbor": ("harbour", "harbours"),
    "bridge": ("bridge", "bridges"), "large-vehicle": ("large vehicle", "large vehicles"),
    "small-vehicle": ("small vehicle", "small vehicles"), "helicopter": ("helicopter", "helicopters"),
    "roundabout": ("roundabout", "roundabouts"), "soccer-ball-field": ("soccer field", "soccer fields"),
    "swimming-pool": ("swimming pool", "swimming pools"), "container-crane": ("container crane", "container cranes"),
    "airport": ("airport", "airports"), "helipad": ("helipad", "helipads"),
}
DIOR_NAMES = {
    "airplane": ("airplane", "airplanes"), "airport": ("airport", "airports"),
    "baseballfield": ("baseball field", "baseball fields"), "basketballcourt": ("basketball court", "basketball courts"),
    "bridge": ("bridge", "bridges"), "chimney": ("chimney", "chimneys"), "dam": ("dam", "dams"),
    "Expressway-Service-area": ("expressway service area", "expressway service areas"),
    "Expressway-toll-station": ("expressway toll station", "expressway toll stations"),
    "golffield": ("golf course", "golf courses"), "groundtrackfield": ("ground track field", "ground track fields"),
    "harbor": ("harbour", "harbours"), "overpass": ("overpass", "overpasses"), "ship": ("ship", "ships"),
    "stadium": ("stadium", "stadiums"), "storagetank": ("storage tank", "storage tanks"),
    "tenniscourt": ("tennis court", "tennis courts"), "trainstation": ("train station", "train stations"),
    "vehicle": ("vehicle", "vehicles"), "windmill": ("windmill", "windmills"),
}
CNT_T = [("cnt_v1", "How many {p} can you see in this image?"), ("cnt_v2", "Count the {p} visible in this image."),
         ("cnt_v3", "How many {p} does this image contain?"), ("cnt_v4", "What is the total number of {p} shown?"),
         ("cnt_v5", "How many {p} appear in this image?"), ("cnt_v6", "Give the number of {p} in this image.")]
PRES_T = [("pres_v1", "Is there {a_s} in this image?"), ("pres_v2", "Does this image contain any {p}?"),
          ("pres_v3", "Can you see {a_s} in this image?"), ("pres_v4", "Is at least one {s} visible here?"),
          ("pres_v5", "Are any {p} shown in this image?"), ("pres_v6", "Does the image contain {a_s}?")]
POS_T = [("pos_v1", "Where in the image is the {s}?"), ("pos_v2", "In which part of the image is the {s} located?"),
         ("pos_v3", "Where is the {s} in this image?"), ("pos_v4", "Which part of the image contains the {s}?")]
REL_T = [("rel_v1", "Where is the {a} relative to the {b}?"),
         ("rel_v2", "How is the {a} positioned with respect to the {b}?"),
         ("rel_v3", "Seen from above, where does the {a} lie relative to the {b}?")]
REGIONS = ["top left", "top right", "bottom left", "bottom right", "centre"]
RELS = ["to the left of it", "to the right of it", "above it", "below it"]


# ------------------------------------------------------------------------------------------------ sources
def dota_extract() -> dict[str, Path]:
    """Extract the DOTA-v2.0 train images once (the v1.0 train images + the v2.0 extras); returns pid -> png."""
    snap = Path(MAN["dota"]["path"])
    with open(snap / "samples.csv") as f:
        wanted = {Path(r["image_path"]).stem for r in csv.DictReader(f) if r["split"] == "train" and r["version"] == "2.0"}
    out = DOTA_DIR / "images"
    out.mkdir(parents=True, exist_ok=True)
    have = {p.stem: p for p in out.glob("*.png")}
    if not wanted <= set(have):
        for tar in ("dotav1.0_images_train.tar.gz", "dotav2.0_images_train.tar.gz"):
            with tarfile.open(snap / tar) as tf:
                for m in tf:
                    stem = Path(m.name).stem
                    if m.isfile() and m.name.endswith(".png") and stem in wanted and stem not in have:
                        dest = out / f"{stem}.png"
                        part = dest.with_suffix(".part")
                        with tf.extractfile(m) as src, open(part, "wb") as dst:
                            while chunk := src.read(1 << 22):
                                dst.write(chunk)
                        part.rename(dest)
                        have[stem] = dest
        print(f"  DOTA train images extracted: {len(have)}", flush=True)
    missing = wanted - set(have)
    if missing:
        raise RuntimeError(f"{len(missing)} DOTA train images missing after extraction, e.g. {sorted(missing)[:3]}")
    return {k: have[k] for k in wanted}


def dota_scenes() -> list[dict]:
    images = dota_extract()
    snap = Path(MAN["dota"]["path"])
    labels = {}
    with tarfile.open(snap / "dotav2.0_annotations_train.tar.gz") as tf:
        for m in tf:
            if m.isfile() and m.name.endswith(".txt"):
                labels[Path(m.name).stem] = tf.extractfile(m).read().decode()
    scenes = []
    for pid, path in sorted(images.items()):
        with Image.open(path) as im:
            w, h = im.size
        objs = []
        for line in labels[pid].splitlines():
            p = line.split()
            if len(p) < 10:
                continue
            xs, ys = [float(v) for v in p[0:8:2]], [float(v) for v in p[1:8:2]]
            side = ((max(xs) - min(xs)) * (max(ys) - min(ys))) ** 0.5  # s: box side in pixels (sqrt of its area)
            objs.append({"c": p[8], "x": sum(xs) / 4, "y": sum(ys) / 4, "s": side, "diff": p[9] == "1"})
        scenes.append({"pid": pid, "path": path, "size": (w, h), "objs": objs})
    return scenes


def dota_tiles(scenes) -> list[dict]:
    tiles = []
    for s in scenes:
        w, h = s["size"]
        for ty in range(0, h, TILE):
            for tx in range(0, w, TILE):
                tw, th = min(TILE, w - tx), min(TILE, h - ty)
                if min(tw, th) < MIN_SIDE:
                    continue
                inside = [dict(o, x=(o["x"] - tx) / tw, y=(o["y"] - ty) / th) for o in s["objs"]
                          if tx <= o["x"] < tx + TILE and ty <= o["y"] < ty + TILE]
                tiles.append({"src": "DOTA", "id": f"{s['pid']}_{tx}_{ty}", "scene": s["pid"], "path": s["path"],
                              "box": (tx, ty, tx + tw, ty + th), "objs": inside,
                              "scene_classes": {o["c"] for o in s["objs"]}})
    return tiles


def rsvg_excluded() -> set[str]:
    """Image ids of every DIOR-RSVG val or test pair, in the official loader's order (sorted XMLs, one per object)."""
    obj_re = re.compile(r"<object>.*?</object>", re.S)
    pairs = []
    with zipfile.ZipFile(RAW / "dior-rsvg" / "Annotations.zip") as zf:
        for m in sorted(n for n in zf.namelist() if n.endswith(".xml")):
            x = zf.read(m).decode()
            img = Path(re.search(r"<filename>(.*?)</filename>", x).group(1)).stem
            pairs += [img] * len(obj_re.findall(x))
    out = set()
    for split in ("val", "test"):
        for tok in (RAW / "dior-rsvg" / f"{split}.txt").read_text().split():
            out.add(pairs[int(tok)])
    print(f"  DIOR-RSVG pairs {len(pairs)}; val+test images excluded: {len(out)}", flush=True)
    return out


def dior_images() -> list[dict]:
    off = RAW / "train" / "dior_official"
    with zipfile.ZipFile(off / "ImageSets.zip") as zf:
        val = set(next(zf.read(n) for n in zf.namelist() if n.endswith("Main/val.txt")).decode().split())
    excluded = val | rsvg_excluded()
    obj_re = re.compile(r"<name>(.*?)</name>.*?<xmin>([\d.]+)</xmin>\s*<ymin>([\d.]+)</ymin>\s*<xmax>([\d.]+)</xmax>"
                        r"\s*<ymax>([\d.]+)</ymax>", re.S)
    labels = {}
    with zipfile.ZipFile(off / "Annotations.zip") as zf:
        for n in zf.namelist():
            if "Horizontal Bounding Boxes" in n and n.endswith(".xml"):
                x = zf.read(n).decode()
                size = re.search(r"<width>(\d+)</width>\s*<height>(\d+)</height>", x)
                objs = [{"c": o[0], "x": (float(o[1]) + float(o[3])) / 2, "y": (float(o[2]) + float(o[4])) / 2,
                         "s": ((float(o[3]) - float(o[1])) * (float(o[4]) - float(o[2]))) ** 0.5, "diff": False}
                        for o in obj_re.findall(x)]
                labels[Path(n).stem] = ((int(size.group(1)), int(size.group(2))) if size else (800, 800), objs)
    snap = Path(MAN["dior"]["path"])
    out = []
    for zname in ("Images_trainval.zip", "Images_test.zip"):
        with zipfile.ZipFile(snap / zname) as zf:
            for n in zf.namelist():
                stem = Path(n).stem
                if n.endswith(".jpg") and stem not in excluded and stem in labels:
                    (w, h), objs = labels[stem]
                    out.append({"src": "DIOR", "id": stem, "scene": stem, "zip": snap / zname, "member": n,
                                "box": None, "size": (w, h),
                                "objs": [dict(o, x=o["x"] / w, y=o["y"] / h) for o in objs],
                                "scene_classes": {o["c"] for o in objs}})
    print(f"  DIOR images kept {len(out)} (excluded {len(excluded)} val / DIOR-RSVG val+test ids)", flush=True)
    return out


# ------------------------------------------------------------------------------------------------ images
class ImageStore:
    """Two passes: rows are built with a placeholder image path; `materialize` then writes each used DOTA tile /
    DIOR image once, grouped by scene so that every large DOTA scene is decoded only once, and `fill` puts the real
    paths, sizes and hashes into the rows."""

    def __init__(self):
        self.wanted: dict[str, dict] = {}
        self.done: dict[str, tuple] = {}

    def get(self, t: dict) -> tuple[str, None, None]:
        key = f"{t['src']}:{t['id']}"
        self.wanted.setdefault(key, t)
        return f"@{key}", None, None

    def materialize(self) -> None:
        pid, img = None, None
        zips: dict[Path, zipfile.ZipFile] = {}
        for key, t in sorted(self.wanted.items(), key=lambda kv: (kv[1]["src"], kv[1]["scene"])):
            if t["src"] == "DOTA":
                if pid != t["scene"]:
                    img = Image.open(t["path"]).convert("RGB")
                    pid = t["scene"]
                buf = io.BytesIO()
                img.crop(t["box"]).save(buf, format="PNG")
                self.done[key] = store(buf.getvalue(), f"dota/{t['id']}")
            else:
                zf = zips.setdefault(t["zip"], zipfile.ZipFile(t["zip"]))
                self.done[key] = store(zf.read(t["member"]), f"dior/{t['id']}")
        for zf in zips.values():
            zf.close()

    def fill(self, rows: list[dict]) -> None:
        for r in rows:
            rel, size, sha = self.done[r["images"][0][1:]]
            r["images"] = [rel]
            r["meta"]["image_sha256"], r["meta"]["orig_px"] = sha, size


# ------------------------------------------------------------------------------------------------ questions
def names(src: str) -> dict:
    return DOTA_NAMES if src == "DOTA" else DIOR_NAMES


def with_article(noun: str) -> str:
    return ("an " if noun[0].lower() in "aeiou" else "a ") + noun


def pick_name(rng, src: str, c: str) -> tuple[str, str, bool]:
    """(singular, plural, is_synonym): the class's English name, or in SYN_RATE of questions one of its synonyms."""
    if c in SYNONYMS and rng.random() < SYN_RATE:
        return (*rng.choice(SYNONYMS[c]), True)
    return (*names(src)[c], False)


def bin_quota(n: int, shares: list[float], avail: list[int]) -> list[int]:
    """Questions per bin: each bin its share of n, at most what it has; the spare goes to the others pro rata."""
    alloc, left = [0] * len(shares), n
    open_ = [i for i, a in enumerate(avail) if a > 0]
    while left > 0 and open_:
        total = sum(shares[i] for i in open_)
        for i in open_:
            give = min(max(1, int(left * shares[i] / total)), avail[i] - alloc[i], left)
            alloc[i] += give
            left -= give
        open_ = [i for i in open_ if alloc[i] < avail[i]]
    return alloc


def class_round_robin(rng, pairs: list, k: int) -> list:
    """k (tile, class, count) pairs, one class after another so that frequent classes do not dominate."""
    by_class = collections.defaultdict(list)
    for p in pairs:
        by_class[p[1]].append(p)
    for v in by_class.values():
        rng.shuffle(v)
    out = []
    while len(out) < k and any(by_class.values()):
        for c in sorted(by_class):
            if by_class[c] and len(out) < k:
                out.append(by_class[c].pop())
    return out


def k_per_class(tiles) -> dict[tuple, int]:
    per = collections.defaultdict(list)
    for t in tiles:
        for c, n in collections.Counter(o["c"] for o in t["objs"]).items():
            per[(t["src"], c)].append(n)
    ks = {}
    for key, v in per.items():
        v.sort()
        ks[key] = min(K_MAX, max(K_MIN, v[min(len(v) - 1, int(0.99 * len(v)))]))
    return ks


def range_options(rng, count: int) -> tuple[list[str], str]:
    """Count ranges in ascending order ("0", "1 to 2", ..., "more than 20" / "20 or more"), one scheme per row."""
    lo, *rest = rng.choice(RANGE_SCHEMES)
    bins, start = [], lo
    for edge in rest:
        bins.append((start, edge))
        start = edge + 1
    opts = [str(a) if a == b else f"{a} to {b}" for a, b in bins]
    top = f"more than {start - 1}" if rng.random() < 0.5 else f"{start} or more"
    opts.append(top)
    gold = next((o for (a, b), o in zip(bins, opts) if a <= count <= b), top)
    return opts, gold


def count_options(rng, count: int, k: int) -> tuple[list[str], str, str]:
    """One of four layouts, at LAYOUT_RATES (45 / 15 / 10 / 30):
    - full: 0..K and "more than K", ascending;
    - shuffled: a short list of 4-6 values around the count, in random order;
    - ranges: count ranges;
    - short: the short list in ascending order.
    Full lists always start at 0, so with 60% of them the model learned to favour the first option on other
    benchmarks' counting questions."""
    gold = str(count) if count <= k else f"more than {k}"
    u = rng.random()
    if u < LAYOUT_RATES["full"]:
        return [str(i) for i in range(k + 1)] + [f"more than {k}"], gold, "full"
    if u < LAYOUT_RATES["full"] + LAYOUT_RATES["ranges"]:
        opts, gold = range_options(rng, count)
        return opts, gold, "ranges"
    shuffled = u < LAYOUT_RATES["full"] + LAYOUT_RATES["ranges"] + LAYOUT_RATES["shuffled"]
    opts, gold = short_options(rng, count)
    if shuffled:
        rng.shuffle(opts)
    return opts, gold, "shuffled" if shuffled else "short"


def short_options(rng, count: int) -> tuple[list[str], str]:
    """4-6 values around the count, ascending, with the gold at a uniformly random position where possible."""
    m = rng.choice([4, 5, 6])
    pos = rng.randrange(m)
    step = max(1, round(count * rng.uniform(0.1, 0.35)))
    below = [count - step * (i + 1) for i in range(pos)]
    above = [count + step * (i + 1) for i in range(m - 1 - pos)]
    if below and below[-1] < 0:  # cannot go below zero: shift the gold position up
        below = [v for v in below if v >= 0]
        above = [count + step * (i + 1) for i in range(m - 1 - len(below))]
    vals = sorted(set(below + [count] + above))
    return [str(v) for v in vals], str(count)


def cooccurrence(items) -> dict:
    """Per source: how many tiles hold each class (freq) and each pair of classes (pair)."""
    freq, pair = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for t in items:
        cs = sorted({o["c"] for o in t["objs"]})
        freq[t["src"]].update(cs)
        pair[t["src"]].update(itertools.combinations(cs, 2))
    return {"freq": freq, "pair": pair}


def pick_absent(rng, t, present, stats) -> str | None:
    """A class absent from tile t: a third each 'scene' (seen elsewhere in the same scene), 'adversarial' (weighted
    by co-occurrence with the tile's classes) and 'popular' (weighted by how many tiles hold it); a kind without
    candidates falls through to the next."""
    present = set(present)
    absent = sorted(set(names(t["src"])) - present)
    if not absent:
        return None
    kinds = ["scene", "adversarial", "popular"]
    start = rng.randrange(3)
    for kind in kinds[start:] + kinds[:start]:
        if kind == "scene":
            cand = sorted(t["scene_classes"] - present)
            if cand:
                return rng.choice(cand)
            continue
        if kind == "adversarial":
            pair = stats["pair"][t["src"]]
            w = [sum(pair[tuple(sorted((a, p)))] for p in present) for a in absent]
        else:
            w = [stats["freq"][t["src"]][a] for a in absent]
        if sum(w) > 0:
            return rng.choices(absent, weights=w)[0]
    return rng.choice(absent)


def counting_rows(rng, items, ks, n, source, imgs) -> list[dict]:
    stats = cooccurrence(items)
    pairs = []
    for t in items:
        cnt = collections.Counter(o["c"] for o in t["objs"])
        for c, v in cnt.items():
            pairs.append((t, c, v))
        a = pick_absent(rng, t, cnt, stats)
        if a:
            pairs.append((t, a, 0))
    zero = [p for p in pairs if p[2] == 0]
    picked = rng.sample(zero, min(int(n * ABSENT_COUNT_RATE), len(zero)))
    bins = [[p for p in pairs if lo <= p[2] <= hi] for lo, hi, _ in COUNT_BINS]  # present pairs by count bin
    quota = bin_quota(n - len(picked), [s for *_, s in COUNT_BINS], [len(b) for b in bins])
    for b, q in zip(bins, quota):
        picked += class_round_robin(rng, b, q)
    rows = []
    for t, c, v in picked:
        _, plur, syn = pick_name(rng, t["src"], c)
        label = c if rng.random() < RAW_NAME_RATE else plur
        syn = syn and label == plur
        tid, text = rng.choice(CNT_T)
        k = ks.get((t["src"], c), K_MIN)
        opts, ans, form = count_options(rng, v, k)
        rel, size, sha = imgs.get(t)
        rows.append(make_row(rid=f"tr:fam2:{source}:{t['id']}:{c}", group_id=f"{source}:{t['scene']}", family="fam2",
                             source=source, task_family="count", images=[rel], state="",
                             questions={"q": {"type": "choice", "instructions": text.format(p=label),
                                              "criteria": {o: None for o in opts}}},
                             answer={"q": ans},
                             meta={"template": tid, "class": c, "count": v, "k": k, "options_form": form,
                                   "raw_name": label == c, "synonym": syn, "image_sha256": sha, "orig_px": size,
                                   "licence": "DOTA: academic use only" if t["src"] == "DOTA" else "DIOR: CC BY-NC 4.0"}))
    return rows


def region_of(x: float, y: float) -> str | None:
    if 0.35 <= x <= 0.65 and 0.35 <= y <= 0.65:
        return "centre"
    if abs(x - 0.5) < 0.08 or abs(y - 0.5) < 0.08 or (0.28 < x < 0.72 and 0.28 < y < 0.72):
        return None  # too close to a boundary to be unambiguous
    return ("top " if y < 0.5 else "bottom ") + ("left" if x < 0.5 else "right")


def abstention_rows(rng, items, n_pres, n_pos, n_rel, imgs) -> list[dict]:
    stats = cooccurrence(items)
    rows = []
    pool = items[:]
    rng.shuffle(pool)
    # presence: one yes + one no question per row
    for t in pool:
        if sum(r["meta"]["kind"] == "presence" for r in rows) >= n_pres:
            break
        present = sorted({o["c"] for o in t["objs"]})
        absent = pick_absent(rng, t, present, stats)
        if not present or not absent:
            continue
        hard = rng.random() < HARD_POS_RATE
        if hard:  # fewest objects, then smallest median box side
            sides = collections.defaultdict(list)
            for o in t["objs"]:
                sides[o["c"]].append(o["s"])
            yes_c = min(present, key=lambda c: (len(sides[c]), sorted(sides[c])[len(sides[c]) // 2]))
        else:
            yes_c = rng.choice(present)
        qs = [(yes_c, True), (absent, False)]
        rng.shuffle(qs)
        questions, answer, tids, syns = {}, {}, [], []
        for j, (c, yes) in enumerate(qs):
            tid, text = rng.choice(PRES_T)
            sing, plur, syn = pick_name(rng, t["src"], c)
            questions[f"q{j}"] = {"type": "noul", "criteria": None,
                                  "instructions": text.format(s=sing, p=plur, a_s=with_article(sing))}
            answer[f"q{j}"] = yes
            tids.append(tid)
            syns.append(syn)
        rel, size, sha = imgs.get(t)
        rows.append(make_row(rid=f"tr:fam9:{t['src']}:{t['id']}:pres", group_id=f"{t['src']}:{t['scene']}",
                             family="fam9", source=t["src"], task_family="presence", images=[rel], state="",
                             questions=questions, answer=answer,
                             meta={"kind": "presence", "template": tids, "asked": qs, "hard_positive": hard,
                                   "synonym": syns, "image_sha256": sha, "orig_px": size}))
    # position: single-object class in a clear region, or an absent class (abstain answer). A quota keeps exactly
    # DECEPTIVE_SHARE deceptive: tiles without a usable single object are skipped for factual rows, which would
    # otherwise inflate the deceptive share.
    k_pos = n_dec = 0
    for t in pool:
        if k_pos >= n_pos:
            break
        cnt = collections.Counter(o["c"] for o in t["objs"])
        nm = names(t["src"])
        singles = [(o["c"], region_of(o["x"], o["y"])) for o in t["objs"] if cnt[o["c"]] == 1]
        singles = [s for s in singles if s[1]]
        absent = pick_absent(rng, t, cnt, stats)
        deceptive = n_dec < (k_pos + 1) * DECEPTIVE_SHARE
        if deceptive and absent:
            n_dec += 1
            c, gold = absent, None
        elif singles and not deceptive:
            c, gold = rng.choice(singles)
        else:
            continue
        abstain = f"there is no {nm[c][0]} in this image"
        opts = REGIONS + [abstain]
        rng.shuffle(opts)
        tid, text = rng.choice(POS_T)
        rel, size, sha = imgs.get(t)
        rows.append(make_row(rid=f"tr:fam9:{t['src']}:{t['id']}:pos:{c}", group_id=f"{t['src']}:{t['scene']}",
                             family="fam9", source=t["src"], task_family="position", images=[rel], state="",
                             questions={"q": {"type": "choice", "instructions": text.format(s=nm[c][0]),
                                              "criteria": {o: None for o in opts}}},
                             answer={"q": gold or abstain},
                             meta={"kind": "position", "template": tid, "class": c, "deceptive": gold is None,
                                   "image_sha256": sha, "orig_px": size}))
        k_pos += 1
    # relative position between two single-object classes (same DECEPTIVE_SHARE quota)
    k_rel = n_dec_rel = 0
    for t in pool:
        if k_rel >= n_rel:
            break
        cnt = collections.Counter(o["c"] for o in t["objs"])
        nm = names(t["src"])
        singles = [o for o in t["objs"] if cnt[o["c"]] == 1]
        cands = []
        for a in singles:
            for b in singles:
                if a["c"] == b["c"]:
                    continue
                dx, dy = a["x"] - b["x"], a["y"] - b["y"]
                if abs(dx) > 0.2 and abs(dx) > 2 * abs(dy):
                    cands.append((a["c"], b["c"], RELS[0] if dx < 0 else RELS[1]))
                elif abs(dy) > 0.2 and abs(dy) > 2 * abs(dx):
                    cands.append((a["c"], b["c"], RELS[2] if dy < 0 else RELS[3]))
        if not cands:
            continue
        a, b, gold = rng.choice(cands)
        absent = pick_absent(rng, t, cnt, stats)
        if n_dec_rel < (k_rel + 1) * DECEPTIVE_SHARE and absent:
            a, gold = absent, None
            n_dec_rel += 1
        abstain = "one of them is not in this image"
        opts = RELS + [abstain]
        rng.shuffle(opts)
        tid, text = rng.choice(REL_T)
        rel, size, sha = imgs.get(t)
        rows.append(make_row(rid=f"tr:fam9:{t['src']}:{t['id']}:rel:{a}:{b}", group_id=f"{t['src']}:{t['scene']}",
                             family="fam9", source=t["src"], task_family="relative_position", images=[rel], state="",
                             questions={"q": {"type": "choice", "instructions": text.format(a=nm[a][0], b=nm[b][0]),
                                              "criteria": {o: None for o in opts}}},
                             answer={"q": gold or abstain},
                             meta={"kind": "relative", "template": tid, "a": a, "b": b, "deceptive": gold is None,
                                   "image_sha256": sha, "orig_px": size}))
        k_rel += 1
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0, help="only this many rows per part, for review")
    args = p.parse_args()
    target = {k: (args.sample or v) for k, v in TARGET.items()}
    dota = dota_tiles(dota_scenes())
    dior = dior_images()
    print(f"  DOTA tiles {len(dota)}, DIOR images {len(dior)}", flush=True)
    ks = k_per_class(dota + dior)
    imgs = ImageStore()
    rng = random.Random(0)
    fam2 = counting_rows(random.Random("fam2:dota"), dota, ks, target["count_dota"], "DOTA", imgs)
    fam2 += counting_rows(random.Random("fam2:dior"), dior, ks, target["count_dior"], "DIOR", imgs)
    half = lambda n: (n + 1) // 2  # noqa: E731
    fam9 = abstention_rows(random.Random("fam9:dota"), dota, half(target["presence"]), half(target["position"]),
                           half(target["relative"]), imgs)
    fam9 += abstention_rows(random.Random("fam9:dior"), dior, target["presence"] // 2, target["position"] // 2,
                            target["relative"] // 2, imgs)
    print(f"  writing {len(imgs.wanted)} images", flush=True)
    imgs.materialize()
    imgs.fill(fam2 + fam9)
    suffix = ".sample" if args.sample else ""
    for name, rows in (("fam2_counting", fam2), ("fam9_abstention", fam9)):
        write_jsonl(ROWS / f"{name}{suffix}.jsonl", rows)
        with open(ROWS / f"{name}{suffix}.txt", "w") as f:
            for r in (rows if args.sample else rng.sample(rows, min(30, len(rows)))):
                f.write(render(r) + "\n\n")
        kinds = collections.Counter(r["task_family"] for r in rows)
        print(f"{name}: {len(rows)} rows; {dict(kinds)}; splits {dict(collections.Counter(r['split'] for r in rows))}")
    zero = sum(r["meta"]["count"] == 0 for r in fam2)
    by_bin = collections.Counter(next((f"{lo}-{hi}" if hi < 10**9 else f"{lo}+" for lo, hi, _ in COUNT_BINS
                                       if lo <= r["meta"]["count"] <= hi), "0") for r in fam2)
    print("counting: answers by count bin (share %):",
          {b: round(100 * v / len(fam2), 1) for b, v in sorted(by_bin.items(), key=lambda kv: int(kv[0].split("-")[0].rstrip("+")))})
    pres = [r for r in fam9 if r["task_family"] == "presence"]
    print(f"presence: {len(pres)} rows, hard positives {sum(r['meta']['hard_positive'] for r in pres)}, "
          f"questions with a synonym {sum(sum(r['meta']['synonym']) for r in pres)}")
    forms = collections.Counter(r["meta"]["options_form"] for r in fam2)
    dec = collections.Counter(r["meta"].get("deceptive") for r in fam9)
    print(f"counting: zero-count {zero}, option forms {dict(forms)}; abstention deceptive {dict(dec)}; "
          f"images written {len(imgs.done)}")


if __name__ == "__main__":
    main()
