"""C29 SpatialUAV (arXiv 2606.27876; 4,331 items, 14 task types) -> closed-form rows (adapted subset).

Source: HF `Hyu-Zhang/SpatialUAV` @ 0c3df2496c263edf075af73d1fb18e93f4ab8e25 (licence "other": the images keep the
terms of BEDI, AirCopBench, MAVREC, AirScape and University-1652): `annotations_shuffled.parquet` and the image
archives, extracted to raw/spatialuav/SpatialUAV/ (the annotation paths `./SpatialUAV/...` resolve from
raw/spatialuav/). Scoring rules: github.com/Hyu-Zhang/SpatialUAV eval_spatialuav.py (deterministic per task).

ADAPTED (closed-form subset, answers turned into options, resolution); report it as "not the published protocol".
Kept tasks and their conversion (published answer formats in brackets):
  a2a_collaboration_recognition (420) / a2g_collaboration_recognition (500)   [option letter A-D]
      5 images (query + 4 candidates); `choice` over "Image 2" .. "Image 5" (the question says images 2-5 are
      options A-D, so the letters line up with the published ones).
  direction_recognition (455)   [o'clock]   12-way `choice` "1 o'clock" .. "12 o'clock".
  distance_comparison (261)   [one region label]   `choice` over the region labels the question names
      ("Among Region 1 and 6" -> 2-way; "Regions 1, 3, and 6" -> 3-way) or, for "among all the <class> objects
      boxed / all the regions containing <class>", over every region label in the image.
  region_recognition (582) / anomaly_detection (17)   [set of region labels]   items whose answer is one region ->
      `choice` over every region label in the image; items with 2+ regions -> one `noul` per region label in the
      image ("... Is Region k one of them?"), task_family suffix "_multi", scored by exact-set match (the
      published "correct" flag is the same exact-set match; its partial "region_score" is not reproduced).
  a2g_path_planning (144)   [Direction 1-6]   6-way `choice` "Direction 1" .. "Direction 6": the ground image always
      carries the same six numbered arrows (gen_path_planning_arrows.py), so the option set is fixed.
Region labels: single images (samples_Single_Image) carry boxes labelled "Region 0" .. "Region N-1" whose colours are
evenly spaced hues (box k has hue k/N), so N is read off the image: REGION_COUNT below was derived with
`--detect-regions` (hue-constant hollow rectangles; N = the hue grid they sit on) and compared with the highest
region id the questions use; the 13 images where the two disagree, and 5 others, were checked by eye (all
confirmed the detector; img0121, where too few boxes were detected, is set by eye).
Each question's format sentence ("Answer with only the region labels, formatted exactly like ...", "Respond with
only one o'clock direction ...", etc.) is removed (the runners ask for a letter / yes-no); the rest of the published
question is kept verbatim (meta.published_question has the original).
Images: every image above 1,638,400 px is downscaled (Lanczos, JPEG q95): single images ~15 MP -> ~1.6 MP (x0.33),
A2G 2704x1520 / 3840x2160 -> x0.63 / x0.44, A2A 1920x1080 -> x0.89. The region labels are drawn on the source
pixels; after the resize their text is still >= 16.8 px high including descenders (cap height ~13 px; minimum
over all 154 images measured with --detect-regions), so no tiling is needed. The official Qwen inference
(run_spatialuav_inference.py) caps images at 1280*28*28 = 1,003,520 px, smaller than ours.
Dropped (counted in the run log; 2,356 of the 2,379 items of the kept tasks remain, 4,245 questions):
  - malformed ground truths / questions: Distance "Region ." (3), Direction questions naming "Region " without a
    number (3), Distance answers not among the regions the question names (3);
  - items naming or answering a region label the image does not have (Direction 1, Region Recognition 1; e.g.
    "Region 10" on an image with 8 boxes);
  - img0066's 12 questions: its question set contains all 8 of img0077's and does not match its own image (checked
    by eye: e.g. its "trees" answer regions are a building and a lamp post);
  - tasks without a closed form (1,969 items): Motion_Understanding (1,000 free-text descriptions, GPT-judged),
    A2A Camera Transformation (461 angles / distances), A2A Object Matching (212 boxes or region labels on
    unlabelled candidate sets), A2A / A2G Shared Association (75 / 65 region-pair lists), A2A Occlusion Removal (63)
    and A2G View Translation (76): region labels whose candidate set is not recoverable from the release, and the
    A2G labels are ~6 px high after the resize.
Result: 2,356 rows; choice 2,084 (collaboration 420 + 500, direction 448, distance 252 [2-way 77, 3-way 15,
all-regions 160], region recognition 305, anomaly 15, path planning 144) and noul 2,161 (region recognition 270
multi-region items -> 2,143, anomaly 2 -> 18).

    MODULE=bench.converters.spatialuav sbatch slurm/convert_cpu.sbatch     (or: python -m bench.converters.spatialuav)
    python -m bench.converters.spatialuav --detect-regions   (re-derive the region counts; prints disagreements)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from bench.common import CONVERTED, DATASETS, RAW, validate_row, write_jsonl
from bench.converters.uav_common import answer_balance, show_samples, store_fitted

REV = "0c3df2496c263edf075af73d1fb18e93f4ab8e25"
SNAP = (Path(os.environ.get("EOJEV_DATASET_HUB", DATASETS / "hf_hub")) / "datasets--Hyu-Zhang--SpatialUAV" / "snapshots"
        / REV)
SRC = RAW / "spatialuav"
OUT = CONVERTED / "spatialuav"
REGION_TASKS = ("Region_Recognition", "Direction_Recognition", "Distance_Comparison", "Anomaly_Detection")
KEPT = ("A2A_Collaboration_Recognition", "A2G_Collaboration_Recognition", "A2G_Path_Planning") + REGION_TASKS
MISASSIGNED_IMAGES = {"./SpatialUAV/samples_Single_Image/img0066.jpg"}
FORMAT_SENTENCES = [
    r"\s*Answer with only one option letter: A, B, C, or D\. Provide no explanation\.",
    r"\s*Answer with only the region labels, formatted exactly like `Region 1, 2`\. No explanation\.",
    r"\s*Respond with only one o'clock direction, formatted exactly as `2 o'clock`\. Do not include any other words "
    r"or explanation\.",
    r"\s*Output exactly one region label in the format `Region 6`\. Output only the answer and nothing else\.",
    r"\s*Output exactly one label in the format `Direction 3`, with no additional text\.",
]
# Region boxes per single image (labels "Region 0" .. "Region N-1"), from `--detect-regions` (2026-10-21, job log
# logs/eojev-conv-suav-detect-330073.out). Checked by eye: the 13 images where it disagrees with the highest region id
# the questions use (img0008 7, img0021 8, img0048 10, img0066 8, img0077 8, img0127 9, img0143 10, img0144 11 all
# confirmed; img0051 9, img0065 7, img0080 10, img0140 6 had every box detected) plus img0006, img0017, img0022,
# img0095 and img0104 (all confirmed).
REGION_COUNT: dict[str, int] = {
    "img0001": 8, "img0002": 7, "img0003": 6, "img0004": 6, "img0005": 5, "img0006": 11, "img0007": 8, "img0008": 7,
    "img0009": 6, "img0010": 6, "img0011": 7, "img0012": 10, "img0013": 5, "img0014": 9, "img0015": 7, "img0016": 9,
    "img0017": 4, "img0018": 11, "img0019": 7, "img0020": 6, "img0021": 8, "img0022": 5, "img0023": 9, "img0024":
    10, "img0025": 8, "img0026": 5, "img0027": 7, "img0028": 7, "img0029": 11, "img0030": 8, "img0031": 6,
    "img0032": 7, "img0033": 7, "img0034": 7, "img0035": 8, "img0036": 9, "img0037": 6, "img0038": 9, "img0039": 7,
    "img0040": 12, "img0041": 7, "img0042": 5, "img0043": 5, "img0044": 7, "img0045": 7, "img0046": 6, "img0047": 8,
    "img0048": 10, "img0049": 8, "img0050": 9, "img0051": 9, "img0052": 5, "img0053": 6, "img0054": 6, "img0055": 4,
    "img0056": 8, "img0057": 7, "img0058": 8, "img0059": 10, "img0060": 9, "img0061": 7, "img0062": 5, "img0063": 8,
    "img0064": 7, "img0065": 7, "img0066": 8, "img0067": 8, "img0068": 5, "img0069": 8, "img0070": 6, "img0071": 11,
    "img0072": 6, "img0073": 7, "img0074": 6, "img0075": 8, "img0076": 7, "img0077": 8, "img0078": 9, "img0079": 6,
    "img0080": 10, "img0081": 12, "img0082": 7, "img0083": 4, "img0084": 8, "img0085": 9, "img0086": 9, "img0087":
    7, "img0088": 7, "img0089": 7, "img0090": 8, "img0091": 11, "img0092": 6, "img0093": 7, "img0094": 4, "img0095":
    10, "img0096": 7, "img0097": 11, "img0098": 7, "img0099": 9, "img0100": 10, "img0101": 6, "img0102": 5,
    "img0103": 9, "img0104": 6, "img0105": 8, "img0106": 8, "img0107": 8, "img0108": 6, "img0109": 7, "img0110": 5,
    "img0111": 9, "img0112": 7, "img0113": 10, "img0114": 7, "img0115": 7, "img0116": 9, "img0117": 6, "img0118": 6,
    "img0119": 5, "img0120": 4, "img0122": 7, "img0123": 7, "img0124": 12, "img0125": 8, "img0126": 7, "img0127": 9,
    "img0128": 8, "img0129": 6, "img0130": 11, "img0131": 11, "img0132": 8, "img0133": 5, "img0134": 6, "img0135":
    8, "img0136": 5, "img0137": 6, "img0138": 11, "img0139": 5, "img0140": 6, "img0141": 9, "img0142": 6, "img0143":
    10, "img0144": 11, "img0145": 6, "img0146": 10, "img0147": 5, "img0148": 8, "img0149": 9, "img0150": 5,
    "img0151": 5, "img0152": 11, "img0153": 10, "img0154": 9
}
REGION_COUNT_BY_EYE: dict[str, int] = {"img0121": 13}  # 9 of 13 boxes detected: below the detector's N-3 rule


# ---- region-count detection (single images) -----------------------------------------------------------------------

def detect_boxes(path: Path, work: float = 0.5, min_run: int = 20) -> list[dict]:
    """Hollow, hue-constant rectangles drawn in saturated colours (the region boxes); returns [{box, hue}] at the
    working scale. Hue in OpenCV units (0-180)."""
    import cv2

    bgr = cv2.imread(str(path))
    bgr = cv2.resize(bgr, None, fx=work, fy=work, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(np.int16)
    sat = (hsv[..., 1] >= 150) & (hsv[..., 2] >= 200)  # the palette is S~0.75, V~0.96
    found = []
    for c in range(0, 180, 2):
        d = np.abs(hue - c)
        m = (sat & (np.minimum(d, 180 - d) <= 3)).astype(np.uint8)
        if m.sum() < 4 * min_run:
            continue
        oh = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, min_run), np.uint8))
        ov = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((min_run, 1), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(oh | ov, connectivity=8)
        for i in range(1, n):
            x, y, w, h, _ = st[i]
            if w < min_run or h < min_run:
                continue
            comp = lab[y:y + h, x:x + w] == i
            if not (oh[y:y + h, x:x + w][comp].any() and ov[y:y + h, x:x + w][comp].any()):
                continue
            t = max(3, int(round(min(w, h) * 0.12)))
            band = np.zeros_like(comp)
            band[:t, :] = band[-t:, :] = band[:, :t] = band[:, -t:] = True
            sides = sorted([comp[:t, :].any(axis=0).mean(), comp[-t:, :].any(axis=0).mean(),
                            comp[:, :t].any(axis=1).mean(), comp[:, -t:].any(axis=1).mean()])
            if (comp & band).sum() / comp.sum() < 0.85 or sides[1] < 0.6 or sides[0] < 0.25:
                continue
            found.append({"box": [int(x), int(y), int(w), int(h)], "c": c})
    groups = []  # the same box found at neighbouring hue bins
    for f in found:
        for g in groups:
            if all(abs(a - b) <= (6 if j < 2 else 12) for j, (a, b) in enumerate(zip(f["box"], g["box"]))):
                g["cs"].append(f["c"])
                break
        else:
            groups.append({"box": f["box"], "cs": [f["c"]]})
    for g in groups:
        ang = np.array(g["cs"], float) / 180 * 2 * np.pi
        g["hue"] = round(float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * np.pi) * 180 % 180), 1)
    return groups


def label_height(path: Path, group: dict, work: float = 0.5) -> int:
    """Height (source px) of the box's "Region k" label: same-hue saturated pixels in the band above the box."""
    import cv2

    bgr = cv2.imread(str(path))
    x, y, w, h = [int(round(v / work)) for v in group["box"]]
    top = max(0, y - 120)
    if y - top < 20:
        return 0
    hsv = cv2.cvtColor(bgr[top:y - 4, x:x + max(w, 300)], cv2.COLOR_BGR2HSV)
    d = np.abs(hsv[..., 0].astype(np.int16) - int(round(group["hue"])))
    m = (hsv[..., 1] >= 150) & (hsv[..., 2] >= 200) & (np.minimum(d, 180 - d) <= 4)
    rows = np.where(m.sum(axis=1) >= 2)[0]
    if not len(rows):
        return 0
    # the label is the lowest run of text rows above the box
    runs, start = [], rows[0]
    for a, b in zip(rows[:-1], rows[1:]):
        if b - a > 3:
            runs.append((start, a))
            start = b
    runs.append((start, rows[-1]))
    lo, hi = runs[-1]
    return int(hi - lo + 1)


def hues_fit(hues: list[float], n: int) -> tuple[int, set]:
    step = 180 / n
    tol = min(3.5, step / 3)
    ks, fit = set(), 0
    for h in hues:
        k = round(h / step) % n
        d = abs(h - k * step)
        if min(d, 180 - d) <= tol:
            fit += 1
            ks.add(k)
    return fit, ks


def infer_count(hues: list[float]) -> tuple[int | None, str]:
    """Number of region boxes N from the detected box hues (box k has hue k * 180 / N): the smallest N whose grid
    holds every detected hue (one outlier allowed) and at least N - 3 distinct grid points (a few boxes are
    missed when they overlap other boxes or blend into the scene)."""
    if len(hues) < 2:
        return None, "fewer than 2 boxes detected"
    for n in range(2, 25):
        fit, ks = hues_fit(hues, n)
        if fit >= len(hues) - 1 and len(ks) >= max(2, n - 3):
            return n, f"{len(ks)}/{n} boxes found"
    return None, "hues fit no grid"


def _detect_job(path: str) -> tuple[str, list[dict], list[int]]:
    groups = detect_boxes(SRC / path)
    heights = [label_height(SRC / path, g) for g in groups]
    return path, groups, heights


# ---- items -----------------------------------------------------------------------------------------------------------

def task_of(x: dict) -> str:
    return x["id"].rsplit("_", 1)[0]


def strip_format(q: str) -> str:
    for pat in FORMAT_SENTENCES:
        q = re.sub(pat, "", q)
    return q.strip()


def region_ids(text: str) -> list[int]:
    """Region ids a question names ("Region 3", "Regions 1, 3, and 6", "Region 1 and 6")."""
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"\d+ o'clock", "", text)
    ids = []
    for m in re.finditer(r"Regions?\s+(\d+(?:\s*(?:,\s*and|,|and)\s*\d+)*)", text):
        ids += [int(i) for i in re.findall(r"\d+", m.group(1))]
    return ids


def load_items() -> list[dict]:
    return pq.read_table(SNAP / "annotations_shuffled.parquet").to_pylist()


def referenced_ids(items: list[dict]) -> dict[str, set]:
    ref = defaultdict(set)
    for x in items:
        t = task_of(x)
        if t not in REGION_TASKS:
            continue
        ref[x["image"][0]] |= set(region_ids(x["conversations"][0]["value"]))
        if t != "Direction_Recognition":
            ref[x["image"][0]] |= {int(i) for i in re.findall(r"\d+", x["GT"])}
    return ref


def detect_regions(workers: int) -> None:
    items = load_items()
    ref = referenced_ids(items)
    paths = sorted(ref)
    out, scale = {}, {}
    from PIL import Image

    with mp.get_context("spawn").Pool(workers) as pool:
        for p, groups, heights in pool.imap_unordered(_detect_job, paths):
            out[p] = (groups, heights)
    table, disagree, label_px = {}, [], []
    for p in paths:
        groups, heights = out[p]
        n_ref = max(ref[p]) + 1
        n, why = infer_count([g["hue"] for g in groups])
        table[p] = n
        with Image.open(SRC / p) as im:
            w, h = im.size
        s = min(1.0, (1_638_400 / (w * h)) ** 0.5)
        good = [hh for hh in heights if hh >= 15]
        if good:
            label_px.append((round(min(good) * s, 1), p))
        if n != n_ref:
            disagree.append((p, n_ref, n, why, sorted(g["hue"] for g in groups)))
    print(json.dumps({p.rsplit("/", 1)[-1]: n for p, n in table.items()}))
    print(f"{len(disagree)} images where the boxes and the highest region id the questions use (+1) disagree:")
    for d in disagree:
        print("  ", d[0].rsplit("/", 1)[-1], "n_ref", d[1], "detected", d[2], d[3], d[4])
    label_px.sort()
    print("smallest label text heights after resize (px):", label_px[:8])


# ---- conversion ------------------------------------------------------------------------------------------------------

def _image_job(path: str) -> tuple[str, dict]:
    rel_src = Path(path.removeprefix("./SpatialUAV/"))
    return path, store_fitted(SRC / "SpatialUAV" / rel_src, OUT / "images" / rel_src.with_suffix(""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--detect-regions", action="store_true")
    ap.add_argument("--images-only", action="store_true", help="render the images and stop")
    args = ap.parse_args()
    if args.detect_regions:
        detect_regions(args.workers)
        return
    counts = {f"./SpatialUAV/samples_Single_Image/{k}.jpg": v for k, v in {**REGION_COUNT, **REGION_COUNT_BY_EYE}.items()}
    items = load_items()
    dropped, rows, need = Counter(), [], set()
    kept_items = []
    for x in items:
        t = task_of(x)
        if t not in KEPT:
            dropped[f"task not closed-form: {t}"] += 1
            continue
        kept_items.append(x)
        need.update(x["image"])
    t0 = time.time()
    infos = {}
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=100) as pool:
        for i, (p, info) in enumerate(pool.imap_unordered(_image_job, sorted(need), chunksize=4), 1):
            infos[p] = info
            if i % 500 == 0 or i == len(need):
                print(f"[images {i}/{len(need)}] {time.time() - t0:.0f}s", flush=True)
    if args.images_only:
        return

    for x in kept_items:
        t = task_of(x)
        q_pub = x["conversations"][0]["value"]
        q = strip_format(q_pub)
        gt = x["GT"].strip()
        img0 = x["image"][0]
        family = t.lower()
        questions, answer = None, None
        if t in ("A2A_Collaboration_Recognition", "A2G_Collaboration_Recognition"):
            if len(x["image"]) != 5 or gt not in "ABCD" or len(gt) != 1:
                raise ValueError(f"{x['id']}: {len(x['image'])} images, GT {gt!r}")
            opts = [f"Image {i}" for i in range(2, 6)]
            questions = {"q": {"type": "choice", "instructions": q, "criteria": {o: None for o in opts}}}
            answer = {"q": opts["ABCD".index(gt)]}
        elif t == "A2G_Path_Planning":
            m = re.fullmatch(r"Direction ([1-6])", gt)
            opts = [f"Direction {i}" for i in range(1, 7)]
            questions = {"q": {"type": "choice", "instructions": q, "criteria": {o: None for o in opts}}}
            answer = {"q": f"Direction {m.group(1)}"}
        else:
            if img0 in MISASSIGNED_IMAGES:
                dropped["img0066 (question set duplicated from img0077)"] += 1
                continue
            n = counts[img0]
            named = region_ids(q_pub)
            if re.search(r"Region\s+of\b", q_pub) or (t == "Direction_Recognition" and not named):
                dropped[f"{t}: question names no region"] += 1
                continue
            if t == "Direction_Recognition":
                m = re.fullmatch(r"(\d{1,2}) o'clock\.?", gt)
                if not m:
                    dropped[f"{t}: malformed GT {gt!r}"] += 1
                    continue
                if any(i >= n for i in named):
                    dropped[f"{t}: names a region the image does not have"] += 1
                    continue
                opts = [f"{i} o'clock" for i in range(1, 13)]
                questions = {"q": {"type": "choice", "instructions": q, "criteria": {o: None for o in opts}}}
                answer = {"q": f"{int(m.group(1))} o'clock"}
            else:
                golds = [int(i) for i in re.findall(r"\d+", gt)]
                if not golds:
                    dropped[f"{t}: malformed GT {gt!r}"] += 1
                    continue
                if any(i >= n for i in golds + named):
                    dropped[f"{t}: names / answers a region the image does not have"] += 1
                    continue
                labels = [f"Region {i}" for i in range(n)]
                if t == "Distance_Comparison":
                    opts = [f"Region {i}" for i in named] if named else labels
                    if len(golds) != 1 or f"Region {golds[0]}" not in opts:
                        dropped[f"{t}: GT not among the named regions"] += 1
                        continue
                    questions = {"q": {"type": "choice", "instructions": q, "criteria": {o: None for o in opts}}}
                    answer = {"q": f"Region {golds[0]}"}
                elif len(set(golds)) == 1:
                    questions = {"q": {"type": "choice", "instructions": q, "criteria": {o: None for o in labels}}}
                    answer = {"q": f"Region {golds[0]}"}
                else:
                    family += "_multi"
                    questions = {f"r{i}": {"type": "noul", "instructions": f"{q} Is Region {i} one of them?"}
                                 for i in range(n)}
                    answer = {f"r{i}": i in golds for i in range(n)}
        images = [infos[p]["path"] for p in x["image"]]
        row = {
            "id": f"SpatialUAV:{x['id']}",
            "group_id": f"SpatialUAV:{img0}",
            "benchmark": "SpatialUAV",
            "split": "test",
            "task_family": family,
            "skills": ["uav", "spatial"],
            "images": images,
            "state": "",
            "questions": questions,
            "answer": answer,
            "meta": {"task": t, "source_id": x["id"], "published_question": q_pub, "published_gt": x["GT"],
                     "sources": x["image"], "region_count": counts.get(img0) if t in REGION_TASKS else None,
                     "images": [{k: infos[p][k] for k in ("orig_px", "px", "scale")} for p in x["image"]]},
        }
        validate_row(row)
        rows.append(row)
    n = write_jsonl(OUT / "spatialuav_closed.jsonl", rows)
    nq = Counter((r["task_family"], q["type"]) for r in rows for q in r["questions"].values())
    print(f"SpatialUAV: {n} rows -> {OUT / 'spatialuav_closed.jsonl'}")
    print("rows per family:", json.dumps(Counter(r["task_family"] for r in rows)))
    print("questions per (family, type):", {f"{a}/{b}": c for (a, b), c in sorted(nq.items())})
    print("options per choice:", dict(Counter((r["task_family"], len(r["questions"]["q"]["criteria"]))
                                             for r in rows if "q" in r["questions"])))
    print("dropped:", json.dumps(dropped, indent=0))
    print("images:", len(infos), "resized:", sum(1 for i in infos.values() if i["scale"] < 1),
          "upscaled:", sum(1 for i in infos.values() if i["scale"] > 1),
          "scales:", dict(Counter(round(i["scale"], 2) for i in infos.values()).most_common(8)))
    print("gold position / yes rate per family:", json.dumps(answer_balance(rows)))
    show_samples(rows)


if __name__ == "__main__":
    main()
