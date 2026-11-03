"""C18 detection counting on NWPU VHR-10 and DOTA-v2.0 val, following RSThinker's counting protocol -> rows. ADAPTED.

RSThinker (ICLR'26, arXiv 2509.22221) did not release its counting test items (RS_Tester's OC_VHR / OC_DOTA question
files are not public), so we rebuild them the way its released code and training data do it:
  - Question = one class per question, only classes PRESENT in the image: none of the 25,769 DOTA-v2 and 24,784
    HRRSD counting questions in its released SFT set (HF minglanga/GeoCoT380k @ 0f555754) has the answer 0, and an
    image gets one question per present class (up to 7).
  - Wording: one of five templates, "How many {c} are there in the image?", "What is the number of {c} in the
    image?", "What is the amount of {c} in the image?", "Count how many {c} are in the image.", "What is the count of
    {c} in the image?", followed in RSThinker by "\\nAnswer the question using a single word or phrase." Class names
    are used raw: DOTA's hyphenated labels ("small-vehicle"); for NWPU VHR-10 the HRRSD-style names RSThinker was
    trained on ("storage tank", "ground track field", ...), which equal VHR-10's class names.
  - DOTA images are cut into non-overlapping 800 x 800 tiles (row-major, stride 800, edge tiles kept at their
    smaller size, not padded; its files are named P0000_<i>_<j>_<x0>_<y0>_<x1>_<y1>.png), and counting is per tile.
    The object-to-tile rule is not released: we assign an object to the tile holding its polygon centroid and count
    every annotated object, difficult ones included (RSThinker's DOTA-train answers reach p99 = 417 small vehicles
    per tile, which matches all-object counts, not the non-difficult ones). meta also has the non-difficult count.
  - Metrics (RS_Tester metrics.py OCAccuracy): exact accuracy, accuracy within 10 / 30 / 50% of the true count, and
    MAE. Its paper's Table 5 sizes, inferred from the reported accuracies: DOTAv2-val 453 questions, NWPU-VHR 200.
Sources:
  - NWPU VHR-10 (CC-BY-NC-4.0): HF torchgeo/vhr10 @ 2cfe037d "NWPU VHR-10 dataset.zip" (the original RAR's files;
    650 positive images with "(x1,y1),(x2,y2),class" ground truth, 150 negative images).
  - DOTA-v2.0 val (academic, no redistribution): images = the 458 DOTA-v1.0 val PNGs in raw/dota-v2-val/
    dota-v1.0-val/part1.zip (user decision 2026-10-21: the 135 v2-only val images, Baidu-only, are not used); labels
    = official DOTA-v2.0_val.zip (OBB, "x1 y1 ... x4 y4 class difficult").
Our version (adapted; we run RSThinker ourselves on it):
  - det_count_nwpu_vhr10.jsonl: every (positive image, present class) pair: 888 questions on 650 images, one row
    per image. 3 images above MAX_PIXELS are downscaled (Lanczos; factor in meta), the rest keep their JPEG bytes.
  - det_count_dota_v2_val.jsonl: tiles with a side < 256 px are dropped; of the 4,798 (tile, present class) pairs
    a seeded (seed 0) class-proportional sample of 1,500, one row per tile. Tiles are 800 x 800 at native
    resolution (no downscale, so small objects stay as visible as in RSThinker's protocol); saved as PNG.
  - Answers: a `choice` over "0" .. "K" (+ "more than K" when some count of the class exceeds K), K fixed per
    dataset and class = the 99th percentile of that class's counts over all candidate pairs, clamped to [10, 100]
    (the floor keeps one-per-image classes such as ground track field from being trivial). The exact count is in
    meta.count, the option values in meta.option_values ("more than K" = mean of that class's counts above K) for
    expected-value MAE. The template is drawn per question with random.Random(question id); `instructions` omits
    RSThinker's answer-format suffix (the runners add the options), meta.native_prompt has the full RSThinker text.
  - det_count_vhr10_absent.jsonl (EXTENSION, not RSThinker's protocol): zero-count questions: one random VHR-10
    class for each of the 150 negative images plus one random absent class for 150 seeded positive images (answer
    "0", same options as the main file), for a hallucinated-count check. Report it separately.

    python -m bench.converters.det_count          (long: run as MODULE=bench.converters.det_count sbatch slurm/convert_cpu.sbatch)
"""

from __future__ import annotations

import io
import math
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from bench.common import CONVERTED, DATASETS, RAW, save_png, validate_row, write_jsonl
from bench.converters.perception_utils import fit, sample_text, store_fitted, stratified, summary

Image.MAX_IMAGE_PIXELS = None
OUT = CONVERTED / "det_count"
VHR_ZIP = (DATASETS / "hf_hub" / "datasets--torchgeo--vhr10" / "snapshots" / "2cfe037d20f68acb4a10f44b7cc1dc4324ebf50b"
           / "NWPU VHR-10 dataset.zip")
DOTA_IMAGES = RAW / "dota-v2-val" / "dota-v1.0-val" / "part1.zip"
DOTA_LABELS = RAW / "dota-v2-val" / "labelTxt-v2.0" / "DOTA-v2.0_val.zip"
VHR_CLASSES = {1: "airplane", 2: "ship", 3: "storage tank", 4: "baseball diamond", 5: "tennis court",
               6: "basketball court", 7: "ground track field", 8: "harbor", 9: "bridge", 10: "vehicle"}
TEMPLATES = ["How many {c} are there in the image?", "What is the number of {c} in the image?",
             "What is the amount of {c} in the image?", "Count how many {c} are in the image.",
             "What is the count of {c} in the image?"]
SUFFIX = "\nAnswer the question using a single word or phrase."
TILE, MIN_TILE_SIDE, N_DOTA, K_MIN, K_MAX = 800, 256, 1500, 10, 100


def options_for(counts: list[int]) -> tuple[list[str], list[float], int]:
    """Options "0".."K" (+ "more than K") for a class from all its candidate counts; also numeric option values."""
    v = sorted(counts)
    p99 = v[min(len(v) - 1, int(0.99 * len(v)))]
    k = min(K_MAX, max(K_MIN, p99))
    opts = [str(i) for i in range(k + 1)]
    values = [float(i) for i in range(k + 1)]
    above = [c for c in v if c > k]
    if above:
        opts.append(f"more than {k}")
        values.append(round(sum(above) / len(above), 1))
    return opts, values, k


def answer_option(count: int, opts: list[str]) -> str:
    return str(count) if str(count) in opts else opts[-1]


def question(row_id: str, cls: str, count: int, opts: list[str], values: list[float], k: int, extra: dict) -> tuple[dict, str, dict]:
    qid = cls.replace(" ", "_")
    text = random.Random(f"{row_id}:{qid}").choice(TEMPLATES).format(c=cls)
    q = {"type": "choice", "instructions": text, "criteria": {o: None for o in opts}}
    meta = {"class": cls, "count": count, "k": k, "option_values": values, "native_prompt": text + SUFFIX, **extra}
    return q, answer_option(count, opts), meta


# ------------------------------------------------------------------------------------------------ NWPU VHR-10
def vhr10() -> tuple[list[dict], list[dict], Counter]:
    stats = Counter()
    with zipfile.ZipFile(VHR_ZIP) as zf:
        names = zf.namelist()
        gt = {}
        for n in sorted(names):
            if "/ground truth/" in n and n.endswith(".txt"):
                cls = [int(m) for m in re.findall(r"\),\s*(\d+)\s*$", zf.read(n).decode(), re.M)]
                gt[Path(n).stem] = Counter(VHR_CLASSES[c] for c in cls)
        positives = sorted(Path(n).stem for n in names if "/positive image set/" in n and n.endswith(".jpg"))
        negatives = sorted(Path(n).stem for n in names if "/negative image set/" in n and n.endswith(".jpg"))
        if set(positives) != set(gt):
            raise ValueError("positive images and ground-truth files differ")
        per_class: dict[str, list[int]] = defaultdict(list)
        for c in gt.values():
            for cls, v in c.items():
                per_class[cls].append(v)
        opts = {cls: options_for(v) for cls, v in per_class.items()}

        seen: dict[tuple[str, str], dict] = {}

        def image(stem: str, folder: str) -> dict:
            if (stem, folder) not in seen:
                info = store_fitted(zf.read(f"NWPU VHR-10 dataset/{folder}/{stem}.jpg"),
                                    OUT / "images" / "vhr10" / folder.split()[0] / stem)
                stats["images"] += 1
                stats["downscaled"] += info["resize_factor"] < 1
                stats["upscaled"] += info["resize_factor"] > 1
                seen[(stem, folder)] = info
            return seen[(stem, folder)]

        rows = []
        for stem in positives:
            info = image(stem, "positive image set")
            row_id = f"DetCount-VHR10:{stem}"
            questions, answers, qmeta = {}, {}, {}
            for cls in sorted(gt[stem]):
                o, vals, k = opts[cls]
                q, a, m = question(row_id, cls, gt[stem][cls], o, vals, k, {})
                qid = cls.replace(" ", "_")
                questions[qid], answers[qid], qmeta[qid] = q, a, m
            rows.append({
                "id": row_id, "group_id": row_id, "benchmark": "DetCount-NWPU-VHR10", "split": "test",
                "task_family": "det_count", "skills": ["counting", "detection"], "images": [info["path"]], "state": "",
                "questions": questions, "answer": answers,
                "meta": {"source": "NWPU VHR-10 (torchgeo/vhr10@2cfe037d zip of the original RAR)", "image": f"{stem}.jpg",
                         "orig_px": info["orig_px"], "px": info["px"], "resize_factor": info["resize_factor"],
                         "questions": qmeta, "protocol": "RSThinker counting (present classes only)"},
            })

        # extension: zero-count questions (negatives + one absent class on 150 seeded positives)
        absent = []
        rng = random.Random(0)
        classes = sorted(VHR_CLASSES.values())
        for stem in negatives:
            info = image(stem, "negative image set")
            absent.append((f"DetCount-VHR10-absent:neg{stem}", stem, "negative", info, rng.choice(classes)))
        for stem in sorted(rng.sample(positives, 150)):
            info = image(stem, "positive image set")
            missing = [c for c in classes if c not in gt[stem]]
            absent.append((f"DetCount-VHR10-absent:pos{stem}", stem, "positive", info, rng.choice(missing)))
        absent_rows = []
        for row_id, stem, kind, info, cls in absent:
            o, vals, k = opts[cls]
            q, a, m = question(row_id, cls, 0, o, vals, k, {"extension": "absent class (not RSThinker's protocol)"})
            qid = cls.replace(" ", "_")
            absent_rows.append({
                "id": row_id, "group_id": f"DetCount-VHR10:{kind}:{stem}", "benchmark": "DetCount-NWPU-VHR10-absent",
                "split": "test", "task_family": "det_count_absent", "skills": ["counting", "hallucination"],
                "images": [info["path"]], "state": "", "questions": {qid: q}, "answer": {qid: a},
                "meta": {"source": "NWPU VHR-10", "image": f"{stem}.jpg", "image_set": kind, "orig_px": info["orig_px"],
                         "px": info["px"], "resize_factor": info["resize_factor"], "questions": {qid: m}},
            })
    stats["options_K"] = {c: o[2] for c, o in sorted(opts.items())}
    return rows, absent_rows, stats


# ------------------------------------------------------------------------------------------------ DOTA-v2 val
def dota() -> tuple[list[dict], Counter]:
    stats = Counter()
    zi, zl = zipfile.ZipFile(DOTA_IMAGES), zipfile.ZipFile(DOTA_LABELS)
    members = {Path(n).stem: n for n in zi.namelist() if n.endswith(".png")}
    pairs, per_class = [], defaultdict(list)
    for pid in sorted(members):
        with zi.open(members[pid]) as f:
            w, h = Image.open(f).size
        objs = []
        for line in zl.read(pid + ".txt").decode().splitlines():
            p = line.split()
            if len(p) < 10:
                continue
            xs, ys = [float(v) for v in p[0:8:2]], [float(v) for v in p[1:8:2]]
            objs.append((sum(xs) / 4, sum(ys) / 4, p[8], p[9] == "1"))
        for ty in range(0, h, TILE):
            for tx in range(0, w, TILE):
                tw, th = min(TILE, w - tx), min(TILE, h - ty)
                stats["tiles"] += 1
                if min(tw, th) < MIN_TILE_SIDE:
                    stats["tiles dropped (side < 256 px)"] += 1
                    continue
                inside = [o for o in objs if tx <= o[0] < tx + TILE and ty <= o[1] < ty + TILE]
                all_c = Counter(o[2] for o in inside)
                easy_c = Counter(o[2] for o in inside if not o[3])
                for cls, n in all_c.items():
                    tile = f"{pid}_{tx}_{ty}"
                    pairs.append({"tile": tile, "pid": pid, "box": (tx, ty, tx + tw, ty + th), "img": (w, h),
                                  "class": cls, "count": n, "count_nondifficult": easy_c.get(cls, 0)})
                    per_class[cls].append(n)
    stats["candidate pairs"] = len(pairs)
    opts = {cls: options_for(v) for cls, v in per_class.items()}
    picked = stratified(pairs, "class", N_DOTA, random.Random(0))
    by_tile: dict[str, list[dict]] = defaultdict(list)
    for p in picked:
        by_tile[p["tile"]].append(p)

    rows, cache_pid, cache_img = [], None, None
    for tile in sorted(by_tile, key=lambda t: (by_tile[t][0]["pid"], by_tile[t][0]["box"][1], by_tile[t][0]["box"][0])):
        ps = sorted(by_tile[tile], key=lambda p: p["class"])
        pid, box = ps[0]["pid"], ps[0]["box"]
        dest = OUT / "images" / "dota" / f"{tile}.png"
        if not dest.exists():
            if cache_pid != pid:
                with zi.open(members[pid]) as f:
                    cache_img = Image.open(io.BytesIO(f.read())).convert("RGB")
                cache_pid = pid
            crop, factor = fit(cache_img.crop(box))
            if factor != 1.0:
                raise ValueError(f"{tile}: tile outside the pixel limits")
            save_png(crop, dest)
            stats["tiles written this run"] += 1
        row_id = f"DetCount-DOTA:{tile}"
        questions, answers, qmeta = {}, {}, {}
        for p in ps:
            o, vals, k = opts[p["class"]]
            q, a, m = question(row_id, p["class"], p["count"], o, vals, k,
                               {"count_nondifficult": p["count_nondifficult"]})
            qid = p["class"]
            questions[qid], answers[qid], qmeta[qid] = q, a, m
        rows.append({
            "id": row_id, "group_id": f"DetCount-DOTA:{pid}", "benchmark": "DetCount-DOTA-v2-val", "split": "test",
            "task_family": "det_count", "skills": ["counting", "detection"],
            "images": [str(dest.relative_to(CONVERTED))], "state": "", "questions": questions, "answer": answers,
            "meta": {"source": "DOTA-v1.0 val images (part1.zip) + DOTA-v2.0 val labels", "image": f"{pid}.png",
                     "orig_px": list(ps[0]["img"]), "tile_box_xyxy": list(box),
                     "px": [box[2] - box[0], box[3] - box[1]], "resize_factor": 1.0,
                     "questions": qmeta, "protocol": "RSThinker counting: 800 px tiles, present classes only, "
                                                     "objects assigned by polygon centroid, difficult included"},
        })
    stats["options_K"] = {c: o[2] for c, o in sorted(opts.items())}
    return rows, stats


def describe(name: str, rows: list[dict]) -> None:
    print(summary(name, rows))
    ans = Counter()
    over = Counter()
    for r in rows:
        for qid, m in r["meta"]["questions"].items():
            ans[min(m["count"], 11)] += 1
            over[r["answer"][qid].startswith("more")] += 1
    print("  exact-count histogram (11 = 11+):", dict(sorted(ans.items())), "| answers 'more than K':", over[True])
    print("  questions per class:", dict(Counter(m["class"] for r in rows for m in r["meta"]["questions"].values()).most_common()))


def main() -> None:
    vrows, arows, vstats = vhr10()
    write_jsonl(OUT / "det_count_nwpu_vhr10.jsonl", [validate_row(r) or r for r in vrows])
    write_jsonl(OUT / "det_count_vhr10_absent.jsonl", [validate_row(r) or r for r in arows])
    drows, dstats = dota()
    write_jsonl(OUT / "det_count_dota_v2_val.jsonl", [validate_row(r) or r for r in drows])
    describe("DetCount NWPU VHR-10", vrows)
    print("  VHR-10 images / K per class:", dict(vstats))
    describe("DetCount VHR-10 absent (extension)", arows)
    describe("DetCount DOTA-v2 val", drows)
    print("  DOTA tiles / K per class:", dict(dstats))
    print(sample_text(vrows, 1, seed=2))
    print(sample_text(drows, 2, seed=2))


if __name__ == "__main__":
    main()
