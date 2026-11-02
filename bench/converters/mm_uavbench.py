"""C28 MM-UAVBench (arXiv 2512.23219; 5,702 MCQ over 19 tasks on low-altitude UAV images and videos) -> rows.

Source: HF `daisq/MM-UAVBench` @ 99db2b961b72998f8d9710926c25169b3c606823 (Apache-2.0): `tasks/<Task>.json` (19 files)
and the evaluation media `images/annotated/` (9,792 JPEGs with the published boxes / points / labels already drawn,
and 791 MP4s). Data types: single_image 3,125, multi_images 960 (2-8 images), video_frames 826 (1-8 frames given as
JPEGs), video 791 (Event Prediction / Tracing / Understanding: 640x640 ERA clips, ~24 fps, 4.9-6.4 s). 2-6 options
(4: 4,533; 5: 773; 3: 337; 2: 37; 6: 22). The published headline is the macro mean over the 19 tasks, so
`task_family` is the task (file) name in lower case; meta.category is Perception / Cognition / Planning (the release's
"Congnition" typo, 207 Scene Analysis items, is merged into Cognition).

ADAPTED (resolution; video frames as in the official VLMEvalKit config); report it as "not the published protocol":
  - 2,846 of the 9,792 images (29%; up to 20 MP) are above 1,638,400 px and are downscaled (Lanczos, JPEG q95);
    31 images under 65,536 px are upscaled (bicubic). Factors are in meta.images. All other images keep their bytes.
  - The published annotations are drawn on the source pixels (red boxes 2-6 px wide, point markers, 10 px labels)
    and would thin to ~1 px when downscaled, so on every downscaled image that carries them (482 images) they are
    re-drawn after resizing at the scaled positions, in the same colour, with the published width scaled but at
    least 2 px (points and labels at their published pixel size); meta.images[i].restroked counts them.
  - Videos -> frames exactly as the official VLMEvalKit dataset (`mmuavbench.py`, MMUAVBench_Video, fps=3.0):
    n = int(n_frames / fps_video * 3) frames at indices int(i * fps_video / 3), i.e. 3 frames per second (14-19
    frames, 17 for most clips), kept at 640x640 (JPEG q95). Frames are decoded with OpenCV (decord is not installed);
    n_frames is the exact decoded count. The state says what the images are, as the official video prompt does.
Kept as published: all 5,702 items, the question text, option texts and their order. Options that are empty
strings (10 Cross-Object Reasoning items have an empty "D") are dropped, as VLMEvalKit's image prompt skips them.
28 items repeat an option text (e.g. two "Soccer field"): there every option text gets its published letter
appended ("Soccer field (A)") so the keys stay distinct (meta.duplicate_options). Images are in the published order.
Not kept: the published system text ("You are an expert in the field of drones ..."); the runners use their common
lettered-option prompt.

    MODULE=bench.converters.mm_uavbench sbatch slurm/convert_cpu.sbatch     (or: python -m bench.converters.mm_uavbench)
    (resumable: images / frames already written are skipped)
"""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from bench.common import CONVERTED, DATASETS, validate_row, write_jsonl
from bench.converters.uav_common import (answer_balance, atomic_save, count_frames, read_frames, rel, show_samples,
                                         store_fitted, video_info)

REV = "99db2b961b72998f8d9710926c25169b3c606823"
SNAP = (Path(os.environ.get("EOJEV_DATASET_HUB", DATASETS / "hf_hub")) / "datasets--daisq--MM-UAVBench" / "snapshots"
        / REV)
OUT = CONVERTED / "mm_uavbench"
FPS = 3.0  # VLMEvalKit MMUAVBench_Video default
# tools/render_annotated.py: tasks whose labels are drawn (the others draw boxes / points only)
LABELLED = {"Ground_Target_Planning", "Air_Ground_Collaborative_Planning", "Swarm_Collaborative_Planning"}
RED = (255, 0, 0)
FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"]


# ---- re-drawing the published annotations after downscaling (tools/util.py's Visualize, scaled) --------------------

def _font(size: int):
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _box_width(box, size) -> int:
    """tools/util.py _adjust_draw_width, on the source image size."""
    x0, y0, x1, y1 = box
    ratio = max(1, x1 - x0) * max(1, y1 - y0) / (size[0] * size[1])
    return 6 if ratio >= 0.05 else 4 if ratio >= 0.01 else 3 if ratio >= 0.005 else 2


def restroke(image, scale, entities, labelled, orig_size):
    """Draw the published boxes / points again on the resized image (same colour, legible width)."""
    draw = ImageDraw.Draw(image)
    font = _font(10)
    for e in entities:
        kind, c = e.get("entity_type", "region"), e.get("bbox", [])
        if kind in ("region", "object", "human") and len(c) == 4:
            x0, y0, x1, y1 = [int(v) for v in c]
            if x0 >= x1 or y0 >= y1:
                continue
            w = max(2, round(_box_width(c, orig_size) * scale))
            b = [round(x0 * scale), round(y0 * scale), round(x1 * scale), round(y1 * scale)]
            draw.rectangle(b, outline=RED, width=w)
            if labelled and e.get("label"):
                tb = draw.textbbox((b[0], b[1]), e["label"], font)
                th = tb[3] - tb[1]
                y0t, y1t = b[1] - th - 2 * w, b[1] + w
                if y0t < 0:
                    y0t, y1t = 0, b[1] + 2 * w + th
                draw.rectangle([b[0], y0t, tb[2] + 2 * w, y1t], fill=RED)
                draw.text((b[0] + w, y0t), e["label"], fill="black", font=font)
        elif kind == "point" and len(c) == 2:
            x, y = round(int(c[0]) * scale), round(int(c[1]) * scale)
            r, bw = 1, 2  # util.py: radius min(8, 1), white border 2
            draw.ellipse([x - r - bw, y - r - bw, x + r + bw, y + r + bw], fill="white", outline="white")
            draw.ellipse([x - r, y - r, x + r, y + r], fill=RED, outline=RED)
            if labelled and e.get("label"):
                lx, ly = x + 5, y - 5
                if ly < 0:
                    ly = y + 15
                tb = draw.textbbox((lx, ly), e["label"], font)
                draw.text((lx, ly - (tb[3] - tb[1])), e["label"], fill="red", font=font)
    return image


def _entities(item: dict, res: dict) -> list[dict]:
    if item["metadata"]["data_type"] == "single_image":
        return item["target_entities"]
    return [e for e in item["target_entities"] if e.get("index") == res["index"]]


# ---- media jobs ----------------------------------------------------------------------------------------------------

def _image_job(job: tuple) -> tuple[str, dict]:
    path, entities, labelled = job
    src = SNAP / path
    with Image.open(src) as im:
        size = im.size
    overlay = None
    if entities and size[0] * size[1] > 1_638_400:
        overlay = functools.partial(restroke, entities=entities, labelled=labelled, orig_size=size)
    stem = OUT / "images" / Path(path).relative_to("images/annotated").with_suffix("")
    info = store_fitted(src, stem, overlay=overlay)
    if overlay is not None:
        info["restroked"] = sum(1 for e in entities if len(e.get("bbox", [])) in (2, 4))
    return path, info


def frame_indices(n_frames: int, fps: float) -> list[int]:
    """VLMEvalKit MMUAVBench_Video.save_video_frames: required = int(duration * 3), index = int(i * fps / 3)."""
    required = int(n_frames / fps * FPS)
    step = fps / FPS
    return [int(i * step) for i in range(required)]


def _video_job(job: tuple) -> tuple[str, dict]:
    (path,) = job
    src = SNAP / path
    vdir = OUT / "frames" / Path(path).relative_to("images/annotated").with_suffix("")
    index = vdir / "_frames.json"
    if index.exists():
        return path, json.loads(index.read_text())
    vi = video_info(src)
    n = count_frames(src)
    idx = frame_indices(n, vi["fps"])
    frames, _ = read_frames(src, idx)
    missing = [i for i in idx if i not in frames]
    if missing:
        raise RuntimeError(f"{path}: frames {missing} not decoded")
    files = []
    for k, i in enumerate(idx):
        f = vdir / f"f{k:02d}.jpg"
        if not f.exists():
            atomic_save(frames[i], f)
        files.append(rel(f))
    info = {"fps": vi["fps"], "n_frames": n, "px": [vi["width"], vi["height"]], "indices": idx,
            "times_s": [round(i / vi["fps"], 3) for i in idx], "files": files}
    tmp = index.with_name(".tmp_frames.json")
    tmp.write_text(json.dumps(info))
    os.replace(tmp, index)
    return path, info


def run_jobs(fn, jobs: list, workers: int, label: str) -> dict:
    out, t0 = {}, time.time()
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=200) as pool:
        for i, (key, info) in enumerate(pool.imap_unordered(fn, jobs, chunksize=4), 1):
            out[key] = info
            if i % 500 == 0 or i == len(jobs):
                print(f"[{label} {i}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
    return out


# ---- rows ----------------------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    items = []
    for f in sorted((SNAP / "tasks").glob("*.json")):
        for x in json.load(open(f)):
            items.append((f.stem, x))
    image_jobs, video_jobs = [], []
    for task, x in items:
        for res in x["metadata"]["data_resources"]:
            if res["path"].endswith(".mp4"):
                video_jobs.append((res["path"],))
            else:
                image_jobs.append((res["path"], _entities(x, res), task in LABELLED))
    print(f"MM-UAVBench: {len(items)} items, {len(image_jobs)} images, {len(video_jobs)} videos", flush=True)
    imgs = run_jobs(_image_job, image_jobs, args.workers, "images")
    vids = run_jobs(_video_job, video_jobs, args.workers, "videos")

    rows, stats = [], Counter()
    for task, x in items:
        published = {k: v for k, v in x["options"].items()}
        letters = [l for l, v in published.items() if str(v).strip() != ""]
        if len(letters) != len(published):
            stats["items_with_empty_option_dropped"] += 1
        texts = [str(published[l]).strip() for l in letters]
        dup = len(set(texts)) != len(texts)
        if dup:
            texts = [f"{t} ({l})" for l, t in zip(letters, texts)]
            stats["duplicate_options"] += 1
        gold = texts[letters.index(x["answer"])]
        dtype = x["metadata"]["data_type"]
        media, state = [], ""
        if dtype == "video":
            v = vids[x["metadata"]["data_resources"][0]["path"]]
            images = v["files"]
            state = (f"The {len(images)} images are separate frames uniformly sampled from a video (3 frames per "
                     "second), in chronological order of the video.")
            media.append({k: v[k] for k in ("fps", "n_frames", "px", "indices", "times_s")})
        else:
            images = []
            for res in x["metadata"]["data_resources"]:
                info = imgs[res["path"]]
                images.append(info["path"])
                media.append({k: info[k] for k in ("orig_px", "px", "scale")} | (
                    {"restroked": info["restroked"]} if "restroked" in info else {}))
                stats["image_resized_down" if info["scale"] < 1 else "image_upscaled" if info["scale"] > 1
                      else "image_native"] += 1
        cat = x["question_meta"]["category"].replace("Congnition", "Cognition")
        row = {
            "id": f"MMUAV:{task}:{x['question_id']}",
            "group_id": f"MMUAV:{task}:{x['question_id']}",
            "benchmark": "MM-UAVBench",
            "split": "test",
            "task_family": task.lower(),
            "skills": ["uav", cat.lower()],
            "images": images,
            "state": state,
            "questions": {"q": {"type": "choice", "instructions": x["question"].strip(),
                                "criteria": {t: None for t in texts}}},
            "answer": {"q": gold},
            "meta": {"task": task, "subcategory": x["question_meta"]["subcategory"], "category": cat,
                     "question_id": x["question_id"], "data_type": dtype, "data_source": x["metadata"]["data_source"],
                     "sources": [r["path"] for r in x["metadata"]["data_resources"]],
                     "published_options": published, "gold_letter": x["answer"], "duplicate_options": dup,
                     "media": media},
        }
        validate_row(row)
        rows.append(row)
    n = write_jsonl(OUT / "mm_uavbench.jsonl", rows)
    print(f"MM-UAVBench: {n} rows -> {OUT / 'mm_uavbench.jsonl'}")
    print("task families:", json.dumps(Counter(r["task_family"] for r in rows), indent=0))
    print("data types:", dict(Counter(r["meta"]["data_type"] for r in rows)))
    print("options per question:", dict(Counter(len(r["questions"]["q"]["criteria"]) for r in rows)))
    print("stats:", dict(stats), "| restroked images:", sum(1 for i in imgs.values() if "restroked" in i),
          "| frames:", sum(len(v["files"]) for v in vids.values()),
          "frames per video:", dict(Counter(len(v["files"]) for v in vids.values())))
    print("gold position per family:", json.dumps(answer_balance(rows)))
    show_samples(rows)


if __name__ == "__main__":
    main()
