"""C30 SIS-Bench ("Self in Space", arXiv 2607.12477; 4,856 four-option MCQ on 1,601 drone videos) -> rows.

Source: HF `choucsan/SIS-Bench` @ da50acd6e80cf27413caa27cf90aaf9a6d04e16b (Apache-2.0; the AirScape, UrbanVideo-Bench
and VisDrone videos keep their own terms): `SIS-Bench.jsonl` and `video/` (AirScape 1,156, UrbanVideo 397,
VisDrone 48 referenced videos; mostly 1280x720 at 24 or 15 fps, 1-359 s, median 10.8 s). 13 tasks
(`task_family` = the published `task_type`): action_recall 804, action_recognition 686, object_existence 492,
landmark_recall 443, object_attribute 387, action_sequence 315, landmark_appearance_order 306, path_planning 272,
action_prediction 263, positional_relationship 252, spatio-temporal_consistency 241, relative_direction 200,
spatial_consistency 195. Memory tasks use videos concatenated from 2-4 clips (meta.concat_num).

ADAPTED (video -> frames); report it as "not the published protocol". The frames copy the official evaluation code
(github.com/IntelliSensing/Self-in-Space, eval/src/uav/vqa/uav_dataset.py, run with eval/run_eval.sh defaults):
  - at most 32 frames per video: when the container reports more than 32 frames, indices
    np.linspace(0, n_frames - 1, 32, dtype=int) over the reported frame count, decoded in order; otherwise every
    frame. (The paper's text describes fps-based sampling - 2 fps under 16 s, 1 fps for 16-32 s, else 32 uniform -
    but the released code samples 32 uniform frames for every video longer than 32 frames, so that is used.)
  - each frame resized into [256*28*28, 512*28*28] = [200,704, 401,408] px (resize_to_pixel_range: bicubic, sizes
    truncated; frames already in range are kept), e.g. 1280x720 -> 844x475.
  - frames are stored once per video as JPEG (q95) and shared by all questions on that video; the state gives the
    frame timestamps (index / fps) so "first clip" / "at the end" questions can be placed in time.
  Decoding uses OpenCV (PyAV is not installed); OpenCV's container frame count stands in for PyAV's stream.frames.
Kept as published: the question text, the four options and their order A-D. 57 items repeat an option text (e.g.
two identical "Moving forward while pitching down"; in some the gold is one of the twins): there every option text
gets its published letter appended ("... (A)") so the keys stay distinct (meta.duplicate_options), which keeps the
published letter scoring.
Not kept: the published prompt template ("Watch the video carefully ... Only output the letter ..."); the runners
use their common lettered-option prompt.
Dropped: the 4 questions on UrbanVideo/UrbanVideo_0021.mp4 (path_planning_0149 / 0150, action_prediction_0148 /
0149): the released file (10,223,664 bytes, same size as on the Hub) has no moov atom and cannot be decoded.

    MODULE=bench.converters.sis_bench sbatch slurm/convert_cpu.sbatch     (or: python -m bench.converters.sis_bench)
    (resumable: videos whose frames are already written are skipped)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from bench.common import CONVERTED, DATASETS, validate_row, write_jsonl
from bench.converters.uav_common import answer_balance, atomic_save, read_frames, rel, show_samples, video_info

REV = "da50acd6e80cf27413caa27cf90aaf9a6d04e16b"
SNAP = (Path(os.environ.get("EOJEV_DATASET_HUB", DATASETS / "hf_hub")) / "datasets--choucsan--SIS-Bench" / "snapshots"
        / REV)
OUT = CONVERTED / "sis_bench"
MAX_FRAMES = 32
MIN_PX, MAX_PX = 256 * 28 * 28, 512 * 28 * 28  # run_uav.py defaults
# README taxonomy: task -> (dimension, level)
TAXONOMY = {
    "object_existence": ("spatial cognition", "perception"), "object_attribute": ("spatial cognition", "perception"),
    "relative_direction": ("spatial cognition", "perception"),
    "landmark_appearance_order": ("spatial cognition", "memory"), "landmark_recall": ("spatial cognition", "memory"),
    "positional_relationship": ("spatial cognition", "memory"),
    "spatial_consistency": ("spatial cognition", "reasoning"),
    "spatio-temporal_consistency": ("spatial cognition", "reasoning"),
    "action_recognition": ("self-awareness", "perception"), "action_sequence": ("self-awareness", "memory"),
    "action_recall": ("self-awareness", "memory"), "action_prediction": ("self-awareness", "reasoning"),
    "path_planning": ("self-awareness", "reasoning"),
}


def resize_to_pixel_range(img: Image.Image) -> Image.Image:
    """uav_dataset.py resize_to_pixel_range (bicubic; unchanged when already in range)."""
    w, h = img.size
    if MIN_PX <= w * h <= MAX_PX:
        return img
    s = ((MIN_PX if w * h < MIN_PX else MAX_PX) / (w * h)) ** 0.5
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)


def _video_job(video_name: str) -> tuple[str, dict]:
    src = SNAP / "video" / video_name
    vdir = OUT / "frames" / Path(video_name).with_suffix("")
    index = vdir / "_frames.json"
    if index.exists():
        return video_name, json.loads(index.read_text())
    vi = video_info(src)
    if not vi["ok"]:
        return video_name, {"ok": False, "error": "cannot open (no moov atom)"}
    total = vi["n_frames"]
    if total > MAX_FRAMES:
        idx = sorted({int(i) for i in np.linspace(0, total - 1, MAX_FRAMES, dtype=int)})
        frames, decoded = read_frames(src, idx)
    else:  # every frame (then thinned to 32 if the container under-reported)
        frames, decoded = read_frames(src, None)
        idx = sorted(frames)
        if len(idx) > MAX_FRAMES:
            idx = [idx[int(i)] for i in np.linspace(0, len(idx) - 1, MAX_FRAMES, dtype=int)]
    idx = [i for i in idx if i in frames]
    if not idx:
        return video_name, {"ok": False, "error": "no decodable frames"}
    files, sizes = [], set()
    for k, i in enumerate(idx):
        f = vdir / f"f{k:02d}.jpg"
        im = resize_to_pixel_range(frames[i])
        sizes.add(im.size)
        if not f.exists():
            atomic_save(im, f)
        files.append(rel(f))
    info = {"ok": True, "fps": vi["fps"], "n_frames_reported": total, "n_decoded_until_last": decoded,
            "orig_px": [vi["width"], vi["height"]], "px": [list(s) for s in sorted(sizes)], "indices": idx,
            "times_s": [round(i / vi["fps"], 2) for i in idx], "duration_s": round(total / vi["fps"], 2),
            "files": files}
    tmp = index.with_name(".tmp_frames.json")
    tmp.write_text(json.dumps(info))
    os.replace(tmp, index)
    return video_name, info


def state_text(v: dict) -> str:
    times = ", ".join(f"{t:g}" for t in v["times_s"])
    return (f"The {len(v['files'])} images are frames sampled uniformly from a {v['duration_s']:g} s drone video, "
            f"in chronological order, at t = {times} s.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    items = [json.loads(l) for l in open(SNAP / "SIS-Bench.jsonl") if l.strip()]
    videos = sorted({x["video_name"] for x in items})
    print(f"SIS-Bench: {len(items)} questions on {len(videos)} videos", flush=True)
    infos, t0 = {}, time.time()
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=50) as pool:
        for i, (v, info) in enumerate(pool.imap_unordered(_video_job, videos), 1):
            infos[v] = info
            if i % 100 == 0 or i == len(videos):
                print(f"[videos {i}/{len(videos)}] {time.time() - t0:.0f}s", flush=True)

    rows, dropped = [], Counter()
    for x in items:
        v = infos[x["video_name"]]
        if not v["ok"]:
            dropped[f"undecodable video {x['video_name']}"] += 1
            continue
        letters = list(x["options"])
        if letters != ["A", "B", "C", "D"] or x["answer"] not in letters:
            raise ValueError(f"{x['question_id']}: options {letters} answer {x['answer']}")
        texts = [x["options"][l].strip() for l in letters]
        dup = len(set(texts)) != len(texts)
        if dup:  # keep every option distinct, as xlrs.py / mm_uavbench.py do
            texts = [f"{t} ({l})" for l, t in zip(letters, texts)]
            dropped["(kept) items with a repeated option text: letter appended"] += 1
        dim, level = TAXONOMY[x["task_type"]]
        row = {
            "id": f"SIS:{x['question_id']}",
            "group_id": f"SIS:{x['video_name']}",
            "benchmark": "SIS-Bench",
            "split": "test",
            "task_family": x["task_type"],
            "skills": ["uav", "video", dim, level],
            "images": v["files"],
            "state": state_text(v),
            "questions": {"q": {"type": "choice", "instructions": x["question"].strip(),
                                "criteria": {t: None for t in texts}}},
            "answer": {"q": texts[letters.index(x["answer"])]},
            "meta": {"question_id": x["question_id"], "video_name": x["video_name"], "concat_num": x["concat_num"],
                     "dimension": dim, "level": level, "gold_letter": x["answer"], "duplicate_options": dup,
                     "frames": {k: v[k] for k in ("fps", "n_frames_reported", "orig_px", "px", "indices",
                                                  "duration_s")}},
        }
        validate_row(row)
        rows.append(row)
    n = write_jsonl(OUT / "sis_bench.jsonl", rows)
    ok = [v for v in infos.values() if v["ok"]]
    print(f"SIS-Bench: {n} rows -> {OUT / 'sis_bench.jsonl'}; dropped: {dict(dropped)}")
    print("task families:", json.dumps(Counter(r["task_family"] for r in rows)))
    print("videos ok:", len(ok), "frames:", sum(len(v["files"]) for v in ok),
          "frames per video:", dict(Counter(len(v["files"]) for v in ok)),
          "frame sizes:", dict(Counter(tuple(map(tuple, v["px"])) for v in ok).most_common(6)))
    print("gold position per family:", json.dumps(answer_balance(rows)))
    show_samples(rows)


if __name__ == "__main__":
    main()
