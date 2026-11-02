"""Shared helpers for the UAV / drone converters (C28 mm_uavbench, C29 spatialuav, C30 sis_bench, C31 spatialsky).

Image sizing follows the converter rules: images above MAX_PIXELS are downscaled (Lanczos, aspect kept), images
under MIN_PIXELS upscaled (bicubic); images already in range keep their native bytes. Files are written atomically
(temp file + rename) so an interrupted run never leaves a truncated image, and converters skip files that exist
(resumable). Video frames are read with OpenCV (opencv-python-headless 5.0 in the eojev env; PyAV and decord are
not installed).
"""

from __future__ import annotations

import math
import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from bench.common import CONVERTED, MAX_PIXELS, MIN_PIXELS

Image.MAX_IMAGE_PIXELS = None
JPEG_KW = {"quality": 95, "subsampling": 0}


def rel(p: Path) -> str:
    return str(p.relative_to(CONVERTED))


def fit_size(w: int, h: int, max_pixels: int = MAX_PIXELS, min_pixels: int = MIN_PIXELS) -> tuple[int, int, float]:
    """Target size and scale factor: downscale to <= max_pixels, upscale to >= min_pixels, else unchanged."""
    if w * h > max_pixels:
        s = math.sqrt(max_pixels / (w * h))
        return max(1, int(w * s)), max(1, int(h * s)), s
    if w * h < min_pixels:
        s = math.sqrt(min_pixels / (w * h))
        return math.ceil(w * s), math.ceil(h * s), s
    return w, h, 1.0


def resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if size == image.size:
        return image
    up = size[0] * size[1] > image.size[0] * image.size[1]
    return image.resize(size, Image.Resampling.BICUBIC if up else Image.Resampling.LANCZOS)


def atomic_save(image: Image.Image, path: Path) -> Path:
    """Write RGB JPEG (.jpg) or PNG (.png) via a temp file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode != "RGB":
        image = image.convert("RGB")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.suffix.lower() in (".jpg", ".jpeg"):
        image.save(tmp, format="JPEG", **JPEG_KW)
    else:
        image.save(tmp, format="PNG")
    os.replace(tmp, path)
    return path


def atomic_copy(src: Path, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, path)
    return path


def store_fitted(src: Path, dest_stem: Path, max_pixels: int = MAX_PIXELS, overlay=None,
                 resized_ext: str = ".jpg") -> dict:
    """Store one source image within [MIN_PIXELS, max_pixels].

    In-range RGB JPEG / PNG sources are copied byte for byte (as bench.common.store_image does). Others are
    resized and re-encoded (`resized_ext`: .jpg at quality 95 / 4:4:4, or .png). `overlay(image, scale)` (optional)
    draws on the resized image (forcing a re-encode). Skips work when the destination exists (resumable).
    Returns {"path", "orig_px", "px", "scale", "copied"}.
    """
    with Image.open(src) as im:
        w, h = im.size
        fmt, mode = im.format, im.mode
        tw, th, s = fit_size(w, h, max_pixels)
        native = (tw, th) == (w, h) and overlay is None and fmt in ("JPEG", "PNG") and mode == "RGB"
        if native:
            dest = dest_stem.with_suffix(".jpg" if fmt == "JPEG" else ".png")
        else:
            dest = dest_stem.with_suffix(resized_ext)
        info = {"path": rel(dest), "orig_px": [w, h], "px": [tw, th], "scale": round(s, 6), "copied": native}
        if dest.exists():
            return info
        if native:
            atomic_copy(src, dest)
            return info
        image = resize(im.convert("RGB"), (tw, th))
    if overlay is not None:
        image = overlay(image, s)
    atomic_save(image, dest)
    return info


# ---- video ---------------------------------------------------------------------------------------------------------

def video_info(path: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok": False}
    info = {"ok": True, "fps": cap.get(cv2.CAP_PROP_FPS), "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


def read_frames(path: Path, indices: list[int] | None) -> tuple[dict[int, Image.Image], int]:
    """Decode the video sequentially and return {index: RGB frame} for the wanted indices (0-based decode order;
    None = every frame), plus the number of frames decoded (stops after the last wanted index)."""
    import cv2

    wanted = None if indices is None else set(indices)
    last = float("inf") if indices is None else (max(indices) if indices else -1)
    cap = cv2.VideoCapture(str(path))
    out, i = {}, 0
    while i <= last:
        if not cap.grab():
            break
        if wanted is None or i in wanted:
            ok, bgr = cap.retrieve()
            if ok:
                out[i] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return out, i


def count_frames(path: Path) -> int:
    """Exact number of decodable frames (decodes the whole stream)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


# ---- checks --------------------------------------------------------------------------------------------------------

def show_samples(rows: list[dict], k: int = 3, seed: int = 0) -> None:
    """Print k rows as the runners render them (bench.specialists.prompting.render: state, question, lettered
    options), with their images and gold answers."""
    import random

    from bench.specialists.prompting import render

    picks = random.Random(seed).sample(rows, min(k, len(rows)))
    for r in picks:
        print("=" * 100)
        print(f"{r['id']}  [{r['task_family']}]  images ({len(r['images'])}): {r['images'][:4]}"
              + (" ..." if len(r["images"]) > 4 else ""))
        for qid in list(r["questions"])[:3]:
            item = render(r, qid)
            print(f"--- {qid} ({item.qtype}) gold = {r['answer'][qid]!r}")
            print(item.text)
        if len(r["questions"]) > 3:
            print(f"... and {len(r['questions']) - 3} more questions")


def answer_balance(rows: list[dict]) -> dict:
    """Gold position (choice: letter index) / yes rate (noul), per task family: checks for degenerate answers."""
    pos, yes = Counter(), Counter()
    for r in rows:
        for qid, q in r["questions"].items():
            if q["type"] == "choice":
                pos[(r["task_family"], "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[list(q["criteria"]).index(r["answer"][qid])])] += 1
            elif q["type"] == "noul":
                yes[(r["task_family"], bool(r["answer"][qid]))] += 1
    fam = {}
    for (f, l), n in sorted(pos.items()):
        fam.setdefault(f, {})[l] = n
    for (f, b), n in sorted(yes.items()):
        fam.setdefault(f, {})["yes" if b else "no"] = n
    return fam
