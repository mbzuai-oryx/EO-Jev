"""C24 BigEarthNet.txt (arXiv 2603.29630) bench split, binary + MCQ VQA -> rows, in two image conditions.

Source: HF BIFOLD-BigEarthNetv2-0/BigEarthNet.txt @ 72d865f2146f0a85b720f7f3ca1cdbaeafc3d316 (BigEarthNet.txt.parquet,
CDLA-Permissive-1.0), rows with split == "bench": binary 6,927 (presence 1,458, area 1,306, count 1,306, adjacency
2,857) and mcq 5,550 (presence 709, area 653, count 538, adjacency 1,311, relative pos 652, country 312, season 700,
climate zone 675) over 1,065 Sentinel-2 / Sentinel-1 patch pairs (the bench's bounding-box 1,582 and captioning 970
rows are not closed-form: not converted). Images: BigEarthNet v2.0 (Zenodo 10891137) S2 L2A bands and S1 GRD VV /
VH (dB) for the bench patches, in raw/bigearthnet-v2-txt-bench/ (120 x 120 px, 10 m).

Questions keep the published text: binary -> `noul` (the `input` verbatim; gold `output` yes / no); mcq -> 4-way
`choice`, the inline "a) .., b) .., c) .., d) .." options parsed out of `input` (letters stripped, order kept),
the text before "a)" is the question. Conditions (two files, same rows and questions):
  s2rgb  Image 1 only: Sentinel-2 true colour B04 / B03 / B02, as the paper's RGB models saw; state "".
  s2s1   Image 1 the same S2 image, Image 2 Sentinel-1 rendered as R = VV, G = VH, B = VV - VH (dB); the state
         says what both are. (The paper's S1+S2 models read the raw bands, not a rendering.)
Rendering: neither the paper nor the dataset code states an RGB stretch for its VLM evaluation (ben_txt_datamodule.py
only mean/std-normalises bands for training), so every channel gets a per-patch 2-98 percentile linear stretch to
0-255 (S2 reflectance bands and S1 dB channels alike). Both 120 x 120 patches are upscaled bicubic to 256 x 256
(65,536 px = MIN_PIXELS; factor 2.1333, recorded in meta).

ADAPTED (not as-published): our percentile stretch and 256 px upscale (the paper's preprocessing for general VLMs
is unstated); the S1 rendering of condition s2s1 is ours; options are lettered A-D by the runner instead of a-d.
Country / season / climate zone are kept in meta only (they are answers of the mcq location / season / climate
questions).

    python -m bench.converters.bigearthnet_txt
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from bench.common import CONVERTED, DATASETS, MIN_PIXELS, RAW, save_png, validate_row, write_jsonl

REV = "72d865f2146f0a85b720f7f3ca1cdbaeafc3d316"
PARQUET = DATASETS / "hf_hub" / "datasets--BIFOLD-BigEarthNetv2-0--BigEarthNet.txt" / "snapshots" / REV / "BigEarthNet.txt.parquet"
PATCHES = RAW / "bigearthnet-v2-txt-bench"
OUT = CONVERTED / "bigearthnet_txt"
SIDE = 256  # 256 x 256 = MIN_PIXELS
MCQ = re.compile(r"^(.*?)\s*a\)\s(.*?),\s*b\)\s(.*?),\s*c\)\s(.*?),\s*d\)\s(.*)$", re.S)
STATES = {
    "s2rgb": "",
    "s2s1": ("Image 1 is a Sentinel-2 true-colour image (red, green, blue = bands B04, B03, B02; 10 m pixels). "
             "Image 2 is a Sentinel-1 radar (SAR) image of the same area at the same scale, shown as false colour: "
             "red = VV backscatter, green = VH backscatter, blue = VV minus VH (all in dB)."),
}


def stretch(band: np.ndarray) -> np.ndarray:
    """Per-patch 2-98 percentile linear stretch to uint8 (non-finite pixels ignored for the percentiles)."""
    band = band.astype(np.float64)
    lo, hi = np.nanpercentile(np.where(np.isfinite(band), band, np.nan), [2, 98])
    out = (band - lo) / max(hi - lo, 1e-6)
    return np.round(np.clip(np.nan_to_num(out, nan=0.0), 0, 1) * 255).astype(np.uint8)


def upscale(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb, "RGB").resize((SIDE, SIDE), Image.Resampling.BICUBIC)


def band(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if im.size != (120, 120):
            raise ValueError(f"{path}: size {im.size}")
        return np.asarray(im)


def render_s2(folder: Path, dest: Path) -> None:
    if dest.exists():
        return
    rgb = np.stack([stretch(band(folder / f"{folder.name}_{b}.tif")) for b in ("B04", "B03", "B02")], axis=-1)
    save_png(upscale(rgb), dest)


def render_s1(folder: Path, dest: Path) -> None:
    if dest.exists():
        return
    vv, vh = (band(folder / f"{folder.name}_{p}.tif").astype(np.float64) for p in ("VV", "VH"))
    save_png(upscale(np.stack([stretch(vv), stretch(vh), stretch(vv - vh)], axis=-1)), dest)


def main() -> None:
    table = pq.read_table(PARQUET, filters=[("split", "=", "bench")])
    records = [r for r in table.to_pylist() if r["type"] in ("binary", "mcq")]
    s2_dirs = {p.name: p for p in (PATCHES / "BigEarthNet-S2").glob("*/*") if p.is_dir()}
    s1_dirs = {p.name: p for p in (PATCHES / "BigEarthNet-S1").glob("*/*") if p.is_dir()}
    images = {}  # patch name -> path under CONVERTED
    for r in records:
        for name, dirs, kind, fn in ((r["patch_id"], s2_dirs, "s2", render_s2), (r["s1_name"], s1_dirs, "s1", render_s1)):
            if name not in images:
                dest = OUT / "images" / kind / f"{name}.png"
                fn(dirs[name], dest)
                images[name] = str(dest.relative_to(CONVERTED))
    variants = {v: [] for v in STATES}
    for r in records:
        if r["type"] == "binary":
            if r["output"] not in ("yes", "no"):
                raise ValueError(f"{r['ID']}: output {r['output']!r}")
            questions = {"q": {"type": "noul", "instructions": r["input"]}}
            answer = {"q": r["output"] == "yes"}
            options = None
        else:
            m = MCQ.match(r["input"])
            if not m or r["output"] not in ("a", "b", "c", "d"):
                raise ValueError(f"{r['ID']}: unparsed mcq {r['input']!r}")
            options = [m.group(k).strip() for k in range(2, 6)]
            if len(set(options)) != 4:
                raise ValueError(f"{r['ID']}: duplicate options")
            questions = {"q": {"type": "choice", "instructions": m.group(1).strip(), "criteria": {o: None for o in options}}}
            answer = {"q": options["abcd".index(r["output"])]}
        family = f"{r['type']}_{r['category'].replace(' ', '_')}"
        for variant, state in STATES.items():
            imgs = [images[r["patch_id"]]] + ([images[r["s1_name"]]] if variant == "s2s1" else [])
            row = {
                "id": f"BigEarthNet.txt:{r['ID']}:{variant}",
                "group_id": f"BigEarthNet.txt:{r['patch_id']}",
                "benchmark": f"BigEarthNet.txt-{variant}",
                "split": "test",
                "task_family": family,
                "skills": ["multisensor" if variant == "s2s1" else "multispectral_rgb", r["category"].replace(" ", "_")],
                "images": imgs,
                "state": state,
                "questions": questions,
                "answer": answer,
                "meta": {"source_id": r["ID"], "variant": variant, "type": r["type"], "category": r["category"],
                         "published_input": r["input"], "published_output": r["output"], "options": options,
                         "patch_id": r["patch_id"], "s1_name": r["s1_name"], "country": r["country"],
                         "season": r["season"], "climate_zone": r["climate_zone"], "lat": r["latitude"],
                         "lon": r["longitude"], "orig_px": [120, 120], "resize": round(SIDE / 120, 4),
                         "resample": "bicubic", "stretch": "per-patch 2-98 percentile per channel",
                         "sensor": "S2 L2A B04/B03/B02" + (" + S1 GRD VV/VH/VV-VH dB" if variant == "s2s1" else ""),
                         "gsd_m": 10},
            }
            validate_row(row)
            variants[variant].append(row)
    for variant, rows in variants.items():
        n = write_jsonl(OUT / f"bigearthnet_txt_{variant}.jsonl", rows)
        print(f"BigEarthNet.txt-{variant}:", n, "rows ->", OUT / f"bigearthnet_txt_{variant}.jsonl")
    rows = variants["s2rgb"]
    print("  questions by type (per file):", dict(Counter(q["type"] for r in rows for q in r["questions"].values())))
    print("  task_family:", json.dumps(dict(sorted(Counter(r["task_family"] for r in rows).items()))))
    n_s2 = sum(1 for p in images.values() if "/s2/" in p)
    assert all(Image.open(CONVERTED / p).size == (SIDE, SIDE) and SIDE * SIDE >= MIN_PIXELS for p in images.values())
    print("  images:", n_s2, "S2 +", len(images) - n_s2, "S1 renderings, all upscaled 120 -> 256 (bicubic)")
    print("  dropped: bench rows of type 'bounding box' and 'captioning' (not closed-form):",
          dict(Counter(r["type"] for r in table.select(["type"]).to_pylist() if r["type"] not in ("binary", "mcq"))))
    print("  binary gold:", dict(Counter(r["meta"]["published_output"] for r in rows if r["meta"]["type"] == "binary")),
          "| mcq gold:", dict(sorted(Counter(r["meta"]["published_output"] for r in rows if r["meta"]["type"] == "mcq").items())))


if __name__ == "__main__":
    main()
