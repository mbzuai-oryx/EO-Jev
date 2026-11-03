"""C6 scene-classification pack -> rows.

UCM and AID follow the GeoChat-Bench protocol (all 2,100 UCM images; the 3,000-image AID list) with its class
names; WHU-RS19 (1,005) and SIRI-WHU (2,400) use every image; EuroSAT-RGB (5,400) and NWPU-RESISC45 (6,300) use
their test splits. Each image gets one `choice` over the dataset's full class list (option order shuffled per item
with a fixed seed). Datasets with more than 16 classes also get a 16-way file: the true class plus 15 seeded
distractors (reported separately; not comparable to N-way accuracy).

    python -m bench.converters.scene_cls            # writes $EOJEV_DATASETS/converted/scene/*.jsonl + images
"""

from __future__ import annotations

import io
import json
import random
import re
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from bench.common import CONVERTED, RAW, save_png, validate_row, write_jsonl

OUT = CONVERTED / "scene"
INSTRUCTIONS = "Which scene category best describes this remote sensing image?"
NATIVE_FORMATS = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}

EUROSAT_NAMES = {
    "AnnualCrop": "annual crop", "Forest": "forest", "HerbaceousVegetation": "herbaceous vegetation",
    "Highway": "highway", "Industrial": "industrial", "Pasture": "pasture", "PermanentCrop": "permanent crop",
    "Residential": "residential", "River": "river", "SeaLake": "sea or lake",
}
WHU_RS19_NAMES = ["airport", "beach", "bridge", "commercial", "desert", "farmland", "football field", "forest",
                  "industrial", "meadow", "mountain", "park", "parking", "pond", "port", "railway station",
                  "residential", "river", "viaduct"]
SIRI_NAMES = ["agriculture", "commercial", "harbor", "idle land", "industrial", "meadow", "overpass", "park", "pond",
              "residential", "river", "water"]

DATASET_INFO = {
    "UCM": {"sensor": "USGS aerial orthophoto, RGB", "gsd_m": 0.3, "license": "public domain (USGS)"},
    "AID": {"sensor": "Google Earth, RGB", "gsd_m": "0.5-8", "license": "research use"},
    "WHU-RS19": {"sensor": "Google Earth, RGB", "gsd_m": "up to 0.5", "license": "cc-by-4.0 (HF tag)"},
    "SIRI-WHU": {"sensor": "Google Earth, RGB", "gsd_m": 2, "license": "research use (HF tag: other)"},
    "EuroSAT-RGB": {"sensor": "Sentinel-2 RGB (B4,B3,B2)", "gsd_m": 10, "license": "MIT"},
    "NWPU-RESISC45": {"sensor": "Google Earth, RGB", "gsd_m": "0.2-30", "license": "research use"},
}


def readable(name: str) -> str:
    """'MediumResidential' / 'medium_residential' -> 'medium residential'."""
    name = name.replace("_", " ")
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return name.lower().strip()


def store_image(data: bytes, dest_stem: Path) -> tuple[str, list[int]]:
    """Keep PNG / JPEG / WebP bytes as they are; re-encode anything else (e.g. TIFF) to PNG."""
    image = Image.open(io.BytesIO(data))
    size = list(image.size)
    ext = NATIVE_FORMATS.get(image.format or "")
    if ext and image.mode in ("RGB", "L"):
        path = dest_stem.with_suffix(ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    else:
        path = save_png(image, dest_stem.with_suffix(".png"))
    return str(path.relative_to(CONVERTED)), size


def make_row(bench: str, item: str, image_rel: str, size: list[int], label: str, classes: list[str],
             options: list[str] | None = None) -> dict:
    rng = random.Random(f"{bench}:{item}")
    opts = list(options or classes)
    rng.shuffle(opts)
    row = {
        "id": f"{bench}:{item}",
        "group_id": f"{bench}:{item}",
        "benchmark": bench,
        "split": "test",
        "task_family": "scene_cls",
        "skills": ["lulc"],
        "images": [image_rel],
        "state": "",
        "questions": {"scene": {"type": "choice", "instructions": INSTRUCTIONS,
                                "criteria": {c: None for c in opts}}},
        "answer": {"scene": label},
        "meta": {"orig_px": size, "n_options": len(opts), "n_classes": len(classes), **DATASET_INFO[bench]},
    }
    validate_row(row)
    return row


def sixteen_way(bench: str, rows: list[dict], classes: list[str]) -> list[dict]:
    out = []
    for row in rows:
        label = row["answer"]["scene"]
        rng = random.Random(f"16way:{row['id']}")
        options = [label] + rng.sample([c for c in classes if c != label], 15)
        item = row["id"].split(":", 1)[1]
        r = make_row(bench, item, row["images"][0], row["meta"]["orig_px"], label, classes, options)
        r["id"] = row["id"] + ":16way"
        out.append(r)
    return out


def geochat_list(name: str) -> list[dict]:
    return [json.loads(line) for line in open(RAW / "geochat-bench" / name)]


def ucm() -> tuple[list[dict], list[str]]:
    items = geochat_list("UCmerced.jsonl")
    classes = sorted({it["ground_truth"] for it in items})
    rows = []
    with zipfile.ZipFile(RAW / "ucmerced" / "UCMerced_LandUse.zip") as zf:
        for it in items:  # e.g. "denseresidential/denseresidential31.tif"
            data = zf.read(f"UCMerced_LandUse/Images/{it['image']}")
            stem = Path(it["image"]).with_suffix("")
            rel, size = store_image(data, OUT / "UCM" / stem)
            rows.append(make_row("UCM", str(stem), rel, size, it["ground_truth"], classes))
    return rows, classes


def aid() -> tuple[list[dict], list[str]]:
    items = geochat_list("aid.jsonl")
    rows, classes = [], set()
    with zipfile.ZipFile(RAW / "aid" / "AID.zip") as zf:
        for it in items:  # e.g. image "MediumResidential/mediumresidential_288.png", gt "MediumResidentialgt"
            folder, fname = it["image"].split("/")
            label = readable(it["ground_truth"].removesuffix("gt"))
            classes.add(label)
            data = zf.read(f"AID/{folder}/{Path(fname).stem}.jpg")
            stem = Path(folder) / Path(fname).stem
            rel, size = store_image(data, OUT / "AID" / stem)
            rows.append((str(stem), rel, size, label))
    classes = sorted(classes)
    return [make_row("AID", s, rel, size, label, classes) for s, rel, size, label in rows], classes


def parquet_dataset(bench: str, files: list[Path], names: list[str]) -> tuple[list[dict], list[str]]:
    classes = sorted(names)
    rows = []
    for f in files:
        table = pq.read_table(f).to_pylist()
        for i, rec in enumerate(table):
            item = rec.get("image_id") or Path(rec["image"].get("path") or f"{i:06d}").stem or f"{i:06d}"
            label = names[rec["label"]]
            rel, size = store_image(rec["image"]["bytes"], OUT / bench / f"{label.replace(' ', '_')}" / item)
            rows.append(make_row(bench, f"{label}/{item}", rel, size, label, classes))
    return rows, classes


def main() -> None:
    def files(name: str, pattern: str) -> list[Path]:
        return sorted((RAW / name / "data").glob(pattern))

    eurosat_meta = json.loads(pq.ParquetFile(files("eurosat-rgb", "test-*.parquet")[0]).schema_arrow.metadata[b"huggingface"])
    eurosat_names = [EUROSAT_NAMES[n] for n in eurosat_meta["info"]["features"]["label"]["names"]]
    resisc_meta = json.loads(pq.ParquetFile(files("resisc45", "test-*.parquet")[0]).schema_arrow.metadata[b"huggingface"])
    resisc_names = [readable(n) for n in resisc_meta["info"]["features"]["label"]["names"]]

    builders = {
        "UCM": ucm,
        "AID": aid,
        "WHU-RS19": lambda: parquet_dataset("WHU-RS19", files("whu-rs19", "*.parquet"), WHU_RS19_NAMES),
        "SIRI-WHU": lambda: parquet_dataset("SIRI-WHU", files("siri-whu", "*.parquet"), SIRI_NAMES),
        "EuroSAT-RGB": lambda: parquet_dataset("EuroSAT-RGB", files("eurosat-rgb", "test-*.parquet"), eurosat_names),
        "NWPU-RESISC45": lambda: parquet_dataset("NWPU-RESISC45", files("resisc45", "test-*.parquet"), resisc_names),
    }
    summary = {}
    for bench, build in builders.items():
        rows, classes = build()
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"{bench}: duplicate row ids")
        write_jsonl(OUT / f"{bench}.jsonl", rows)
        summary[bench] = {"rows": len(rows), "classes": len(classes)}
        if len(classes) > 16:
            n = write_jsonl(OUT / f"{bench}_16way.jsonl", sixteen_way(bench, rows, classes))
            summary[bench]["rows_16way"] = n
        print(bench, summary[bench], flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
