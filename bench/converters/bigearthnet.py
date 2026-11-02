"""C7 BigEarthNet v2 (reBEN) Sentinel-2 RGB, 19-class multi-label -> rows.

A seeded 1,000-patch sample of the official (geographically held-out) test split. Each row is one 120x120
patch (the runtime upsamples to its 65,536-pixel minimum) with 19 `noul` questions, one per class; per-class
AP / macro mAP and micro / sample F1 come from bench/metrics. Country, date and coordinates are kept in meta for
the metadata-in-state ablation (A2).

    python -m bench.converters.bigearthnet
"""

from __future__ import annotations

import io
import json
import random

import pyarrow.parquet as pq
from PIL import Image

from bench.common import CONVERTED, RAW, save_png, validate_row, write_jsonl

N = 1000
OUT = CONVERTED / "bigearthnet"
# The 19-class BigEarthNet nomenclature (CORINE-derived), as readable text.
LABEL_TEXT = {
    "agriculture_with_natural_vegetation": "agriculture with significant areas of natural vegetation",
    "agro_forestry_areas": "agro-forestry areas",
    "arable_land": "arable land",
    "beaches_dunes_sands": "beaches, dunes or sands",
    "broad_leaved_forest": "broad-leaved forest",
    "coastal_wetlands": "coastal wetlands",
    "complex_cultivation_patterns": "complex cultivation patterns",
    "coniferous_forest": "coniferous forest",
    "industrial_commercial_units": "industrial or commercial units",
    "inland_waters": "inland waters",
    "inland_wetlands": "inland wetlands",
    "marine_waters": "marine waters",
    "mixed_forest": "mixed forest",
    "moors_heathland_sclerophyllous_vegetation": "moors, heathland or sclerophyllous vegetation",
    "natural_grassland_sparse_vegetation": "natural grassland or sparsely vegetated areas",
    "pastures": "pastures",
    "permanent_crops": "permanent crops",
    "transitional_woodland_shrub": "transitional woodland or shrub",
    "urban_fabric": "urban fabric",
}


def main() -> None:
    files = sorted((RAW / "bigearthnet-v2-rgb" / "data").glob("test-*.parquet"))
    index = [(f, g, i) for f in files for g in range(pq.ParquetFile(f).num_row_groups)
             for i in range(pq.ParquetFile(f).metadata.row_group(g).num_rows)]
    picks = sorted(random.Random(0).sample(range(len(index)), N))
    by_group: dict[tuple, list[int]] = {}
    for k in picks:
        f, g, i = index[k]
        by_group.setdefault((f, g), []).append(i)

    rows = []
    for (f, g), positions in by_group.items():
        table = pq.ParquetFile(f).read_row_group(g).to_pylist()
        for i in positions:
            rec = table[i]
            present = set(rec["label_names"])
            image = Image.open(io.BytesIO(rec["image"]["bytes"]))
            rel = str(save_png(image, OUT / "images" / f"{rec['image_id']}.png").relative_to(CONVERTED))
            row = {
                "id": f"BigEarthNet-v2:{rec['image_id']}",
                "group_id": f"BigEarthNet-v2:{rec['image_id']}",
                "benchmark": "BigEarthNet-v2",
                "split": "test",
                "task_family": "multilabel_lulc",
                "skills": ["lulc", "multilabel"],
                "images": [rel],
                "state": "",
                "questions": {name: {"type": "noul", "instructions": f"Does this image contain {text}?"}
                              for name, text in LABEL_TEXT.items()},
                "answer": {name: name in present for name in LABEL_TEXT},
                "meta": {"orig_px": list(image.size), "sensor": "Sentinel-2 RGB (B4,B3,B2)", "gsd_m": 10,
                         "country": rec["country"], "date": str(rec["acquisition_time"].date()),
                         "lat": rec["lat"], "lon": rec["lon"], "n_labels": len(present)},
            }
            validate_row(row)
            rows.append(row)
    n = write_jsonl(OUT / "bigearthnet_v2_test_1k.jsonl", rows)
    prevalence = {k: sum(r["answer"][k] for r in rows) for k in LABEL_TEXT}
    print("BigEarthNet-v2:", n, "rows x 19 questions; label counts:", json.dumps(prevalence))


if __name__ == "__main__":
    main()
