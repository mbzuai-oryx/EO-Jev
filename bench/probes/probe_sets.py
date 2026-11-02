"""Train/test row sets for the representation-vs-readout probes (docs/03 §8.3).

  AID      all 10,000 images; test = the GeoChat 3,000-image list (the zero-shot eval set), train = the rest
  EuroSAT  train: seeded 5,000 of the official train split; test: the official 5,400
  NWPU     train: seeded 5,000 of the official train split; test: the official 6,300
Rows are the scene-classification rows of bench/converters/scene_cls.py (one full-list `choice`), so a probe run
also yields zero-shot answers; meta["probe_split"] says train / test.

    python -m bench.probes.probe_sets
"""

from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

from bench.common import CONVERTED, RAW, write_jsonl
from bench.converters.scene_cls import EUROSAT_NAMES, make_row, readable, store_image

OUT = CONVERTED / "probes"


def aid() -> list[dict]:
    test = {it["image"].rsplit(".", 1)[0] for it in map(json.loads, open(RAW / "geochat-bench" / "aid.jsonl"))}
    rows, entries = [], []
    with zipfile.ZipFile(RAW / "aid" / "AID.zip") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".jpg"):
                continue
            folder, fname = name.split("/")[1:3]
            stem = f"{folder}/{Path(fname).stem}"
            rel, size = store_image(zf.read(name), OUT / "AID" / stem)
            entries.append((stem, rel, size, readable(folder), "test" if stem in test else "train"))
    classes = sorted({e[3] for e in entries})
    for stem, rel, size, label, split in entries:
        r = make_row("AID", stem, rel, size, label, classes)
        r["id"] = f"probe:{r['id']}"
        r["meta"]["probe_split"] = split
        rows.append(r)
    return rows


def parquet_split(bench: str, name: str, names: list[str], n_train: int) -> list[dict]:
    classes = sorted(names)
    rows = []
    for split in ("train", "test"):
        records = [r for f in sorted((RAW / name / "data").glob(f"{split}-*.parquet")) for r in pq.read_table(f).to_pylist()]
        if split == "train":
            records = random.Random(0).sample(records, n_train)
        for i, rec in enumerate(records):
            item = rec.get("image_id") or f"{split}{i:06d}"
            label = names[rec["label"]]
            rel, size = store_image(rec["image"]["bytes"], OUT / bench / split / label.replace(" ", "_") / item)
            r = make_row(bench, f"{split}/{label}/{item}", rel, size, label, classes)
            r["id"] = f"probe:{r['id']}"
            r["meta"]["probe_split"] = split
            rows.append(r)
    return rows


def main() -> None:
    eurosat_meta = json.loads(pq.ParquetFile(sorted((RAW / "eurosat-rgb" / "data").glob("test-*"))[0]).schema_arrow.metadata[b"huggingface"])
    eurosat_names = [EUROSAT_NAMES[n] for n in eurosat_meta["info"]["features"]["label"]["names"]]
    resisc_meta = json.loads(pq.ParquetFile(sorted((RAW / "resisc45" / "data").glob("test-*"))[0]).schema_arrow.metadata[b"huggingface"])
    resisc_names = [readable(n) for n in resisc_meta["info"]["features"]["label"]["names"]]
    sets = {"AID": aid(), "EuroSAT-RGB": parquet_split("EuroSAT-RGB", "eurosat-rgb", eurosat_names, 5000),
            "NWPU-RESISC45": parquet_split("NWPU-RESISC45", "resisc45", resisc_names, 5000)}
    for bench, rows in sets.items():
        n = write_jsonl(OUT / f"probe_{bench}.jsonl", rows)
        print(bench, n, "rows:", sum(r["meta"]["probe_split"] == "train" for r in rows), "train /",
              sum(r["meta"]["probe_split"] == "test" for r in rows), "test")


if __name__ == "__main__":
    main()
