"""C15 (part) fMoW-WILDS OOD test (22,108 images, 2016-2017, unseen years) -> rows.

A seeded 2,000-image sample, stratified by WILDS region. Each image gets three questions:
  region    5-way `choice` (Africa, the Americas, Asia, Europe, Oceania)
  country   16-way `choice`: the true country + 15 distractors from the same region where possible (hard)
  land_use  62-way `choice` over the fMoW categories (readable names)
Score with `--by qid` to separate the three. VLM papers report fMoW land use only (GPT-4V .19, GPT-4o 21.4);
no geolocation numbers exist for this split.

    python -m bench.converters.fmow
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict

import pyarrow.parquet as pq
import pycountry

from bench.common import CONVERTED, RAW, store_image, validate_row, write_jsonl

SRC = RAW / "fmow-wilds" / "WILDS"
OUT = CONVERTED / "fmow"
N = 2000
REGIONS = {"Africa": "Africa", "Americas": "the Americas", "Asia": "Asia", "Europe": "Europe", "Oceania": "Oceania"}


def country_name(code: str) -> str | None:
    c = pycountry.countries.get(alpha_3=code)
    return getattr(c, "common_name", None) or getattr(c, "name", None) if c else None


def main() -> None:
    files = sorted(SRC.glob("test-*.parquet"))
    meta_cols = ["category", "country_code", "domain_labels_readable", "lat", "lon", "timestamp", "img_width",
                 "img_height", "img_path", "cloud_cover"]
    records = []
    for f in files:
        for i, r in enumerate(pq.read_table(f, columns=meta_cols).to_pylist()):
            r["_file"], r["_row"] = f, i
            r["region"] = r["domain_labels_readable"]["region"]
            r["country"] = country_name(r["country_code"] or "")
            if r["country"] and r["region"] in REGIONS:
                records.append(r)
    by_region = defaultdict(list)
    for r in records:
        by_region[r["region"]].append(r)
    countries_by_region = {reg: sorted({r["country"] for r in rs}) for reg, rs in by_region.items()}
    all_countries = sorted({r["country"] for r in records})
    categories = sorted({r["category"] for r in records})
    land_use = {c: c.replace("_", " ") for c in categories}

    rng = random.Random(0)
    per_region = N // len(by_region)
    picked = []
    for reg in sorted(by_region):
        rs = by_region[reg]
        picked += rng.sample(rs, min(per_region, len(rs)))
    picked += rng.sample([r for r in records if r not in picked], N - len(picked))

    by_file = defaultdict(list)
    for r in picked:
        by_file[r["_file"]].append(r)
    rows = []
    for f, rs in by_file.items():
        table = pq.read_table(f, columns=["image"]).column("image").to_pylist()
        for r in rs:
            stem = r["img_path"].replace("/", "_").rsplit(".", 1)[0]
            rel, _ = store_image(table[r["_row"]]["bytes"], OUT / "images" / stem)
            item_rng = random.Random(f"fmow:{stem}")
            same = [c for c in countries_by_region[r["region"]] if c != r["country"]]
            distract = item_rng.sample(same, min(15, len(same)))
            if len(distract) < 15:
                distract += item_rng.sample([c for c in all_countries if c != r["country"] and c not in distract],
                                            15 - len(distract))
            countries = [r["country"]] + distract
            item_rng.shuffle(countries)
            regions = list(REGIONS.values())
            item_rng.shuffle(regions)
            uses = list(land_use.values())
            item_rng.shuffle(uses)
            row = {
                "id": f"fMoW-WILDS:{stem}",
                "group_id": f"fMoW-WILDS:{stem}",
                "benchmark": "fMoW-WILDS-OOD",
                "split": "test (OOD years)",
                "task_family": "geo_and_landuse",
                "skills": ["geolocation", "land_use"],
                "images": [rel],
                "state": "",
                "questions": {
                    "region": {"type": "choice", "instructions": "On which continent was this satellite image taken?",
                               "criteria": {o: None for o in regions}},
                    "country": {"type": "choice", "instructions": "In which country was this satellite image taken?",
                                "criteria": {o: None for o in countries}},
                    "land_use": {"type": "choice",
                                 "instructions": "What is the main facility or land use shown in this satellite image?",
                                 "criteria": {o: None for o in uses}},
                },
                "answer": {"region": REGIONS[r["region"]], "country": r["country"], "land_use": land_use[r["category"]]},
                "meta": {"region": r["region"], "country_code": r["country_code"], "lat": r["lat"], "lon": r["lon"],
                         "date": r["timestamp"][:10], "orig_px": [r["img_width"], r["img_height"]],
                         "cloud_cover": r["cloud_cover"], "sensor": "WorldView / QuickBird / GeoEye RGB", "gsd_m": "0.3-1.6"},
            }
            validate_row(row)
            rows.append(row)
    n = write_jsonl(OUT / "fmow_wilds_ood_2k.jsonl", rows)
    print("fMoW-WILDS OOD:", n, "rows;", Counter(r["meta"]["region"] for r in rows), "|", len(all_countries),
          "countries,", len(categories), "categories")
    print("countries per region:", json.dumps({k: len(v) for k, v in countries_by_region.items()}))


if __name__ == "__main__":
    main()
