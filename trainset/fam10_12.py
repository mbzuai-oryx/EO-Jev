"""Families 10-12: multi-select, geo-localisation and low-resolution presence questions.

- **fam10_multiselect: "select every option that applies".**
  - Layout: exactly as the benchmarks score multi-select. Each option is its own yes/no question, "<question> Is
    "<option>" one of them?" (the DisasterM3 converter) or "... part of the correct answer?" (DVL). All options of
    an item sit in one row.
  - Why: trained without it, the model said yes to fewer of the present options (DisasterM3 multi-select
    21.6 → 9.4), though it still ranked them as well as d3-flash did.
  - DOTA train tiles / DIOR images (fam2's sources and exclusions) with at least 2 object types: which types are
    present. Up to 5 present ones and enough absent ones (half from the same scene) that ~35-50% of the options
    are "yes", as in the benchmarks (DisasterM3 40%, DVL 43% of options correct).
  - No BigEarthNet items (TARGET ms_ben = 0): with them, the multi-select yes rate on Sentinel-2 labels set
    BigEarthNet's own yes rate (16% -> BigEarthNet yes 26%, 32% -> 34%, gold 16%; accuracy 76.1 against 81.3
    without). The multi-select format also follows fam10's overall yes rate (21% -> DisasterM3 5.7), so it is set
    by the aerial items alone.
- **fam11_geo: continent (5-way) and country of fMoW train images.**
  - Source: the images' own WILDS metadata.
  - Country options: 16 in 60% of rows, otherwise 4–10, with same-continent distractors as in the eval.
  - Leak rule: images within 0.05° (~5 km) of any fMoW-WILDS eval site are dropped.
  - Why: fMoW country fell 6.7 points without it; this protects the geo-localisation d3 gained from decision training.
- **fam12_presence_lr: broad land-cover areas in 10 m Sentinel-2 patches.**
  - Concepts: water, built-up, industrial / commercial, farmland, forest, wetland, from BigEarthNet labels.
  - Each patch gets one present concept and up to three absent ones (~25% "yes"; one absent per patch gave 50%), and each
    concept is the "yes" answer equally often. "built" and "industrial" count as absent only on remote patches
    (fam1's REMOTE_LABELS): patches without the 25 ha urban label can hold houses. Asking "no built-up /
    residential area" on any patch made sparse residential scenes collapse on UCM / AID / NWPU; asking it on a third
    of the patches recovered only part. The wording "a residential area" is gone: CORINE urban fabric is not it.
  - Why: RSVQA-LR presence fell 10 points when all presence training came from 0.5 m DOTA / DIOR.
  - CORINE labels map areas of at least 25 ha, so questions ask about areas ("a built-up area"), not single
    objects.

    MODULE=trainset.fam10_12 [ARGS="--sample 10"] sbatch -J eojev-fam10 slurm/convert_cpu.sbatch
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from bench.common import CONVERTED, write_jsonl
from bench.converters.bigearthnet import LABEL_TEXT
from bench.converters.fmow import REGIONS, country_name
from trainset.common import ROWS, make_row, render, store
from trainset.fam1_labels import MAN, REMOTE_LABELS, ben_negatives, ben_records, ben_state
from trainset.fam2_9_objects import ImageStore, dior_images, dota_scenes, dota_tiles, names

TARGET = {"ms_ben": 0, "ms_dota": 2400, "ms_dior": 1600, "geo": 3000, "presence_lr": 3000}
MS_YES_SHARE = (0.35, 0.50)  # range of the share of "yes" options in an object multi-select item
SUFFIXES = [("one_of", ' Is "{o}" one of them?'), ("part_of", ' Is "{o}" part of the correct answer?')]
MS_BEN_T = [("msb_v1", "Which land-cover classes appear in this image?"),
            ("msb_v2", "Select every land-cover type that is present in this image."),
            ("msb_v3", "Which of the following land-cover classes can be found in this area?"),
            ("msb_v4", "Which kinds of land cover does this image contain?")]
MS_OBJ_T = [("mso_v1", "Which of these object types are present in this image?"),
            ("mso_v2", "Select every kind of object that can be seen in this image."),
            ("mso_v3", "Which of the following objects appear in this image?"),
            ("mso_v4", "Which object categories does this image contain?")]
CONT_T = [("cont_v1", "Which continent is shown in this satellite image?"),
          ("cont_v2", "This image was taken on which continent?"),
          ("cont_v3", "On what continent is the site in this image located?")]
CTRY_T = [("ctry_v1", "Which country is shown in this satellite image?"),
          ("ctry_v2", "This image was captured in which country?"),
          ("ctry_v3", "In what country is the site in this image located?"),
          ("ctry_v4", "Which country does this satellite image come from?")]
PRLR_T = [("prlr_v1", "Does this image show {np}?"), ("prlr_v2", "Can {np} be seen in this image?"),
          ("prlr_v3", "Does this area include {np}?"), ("prlr_v4", "Does the scene contain {np}?")]
SETTLEMENT_CONCEPTS = {"built", "industrial"}  # asked as absent only on remote patches
CONCEPTS = {
    "water": ({"inland_waters", "marine_waters"}, ["a water body", "open water", "a lake, river or sea"]),
    "built": ({"urban_fabric"}, ["a built-up area", "an urban area"]),
    "industrial": ({"industrial_commercial_units"}, ["an industrial or commercial area", "an industrial zone"]),
    "farmland": ({"arable_land", "permanent_crops", "pastures", "complex_cultivation_patterns",
                  "agriculture_with_natural_vegetation"}, ["farmland", "agricultural land", "cropland or pasture"]),
    "forest": ({"broad_leaved_forest", "coniferous_forest", "mixed_forest"}, ["forest", "a wooded area"]),
    "wetland": ({"inland_wetlands", "coastal_wetlands"}, ["a wetland", "marshland"]),
}
BEN_LICENCE = "CDLA-Permissive-1.0"
GEO_BUFFER_DEG = 0.05


def multiselect_questions(rng, stem: str, opts: list[tuple[str, bool]]) -> tuple[dict, dict, str]:
    sid, suffix = rng.choice(SUFFIXES)
    questions = {f"opt{i}": {"type": "noul", "instructions": stem + suffix.format(o=text), "criteria": None}
                 for i, (text, _) in enumerate(opts)}
    return questions, {f"opt{i}": yes for i, (_, yes) in enumerate(opts)}, sid


def by_parquet(recs: list[dict]):
    """(record, image bytes) for the picked BigEarthNet records, reading each parquet file's images once."""
    files = collections.defaultdict(list)
    for r in recs:
        files[r["_f"]].append(r)
    for f, rs in files.items():
        images = pq.read_table(f, columns=["image"]).column("image")
        for r in rs:
            yield r, images[r["_i"]].as_py()["bytes"]


def ben_multiselect(rng, recs, n) -> list[dict]:
    rows = []
    for r, data in by_parquet(rng.sample(recs, n)):
        present = list(r["label_names"])
        rng.shuffle(present)
        pos = present[:5]
        k = max(rng.choice([6, 8, 10]), len(pos) + 2)
        opts = [(x, True) for x in pos] + [(x, False) for x in ben_negatives(rng, present, k - len(pos))]
        rng.shuffle(opts)
        tid, stem = rng.choice(MS_BEN_T)
        questions, answer, sid = multiselect_questions(rng, stem, [(LABEL_TEXT[x], y) for x, y in opts])
        rel, size, sha = store(data, f"bigearthnet/{r['image_id']}")
        rows.append(make_row(rid=f"tr:fam10:BigEarthNet:{r['image_id']}", group_id=f"BigEarthNet:{r['image_id']}",
                             family="fam10", source="BigEarthNet", task_family="multiselect_landcover", images=[rel],
                             state=ben_state(rng, r), questions=questions, answer=answer,
                             meta={"template": tid, "suffix": sid, "options": [x for x, _ in opts], "n_true": len(pos),
                                   "labels": present, "image_sha256": sha, "orig_px": size, "licence": BEN_LICENCE}))
    return rows


def obj_multiselect(rng, items, n, source, imgs) -> list[dict]:
    pool = [t for t in items if len({o["c"] for o in t["objs"]}) >= 2]
    rows = []
    for t in rng.sample(pool, min(n, len(pool))):
        nm = names(t["src"])
        present = sorted({o["c"] for o in t["objs"]})
        rng.shuffle(present)
        pos = present[:5]
        k = min(10, max(len(pos) + 1, round(len(pos) / rng.uniform(*MS_YES_SHARE))))
        scene = sorted(t["scene_classes"] - set(present))
        other = sorted(set(nm) - set(present) - set(scene))
        rng.shuffle(scene)
        rng.shuffle(other)
        neg = []
        for _ in range(k - len(pos)):
            pool_ = scene if (scene and (rng.random() < 0.5 or not other)) else other
            if pool_:
                neg.append(pool_.pop())
        opts = [(c, True) for c in pos] + [(c, False) for c in neg]
        rng.shuffle(opts)
        tid, stem = rng.choice(MS_OBJ_T)
        questions, answer, sid = multiselect_questions(rng, stem, [(nm[c][1], y) for c, y in opts])
        rel, size, sha = imgs.get(t)
        rows.append(make_row(rid=f"tr:fam10:{source}:{t['id']}", group_id=f"{source}:{t['scene']}", family="fam10",
                             source=source, task_family="multiselect_objects", images=[rel], state="",
                             questions=questions, answer=answer,
                             meta={"template": tid, "suffix": sid, "options": [c for c, _ in opts], "n_true": len(pos),
                                   "image_sha256": sha, "orig_px": size,
                                   "licence": "DOTA: academic use only" if source == "DOTA" else "DIOR: CC BY-NC 4.0"}))
    return rows


def presence_lr(rng, recs, n) -> list[dict]:
    concept_of = {lab: c for c, (labs, _) in CONCEPTS.items() for lab in labs}
    by_concept = collections.defaultdict(list)
    for r in recs:
        present = {concept_of[x] for x in r["label_names"] if x in concept_of}
        if present and len(present) < len(CONCEPTS):
            for c in present:
                by_concept[c].append(r)
    used, picked = set(), []
    for c in sorted(CONCEPTS):  # each concept is the "yes" answer equally often
        cand = by_concept[c][:]
        rng.shuffle(cand)
        k = 0
        for r in cand:
            if k >= n // len(CONCEPTS):
                break
            if r["image_id"] not in used:
                used.add(r["image_id"])
                picked.append((r, c))
                k += 1
    yes_of = {r["image_id"]: c for r, c in picked}
    rows = []
    for r, data in by_parquet([r for r, _ in picked]):
        present = {concept_of[x] for x in r["label_names"] if x in concept_of}
        yes = yes_of[r["image_id"]]
        remote = set(r["label_names"]) <= REMOTE_LABELS
        absent = [c for c in sorted(set(CONCEPTS) - present) if remote or c not in SETTLEMENT_CONCEPTS]
        asked = [(yes, True)] + [(no, False) for no in rng.sample(absent, min(3, len(absent)))]
        rng.shuffle(asked)
        questions, answer, tids = {}, {}, []
        for j, (c, val) in enumerate(asked):
            tid, text = rng.choice(PRLR_T)
            questions[f"q{j}"] = {"type": "noul", "instructions": text.format(np=rng.choice(CONCEPTS[c][1])),
                                  "criteria": None}
            answer[f"q{j}"] = val
            tids.append(tid)
        rel, size, sha = store(data, f"bigearthnet/{r['image_id']}")
        rows.append(make_row(rid=f"tr:fam12:BigEarthNet:{r['image_id']}", group_id=f"BigEarthNet:{r['image_id']}",
                             family="fam12", source="BigEarthNet", task_family="presence_lowres", images=[rel],
                             state=ben_state(rng, r), questions=questions, answer=answer,
                             meta={"template": tids, "asked": asked, "labels": list(r["label_names"]),
                                   "image_sha256": sha, "orig_px": size, "licence": BEN_LICENCE}))
    return rows


def geo_rows(rng, n) -> list[dict]:
    sites = np.array([[m["lat"], m["lon"]] for m in (json.loads(l)["meta"] for l in
                      open(CONVERTED / "fmow" / "fmow_wilds_ood_2k.jsonl")) if m.get("lat") is not None])
    snap = Path(MAN["fmow"]["path"])
    cols = ["category", "country_code", "domain_labels_readable", "lat", "lon", "timestamp", "img_path"]
    recs, near = [], 0
    for f in sorted((snap / "WILDS").glob("train-*.parquet")):
        for i, r in enumerate(pq.read_table(f, columns=cols).to_pylist()):
            region = (r["domain_labels_readable"] or {}).get("region")
            country = country_name(r["country_code"] or "")
            if not country or region not in REGIONS or r["lat"] is None:
                continue
            d = np.abs(sites - [r["lat"], r["lon"]])
            if ((d[:, 0] < GEO_BUFFER_DEG) & (d[:, 1] < GEO_BUFFER_DEG)).any():
                near += 1
                continue
            recs.append({**r, "region": region, "country": country, "_f": f, "_i": i})
    print(f"  fMoW train images with country and continent: {len(recs)} (dropped {near} within "
          f"{GEO_BUFFER_DEG} deg of an eval site)", flush=True)
    by_region = collections.defaultdict(list)
    for r in recs:
        by_region[r["region"]].append(r)
    countries = {reg: sorted({r["country"] for r in rs}) for reg, rs in by_region.items()}
    all_countries = sorted({r["country"] for r in recs})
    picked = []
    for reg in sorted(by_region):
        picked += rng.sample(by_region[reg], min(n // len(by_region), len(by_region[reg])))
    files = collections.defaultdict(list)
    for r in picked:
        files[r["_f"]].append(r)
    rows = []
    for f, rs in files.items():
        images = pq.read_table(f, columns=["image"]).column("image")
        for r in rs:
            stem = r["img_path"].replace("/", "_").rsplit(".", 1)[0]
            k = 16 if rng.random() < 0.6 else rng.choice([4, 6, 8, 10])
            same = [c for c in countries[r["region"]] if c != r["country"]]
            distract = rng.sample(same, min(k - 1, len(same)))
            if len(distract) < k - 1:
                distract += rng.sample([c for c in all_countries if c != r["country"] and c not in distract],
                                       k - 1 - len(distract))
            ctry = [r["country"]] + distract
            rng.shuffle(ctry)
            conts = list(REGIONS.values())
            rng.shuffle(conts)
            (ct, ctext), (yt, ytext) = rng.choice(CONT_T), rng.choice(CTRY_T)
            rel, size, sha = store(images[r["_i"]].as_py()["bytes"], f"fmow/{stem}")
            state = f"High-resolution satellite image taken in {str(r['timestamp'])[:4]}." if rng.random() < 0.3 else ""
            rows.append(make_row(rid=f"tr:fam11:fMoW:{stem}", group_id=f"fMoW:{stem}", family="fam11", source="fMoW",
                                 task_family="geolocation", images=[rel], state=state,
                                 questions={"continent": {"type": "choice", "instructions": ctext,
                                                          "criteria": {o: None for o in conts}},
                                            "country": {"type": "choice", "instructions": ytext,
                                                        "criteria": {o: None for o in ctry}}},
                                 answer={"continent": REGIONS[r["region"]], "country": r["country"]},
                                 meta={"template": [ct, yt], "region": r["region"], "country_code": r["country_code"],
                                       "lat": r["lat"], "lon": r["lon"], "k_country": k, "image_sha256": sha,
                                       "orig_px": size, "licence": "fMoW Challenge Public License (non-commercial)"}))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0, help="only this many rows per part, for review")
    args = p.parse_args()
    n = {k: (args.sample or v) for k, v in TARGET.items()}
    recs = ben_records()
    fam10 = ben_multiselect(random.Random("fam10:ben"), recs, n["ms_ben"])
    imgs = ImageStore()
    obj = obj_multiselect(random.Random("fam10:dota"), dota_tiles(dota_scenes()), n["ms_dota"], "DOTA", imgs)
    obj += obj_multiselect(random.Random("fam10:dior"), dior_images(), n["ms_dior"], "DIOR", imgs)
    imgs.materialize()
    imgs.fill(obj)
    fam10 += obj
    fam12 = presence_lr(random.Random("fam12"), recs, n["presence_lr"])
    fam11 = geo_rows(random.Random("fam11"), n["geo"])
    suffix = ".sample" if args.sample else ""
    rng = random.Random(0)
    for name, rows in (("fam10_multiselect", fam10), ("fam11_geo", fam11), ("fam12_presence_lr", fam12)):
        write_jsonl(ROWS / f"{name}{suffix}.jsonl", rows)
        with open(ROWS / f"{name}{suffix}.txt", "w") as f:
            for r in (rows if args.sample else rng.sample(rows, min(30, len(rows)))):
                f.write(render(r) + "\n\n")
        yes = [v for r in rows for q, v in r["answer"].items() if r["questions"][q]["type"] == "noul"]
        print(f"{name}: {len(rows)} rows, {sum(len(r['questions']) for r in rows)} questions; by source "
              f"{dict(collections.Counter(r['meta']['source'] for r in rows))}; splits "
              f"{dict(collections.Counter(r['split'] for r in rows))}"
              + (f"; yes rate {sum(yes) / len(yes):.3f}" if yes else ""), flush=True)
    print("geo rows per continent:", dict(collections.Counter(r["meta"]["region"] for r in fam11)))
    print("presence concepts asked (yes):", dict(collections.Counter(
        c for r in fam12 for c, v in r["meta"]["asked"] if v)))


if __name__ == "__main__":
    main()
