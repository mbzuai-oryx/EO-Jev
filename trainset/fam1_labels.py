"""Family 1, scene and land-use labels: ~20k training rows.

Sources, after removing eval material:
- AID: the 7,000 images outside the GeoChat 3k list (C6, C25); 5,000 rows, stratified by class.
- EuroSAT RGB: timm train split only (C6 uses timm test); 4,000 rows, stratified (with 6,000, sparse
  residential scenes collapsed on UCM / AID / NWPU).
- fMoW-WILDS: train split (images before 2013; C15 uses 2016+). Rows within ~100 m of any C15 location are dropped;
  6,000 rows, stratified over the 62 categories.
- BigEarthNet v2 S2 RGB: train split. Every patch named anywhere in the eval rows (C7, C24, EarthMind, ...) is
  dropped, together with its 8 neighbours in the same tile and date. 5,000 patches, each with 2–6 yes/no label questions:
  half present labels, half absent ones. Each absent label is confusable (same group as a present label, e.g. coniferous
  vs mixed forest) with probability HARD_NEG_SHARE, otherwise clearly absent.

Questions are choice (single label) or noul (multi-label), worded by templates that avoid the eval prompts' wording.
Option count, "none of these" rows and confusable distractors follow trainset/common.py. AID and EuroSAT rows carry
short class definitions as option criteria in 30% of rows. AID and fMoW scene questions also use "buildings"
(buildings_option): a wrong option on residential scenes, the answer's name on non-residential block scenes. Images are stored as their native bytes, as the eval
converters store them; the d3 runtime upsamples small images to its minimum.

    MODULE=trainset.fam1_labels [ARGS="--sample 12"] sbatch -J eojev-fam1 slurm/convert_cpu.sbatch
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

from bench.common import CONVERTED, RAW, write_jsonl
from bench.converters.bigearthnet import LABEL_TEXT
from bench.converters.scene_cls import EUROSAT_NAMES, readable
from trainset.common import ROWS, TRAIN, choice_options, make_row, render, store

MAN = json.loads((TRAIN / "downloads.json").read_text())
FAMILY = "fam1"
TARGET = {"AID": 5000, "EuroSAT": 4000, "fMoW": 6000, "BigEarthNet": 5000}
HARD_NEG_SHARE = 0.7  # share of confusable absent BigEarthNet labels (1.0 and 0.5 did worse; see ben_negatives)
# CORINE maps areas of at least 25 ha, so a patch without the urban label can still hold houses, farmsteads and small
# industry. Settlement labels are asked as absent only on patches whose labels are all remote land cover, where
# settlements are unlikely. Asking "not a residential area" on farm patches lost sparse-residential scenes on
# UCM / AID / NWPU, and RSVQA-LR presence (scored from OSM objects) stayed below d3-flash.
SETTLEMENT_LABELS = {"urban_fabric", "industrial_commercial_units"}
REMOTE_LABELS = {"broad_leaved_forest", "coniferous_forest", "mixed_forest", "transitional_woodland_shrub",
                 "natural_grassland_sparse_vegetation", "moors_heathland_sclerophyllous_vegetation",
                 "beaches_dunes_sands", "inland_wetlands", "coastal_wetlands", "inland_waters", "marine_waters"}
DEF_RATE = 0.30
STATE_RATE = 0.70
# UC Merced's "buildings" class: the model answered "buildings" for 45 of 100 dense residential and 25 of 100
# mobile home park scenes. No training question offered "buildings" as a scene option, while xBD's damage questions
# call residential roofs "buildings" 9,852 times. So "buildings" is now a wrong option on residential scenes and,
# with a definition, the answer's name for blocks of large non-residential buildings.
RESIDENTIAL = {"dense residential", "medium residential", "sparse residential", "single-unit residential",
               "multi-unit residential", "mobile home park"}
NONRESIDENTIAL = {"commercial", "center", "office building", "commercial area"}
BUILDINGS_DEF = "large non-residential buildings (offices, malls, warehouses) filling most of the image"
BUILDINGS_WRONG_RATE, BUILDINGS_RIGHT_RATE = 0.5, 0.6

SCENE_T = [("scene_v1", "What is the main land use shown in this image?"),
           ("scene_v2", "Which category best describes this aerial scene?"),
           ("scene_v3", "What kind of place does this image show?"),
           ("scene_v4", "Classify this image into one of the listed categories."),
           ("scene_v5", "Which of these labels fits this overhead image best?"),
           ("scene_v6", "This image was taken from above. What type of scene is it?"),
           ("scene_v7", "Select the scene type shown in the image."),
           ("scene_v8", "What does this overhead image mainly show?")]
FMOW_T = [("fac_v1", "What type of facility or site is shown in this image?"),
          ("fac_v2", "Which facility category best matches this satellite image?"),
          ("fac_v3", "What is the main structure or site in this image?"),
          ("fac_v4", "Classify the site shown in this satellite image."),
          ("fac_v5", "Which of these best describes what this image is centred on?"),
          ("fac_v6", "What kind of site occupies the centre of this image?")]
BEN_T = [("ben_v1", "Does this image contain {label}?"),
         ("ben_v2", "Does this image show {label}?"),  # templates must read well for plural labels too
         ("ben_v3", "Can {label} be seen in this image?"),
         ("ben_v4", "Is part of this area covered by {label}?"),
         ("ben_v5", "Does the land cover in this image include {label}?")]

AID_DEF = {
    "airport": "runways, taxiways, aprons and terminal buildings",
    "bare land": "exposed soil or ground without vegetation or buildings",
    "baseball field": "a diamond-shaped infield with a grass outfield",
    "beach": "a sandy shoreline along the sea or a lake",
    "bridge": "a structure carrying a road or railway over water",
    "center": "a dense city centre with large, tightly packed buildings",
    "church": "a church building, often cross-shaped, with its grounds",
    "commercial": "shops, malls and office blocks with parking",
    "dense residential": "closely packed houses or apartment blocks",
    "desert": "dry sand or rock with little or no vegetation",
    "farmland": "cultivated fields in regular plots",
    "forest": "continuous tree cover",
    "industrial": "factories, warehouses and industrial yards",
    "meadow": "open grassland without trees",
    "medium residential": "houses with gardens at moderate density",
    "mountain": "steep, rugged terrain with ridges and slopes",
    "park": "urban green space with paths, lawns and trees",
    "parking": "rows of parked vehicles on paved lots",
    "playground": "a sports ground or running track, often at a school",
    "pond": "a small body of still water",
    "port": "docks, piers and ships along a waterfront",
    "railway station": "platforms, converging tracks and a station building",
    "resort": "hotels or villas with pools and landscaped grounds",
    "river": "a flowing watercourse with banks",
    "school": "school buildings with a sports ground",
    "sparse residential": "scattered houses with large gaps between them",
    "square": "an open paved public plaza",
    "stadium": "an oval or rectangular sports arena with stands",
    "storage tanks": "groups of round tanks for oil or gas",
    "viaduct": "an elevated road or railway on many supports",
}
AID_GROUP = {**{c: "residential" for c in ("dense residential", "medium residential", "sparse residential")},
             **{c: "urban" for c in ("center", "commercial", "church", "school", "square", "resort", "park")},
             **{c: "industry" for c in ("industrial", "storage tanks")},
             **{c: "sports" for c in ("baseball field", "playground", "stadium")},
             **{c: "transport" for c in ("airport", "bridge", "parking", "railway station", "viaduct", "port")},
             **{c: "water" for c in ("beach", "pond", "river")},
             **{c: "natural" for c in ("bare land", "desert", "farmland", "forest", "meadow", "mountain")}}
EUROSAT_DEF = {
    "annual crop": "fields replanted every year, such as cereals",
    "forest": "continuous tree cover",
    "herbaceous vegetation": "natural grass or shrubland",
    "highway": "a major road crossing the area",
    "industrial": "industrial or commercial buildings",
    "pasture": "grassland used for grazing",
    "permanent crop": "orchards, vineyards or olive groves",
    "residential": "houses and residential streets",
    "river": "a river crossing the area",
    "sea or lake": "open water",
}
EUROSAT_GROUP = {"annual crop": "crop", "permanent crop": "crop", "pasture": "crop", "herbaceous vegetation": "veg",
                 "forest": "veg", "highway": "built", "industrial": "built", "residential": "built",
                 "river": "water", "sea or lake": "water"}
FMOW_GROUP_WORDS = [
    ("transport", "airport helipad runway ground_transportation interchange parking toll_booth tunnel road_bridge "
                  "railway_bridge gas_station flooded_road border_checkpoint"),
    ("industry", "factory nuclear oil_or_gas electric_substation solar_farm wind_farm smokestack storage_tank "
                 "surface_mine waste_disposal water_treatment dam shipyard port construction_site debris"),
    ("residential", "single-unit multi-unit impoverished"),
    ("public", "educational hospital fire_station police_station prison place_of_worship office_building "
               "shopping_mall car_dealership military space_facility burial archaeological"),
    ("recreation", "amusement golf park recreational stadium race_track swimming_pool zoo fountain"),
    ("land", "lake_or_pond aquaculture crop_field barn lighthouse tower"),
]
BEN_GROUP = {
    "broad_leaved_forest": "forest", "coniferous_forest": "forest", "mixed_forest": "forest",
    "transitional_woodland_shrub": "forest", "agro_forestry_areas": "forest",
    "arable_land": "agri", "pastures": "agri", "complex_cultivation_patterns": "agri", "permanent_crops": "agri",
    "agriculture_with_natural_vegetation": "agri",
    "inland_waters": "water", "marine_waters": "water", "coastal_wetlands": "water", "inland_wetlands": "water",
    "natural_grassland_sparse_vegetation": "natural", "moors_heathland_sclerophyllous_vegetation": "natural",
    "beaches_dunes_sands": "natural", "urban_fabric": "urban", "industrial_commercial_units": "urban",
}
# fMoW classes that are partly true of each other's images (an airport image shows runways and hangars). They are
# never offered as wrong options when "none of these" is the answer.
FMOW_OVERLAP_SETS = [{"airport", "airport hangar", "airport terminal", "runway", "helipad"},
                     {"factory or powerplant", "nuclear powerplant", "smokestack", "electric substation"},
                     {"port", "shipyard"}, {"park", "recreational facility", "amusement park"},
                     {"oil or gas facility", "storage tank"}, {"road bridge", "interchange", "flooded road"},
                     {"ground transportation station", "parking lot or garage"}]
FMOW_OVERLAPS = {c: s - {c} for s in FMOW_OVERLAP_SETS for c in s}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def fmow_group(cat: str) -> str:
    for g, words in FMOW_GROUP_WORDS:
        if any(w in cat for w in words.split()):
            return g
    return "other"


def stratified(rng: random.Random, items: list, key, n: int) -> list:
    by = collections.defaultdict(list)
    for it in items:
        by[key(it)].append(it)
    for v in by.values():
        rng.shuffle(v)
    out, k = [], 0
    while len(out) < n and any(by.values()):  # round-robin over classes until n
        for c in sorted(by):
            if by[c] and len(out) < n:
                out.append(by[c].pop())
        k += 1
    return out


def choice_row(rng, *, source, item, group_id, task_family, image_rel, size, sha, gold, classes, groups, templates,
               definitions=None, state="", extra=None, overlaps=None):
    tid, text = rng.choice(templates)
    options, answer, ometa = choice_options(rng, gold, classes, groups, overlaps=overlaps)
    use_def = bool(definitions) and rng.random() < DEF_RATE
    criteria = {o: (definitions.get(o) if use_def and definitions else None) for o in options}
    return make_row(rid=f"tr:{FAMILY}:{source}:{item}", group_id=group_id, family=FAMILY, source=source,
                    task_family=task_family, images=[image_rel], state=state,
                    questions={"q": {"type": "choice", "instructions": text, "criteria": criteria}},
                    answer={"q": answer},
                    meta={"template": tid, "gold_label": gold, "definitions": use_def, **ometa, "image_sha256": sha,
                          "orig_px": size, **(extra or {})})


def buildings_option(rng, row: dict) -> dict:
    """Add "buildings" to a scene question: a wrong option when the answer is a residential class
    (BUILDINGS_WRONG_RATE), or the answer's own name when it is a non-residential block class (BUILDINGS_RIGHT_RATE;
    the other block classes then leave the options, so that only one option fits)."""
    q, gold = row["questions"]["q"], row["meta"]["gold_label"]
    if "buildings" in q["criteria"]:
        return row
    entry = ("buildings", BUILDINGS_DEF if row["meta"]["definitions"] else None)
    items = list(q["criteria"].items())
    if gold in RESIDENTIAL and rng.random() < BUILDINGS_WRONG_RATE:
        items.insert(rng.randrange(len(items) + 1), entry)
        row["meta"]["buildings"] = "wrong option"
    elif gold in NONRESIDENTIAL and row["answer"]["q"] == gold and rng.random() < BUILDINGS_RIGHT_RATE:
        items = [entry if o == gold else (o, d) for o, d in items if o == gold or o not in NONRESIDENTIAL]
        row["answer"]["q"] = "buildings"
        row["meta"]["buildings"] = "answer"
    else:
        return row
    q["criteria"] = dict(items)
    row["meta"]["k"] = len(items)
    return row


def aid_rows(rng, n) -> list[dict]:
    test = {json.loads(l)["image"].split("/")[1].rsplit(".", 1)[0] for l in open(RAW / "geochat-bench" / "aid.jsonl")}
    with zipfile.ZipFile(RAW / "aid" / "AID.zip") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".jpg") and Path(m).stem not in test]
        picked = stratified(rng, members, lambda m: m.split("/")[-2], n)
        classes = sorted({readable(m.split("/")[-2]) for m in members})
        rows = []
        for m in picked:
            folder, stem = m.split("/")[-2], Path(m).stem
            rel, size, sha = store(zf.read(m), f"aid/{folder}/{stem}")
            state = "Google Earth image, about 0.5–8 m per pixel." if rng.random() < STATE_RATE else ""
            rows.append(buildings_option(rng, choice_row(
                rng, source="AID", item=stem, group_id=f"AID:{stem}", task_family="scene_cls", image_rel=rel,
                size=size, sha=sha, gold=readable(folder), classes=classes, groups=AID_GROUP, templates=SCENE_T,
                definitions=AID_DEF, state=state, extra={"licence": "AID: research use (no licence stated)"})))
    return rows


def eurosat_rows(rng, n) -> list[dict]:
    snap = Path(MAN["eurosat"]["path"])
    f = sorted((snap / "data").glob("train-*.parquet"))[0]
    meta = json.loads(pq.ParquetFile(f).schema_arrow.metadata[b"huggingface"])
    names = [EUROSAT_NAMES[x] for x in meta["info"]["features"]["label"]["names"]]
    table = pq.read_table(f).to_pylist()
    picked = stratified(rng, table, lambda r: r["label"], n)
    rows = []
    for r in picked:
        gold = names[r["label"]]
        rel, size, sha = store(r["image"]["bytes"], f"eurosat/{r['image_id']}")
        state = "Sentinel-2 true-colour image, 10 m per pixel." if rng.random() < STATE_RATE else ""
        rows.append(choice_row(rng, source="EuroSAT", item=r["image_id"], group_id=f"EuroSAT:{r['image_id']}",
                               task_family="scene_cls", image_rel=rel, size=size, sha=sha, gold=gold,
                               classes=sorted(names), groups=EUROSAT_GROUP, templates=SCENE_T,
                               definitions=EUROSAT_DEF, state=state, extra={"licence": "MIT"}))
    return rows


def fmow_rows(rng, n) -> list[dict]:
    blocked = set()
    for line in open(CONVERTED / "fmow" / "fmow_wilds_ood_2k.jsonl"):
        m = json.loads(line)["meta"]
        if m.get("lat") is not None:
            blocked.add((round(m["lat"], 3), round(m["lon"], 3)))
    snap = Path(MAN["fmow"]["path"])
    files = sorted((snap / "WILDS").glob("train-*.parquet"))
    cols = ["category", "lat", "lon", "timestamp", "img_path"]
    recs = []
    n_blocked = 0
    for f in files:
        for i, r in enumerate(pq.read_table(f, columns=cols).to_pylist()):
            if r["lat"] is not None and (round(r["lat"], 3), round(r["lon"], 3)) in blocked:
                n_blocked += 1
                continue
            r["_f"], r["_i"] = f, i
            recs.append(r)
    print(f"  fMoW train rows {len(recs)} (dropped {n_blocked} at C15 locations)", flush=True)
    classes = sorted({r["category"].replace("_", " ") for r in recs})
    groups = {c: fmow_group(c.replace(" ", "_")) for c in classes}
    picked = stratified(rng, recs, lambda r: r["category"], n)
    by_file = collections.defaultdict(list)
    for r in picked:
        by_file[r["_f"]].append(r)
    rows = []
    for f, rs in by_file.items():
        images = pq.read_table(f, columns=["image"]).column("image")
        for r in rs:
            stem = r["img_path"].replace("/", "_").rsplit(".", 1)[0]
            rel, size, sha = store(images[r["_i"]].as_py()["bytes"], f"fmow/{stem}")
            year = str(r["timestamp"])[:4]
            state = f"High-resolution satellite image taken in {year}." if rng.random() < STATE_RATE else ""
            rows.append(buildings_option(rng, choice_row(
                rng, source="fMoW", item=stem, group_id=f"fMoW:{stem}", task_family="facility_cls", image_rel=rel,
                size=size, sha=sha, gold=r["category"].replace("_", " "), classes=classes, groups=groups,
                templates=FMOW_T, state=state, overlaps=FMOW_OVERLAPS,
                extra={"licence": "fMoW Challenge Public License (non-commercial)", "lat": r["lat"], "lon": r["lon"]})))
    return rows


V2 = re.compile(r"(S2[AB]_MSIL2A_\d{8}T\d{6}_N\d{4}_R\d{3}_T\w{5})_(\d+)_(\d+)")
V1 = re.compile(r"S2[AB]_MSIL2A_\d{8}T\d{6}_\d+_\d+")


def ben_blocklist() -> tuple[set, set]:
    """Eval patch neighbourhoods (v2 names, ±1 row/col in the same tile and date) and eval v1 names."""
    v2, v1 = set(), set()
    with open(TRAIN / "blocklist" / "eval_source_ids.jsonl") as f:
        for line in f:
            for m in V2.finditer(line):
                prefix, r, c = m.group(1), int(m.group(2)), int(m.group(3))
                v2.update((prefix, r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
            for m in V1.finditer(line):
                if "_N9999_" not in m.group(0):
                    v1.add(m.group(0))
    return v2, v1


def ben_records() -> list[dict]:
    """BigEarthNet v2 train patches outside the eval blocklist; each record knows its parquet file (`_f`) and row (`_i`)."""
    v2_block, v1_block = ben_blocklist()
    print(f"  BigEarthNet blocklist: {len(v2_block) // 9} eval patches (v2 names, with neighbours: {len(v2_block)}), "
          f"{len(v1_block)} v1 names", flush=True)
    snap = Path(MAN["bigearthnet"]["path"])
    files = sorted((snap / "data").glob("train-*.parquet"))
    cols = ["label_names", "image_id", "country", "acquisition_time", "s2v1_name"]
    recs, dropped = [], 0
    for f in files:
        for i, r in enumerate(pq.read_table(f, columns=cols).to_pylist()):
            m = V2.match(r["image_id"] or "")
            key = (m.group(1), int(m.group(2)), int(m.group(3))) if m else None
            if (key and key in v2_block) or r["s2v1_name"] in v1_block:
                dropped += 1
                continue
            r["_f"], r["_i"] = f, i
            recs.append(r)
    print(f"  BigEarthNet train patches {len(recs)} (dropped {dropped} eval patches or their neighbours)", flush=True)
    return recs


def ben_state(rng, r) -> str:
    t = r["acquisition_time"]
    when = f"{MONTHS[t.month - 1]} {t.year}" if hasattr(t, "month") else str(t)[:7]
    return f"Sentinel-2 true-colour image, 10 m per pixel, {r['country']}, {when}." if rng.random() < STATE_RATE else ""


def ben_negatives(rng, present: list[str], k: int) -> list[str]:
    """k absent labels: each one confusable (same group as a present label) with probability HARD_NEG_SHARE, otherwise
    clearly absent (another group). Only confusable ones made the model reluctant to say yes to plausible labels;
    50/50 overshot (yes to look-alike labels in BigEarthNet, EuroSAT crop classes merged).
    Settlement labels count as absent only on remote patches (SETTLEMENT_LABELS)."""
    remote = set(present) <= REMOTE_LABELS
    absent = [x for x in sorted(LABEL_TEXT) if x not in present and (remote or x not in SETTLEMENT_LABELS)]
    groups = {BEN_GROUP.get(p) for p in present}
    hard = [x for x in absent if BEN_GROUP.get(x) in groups]
    easy = [x for x in absent if BEN_GROUP.get(x) not in groups]
    rng.shuffle(hard)
    rng.shuffle(easy)
    out = []
    for _ in range(min(k, len(absent))):
        pool = hard if (hard and (rng.random() < HARD_NEG_SHARE or not easy)) else easy
        out.append(pool.pop())
    return out


def ben_rows(rng, n) -> list[dict]:
    recs = ben_records()
    picked = rng.sample(recs, n)
    by_file = collections.defaultdict(list)
    for r in picked:
        by_file[r["_f"]].append(r)
    rows = []
    for f, rs in by_file.items():
        images = pq.read_table(f, columns=["image"]).column("image")
        for r in rs:
            present = list(r["label_names"])
            rng.shuffle(present)
            n_pos = min(len(present), rng.choice([1, 2, 3]))
            neg = ben_negatives(rng, present, n_pos)
            asked = [(x, True) for x in present[:n_pos]] + [(x, False) for x in neg]
            rng.shuffle(asked)
            questions, answer, tids = {}, {}, []
            for j, (lab, yes) in enumerate(asked):
                tid, text = rng.choice(BEN_T)
                questions[f"q{j}"] = {"type": "noul", "instructions": text.format(label=LABEL_TEXT[lab]),
                                      "criteria": None}
                answer[f"q{j}"] = yes
                tids.append(tid)
            rel, size, sha = store(images[r["_i"]].as_py()["bytes"], f"bigearthnet/{r['image_id']}")
            state = ben_state(rng, r)
            rows.append(make_row(rid=f"tr:{FAMILY}:BigEarthNet:{r['image_id']}", group_id=f"BigEarthNet:{r['image_id']}",
                                 family=FAMILY, source="BigEarthNet", task_family="lulc_multilabel", images=[rel],
                                 state=state, questions=questions, answer=answer,
                                 meta={"template": tids, "labels": present, "image_sha256": sha, "orig_px": size,
                                       "licence": "CDLA-Permissive-1.0"}))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0, help="only this many rows per source, for review")
    args = p.parse_args()
    rng = random.Random(0)
    builders = {"AID": aid_rows, "EuroSAT": eurosat_rows, "fMoW": fmow_rows, "BigEarthNet": ben_rows}
    rows = []
    for name, fn in builders.items():
        n = args.sample or TARGET[name]
        part = fn(random.Random(f"fam1:{name}"), n)
        print(f"{name}: {len(part)} rows", flush=True)
        rows += part
    suffix = ".sample" if args.sample else ""
    out = ROWS / f"fam1_labels{suffix}.jsonl"
    write_jsonl(out, rows)
    with open(ROWS / f"fam1_labels{suffix}.txt", "w") as f:
        shown = collections.Counter()
        for r in rows:
            src = r["meta"]["source"]
            if args.sample or shown[src] < 6:
                f.write(render(r) + "\n\n")
                shown[src] += 1
    stats = collections.Counter((r["meta"]["source"], r["split"]) for r in rows)
    nones = collections.Counter(r["meta"].get("none") for r in rows)
    print(f"wrote {len(rows)} rows to {out}; by source/split {dict(stats)}; none-of-these {dict(nones)}")


if __name__ == "__main__":
    main()
