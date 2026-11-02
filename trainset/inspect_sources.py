"""One-off inspection of the downloaded training sources, before writing the family converters. Writes a plain-text
report to $EOJEV_DATASETS/trainset/inspect_sources.txt.

Reports source layouts, counts after excluding eval images, whether the DOTA mirror's val labels match our official
copy, which VHM SFT rows look like HnstD (honesty) questions, and the scene names DisasterM3 eval rows use.

    MODULE=trainset.inspect_sources sbatch -J eojev-inspect slurm/convert_cpu.sbatch
"""

from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

from bench.common import CONVERTED, DATASETS, RAW

MAN = json.loads((DATASETS / "trainset" / "downloads.json").read_text())
OUT = DATASETS / "trainset" / "inspect_sources.txt"
LINES: list[str] = []


def say(*a) -> None:
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LINES.append(s)


def snap(name: str) -> Path:
    return Path(MAN[name]["path"])


def section(title: str) -> None:
    say("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def parquet_summary(files: list[Path], show: int = 2) -> None:
    n = 0
    for f in files:
        n += pq.ParquetFile(f).metadata.num_rows
    pf = pq.ParquetFile(files[0])
    say(f"files {len(files)}, rows {n}")
    say("schema:", ", ".join(f"{fld.name}:{fld.type}" for fld in pf.schema_arrow))
    meta = pf.schema_arrow.metadata or {}
    if b"huggingface" in meta:
        feats = json.loads(meta[b"huggingface"]).get("info", {}).get("features", {})
        for k, v in feats.items():
            if isinstance(v, dict) and "names" in v:
                say(f"label names [{k}] ({len(v['names'])}):", v["names"])
    cols = [c for c in pf.schema_arrow.names if c not in ("image", "t1_image", "t2_image", "t1_mask", "t2_mask")]
    rows = pf.read_row_group(0, columns=cols).slice(0, show).to_pylist()
    for r in rows:
        say("  sample:", {k: (str(v)[:120]) for k, v in r.items()})


def aid() -> None:
    section("AID (local raw/aid/AID.zip) minus the GeoChat 3k eval list")
    test = {json.loads(l)["image"].split("/")[1].rsplit(".", 1)[0] for l in open(RAW / "geochat-bench" / "aid.jsonl")}
    with zipfile.ZipFile(RAW / "aid" / "AID.zip") as zf:
        imgs = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".png", ".tif"))]
    per = collections.Counter()
    kept = 0
    for n in imgs:
        folder, fname = n.split("/")[-2:]
        if Path(fname).stem not in test:
            kept += 1
            per[folder] += 1
    say(f"images {len(imgs)}, eval list {len(test)}, kept {kept}; classes {len(per)}")
    say("kept per class:", dict(sorted(per.items())))


def eurosat() -> None:
    section("EuroSAT RGB (timm)")
    for split in ("train", "validation"):
        say(f"-- {split}")
        parquet_summary(sorted((snap("eurosat") / "data").glob(f"{split}-*.parquet")))


def bigearthnet() -> None:
    section("BigEarthNet v2 RGB (timm) train")
    parquet_summary(sorted((snap("bigearthnet") / "data").glob("train-*.parquet")))


def dota() -> None:
    section("DOTA (torchgeo mirror)")
    s = snap("dota")
    with open(s / "samples.csv") as f:
        rows = list(csv.DictReader(f))
    say("samples.csv columns:", list(rows[0].keys()))
    say("split x version counts:", dict(collections.Counter((r.get("split"), r.get("version")) for r in rows)))
    # mirror check: v2.0 val labels vs our official copy
    official: dict[str, bytes] = {}
    base = RAW / "dota-v2-val" / "labelTxt-v2.0"
    if base.is_dir():
        for p in base.rglob("*.txt"):
            official[p.name] = p.read_bytes()
        for z in base.rglob("*.zip"):
            if "hbb" in z.name.lower() or "meta" in z.name.lower():
                continue
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    if n.endswith(".txt"):
                        official.setdefault(Path(n).name, zf.read(n))
    say(f"official v2.0 val label files found locally: {len(official)}")
    same = diff = missing = 0
    examples = []
    with tarfile.open(s / "dotav2.0_annotations_val.tar.gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".txt")]
        say(f"mirror val annotation files: {len(members)}; e.g. {[m.name for m in members[:3]]}")
        for m in members:
            data = tf.extractfile(m).read()
            name = Path(m.name).name
            if name not in official:
                missing += 1
            elif hashlib.md5(data).digest() == hashlib.md5(official[name]).digest():
                same += 1
            else:
                diff += 1
                if len(examples) < 3:
                    examples.append(name)
    say(f"mirror vs official val labels: identical {same}, different {diff}, not in official {missing}; diffs e.g. {examples}")
    with tarfile.open(s / "dotav2.0_annotations_train.tar.gz") as tf:
        names = [m.name for m in tf.getmembers() if m.isfile()]
    say(f"train annotation files: {len(names)}; e.g. {names[:3]}")


def dior() -> None:
    section("DIOR (torchgeo images + official labels/splits)")
    s = snap("dior")
    for z in ("Images_trainval.zip", "Images_test.zip", "Annotations_trainval.zip"):
        with zipfile.ZipFile(s / z) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
        say(f"{z}: {len(names)} files; e.g. {names[:2]}")
    off = DATASETS / "raw" / "train" / "dior_official"
    with zipfile.ZipFile(off / "ImageSets.zip") as zf:
        for n in zf.namelist():
            if n.endswith(".txt"):
                say(f"ImageSets {n}: {len(zf.read(n).split())} ids")
    with zipfile.ZipFile(off / "Annotations.zip") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        tops = collections.Counter(n.split("/")[1] if n.count("/") > 1 else n.split("/")[0] for n in names)
        say(f"Annotations.zip: {len(names)} files; folders {dict(tops)}; e.g. {names[:2]}")
    rsvg = set()
    for split in ("test", "val"):
        p = RAW / "dior-rsvg" / f"{split}.txt"
        say(f"DIOR-RSVG {split}.txt present: {p.exists()}")
    pairs = RAW / "dior-rsvg" / "test_pairs_sorted_order.json"
    if pairs.exists():
        rsvg = {it["image"] for it in json.loads(pairs.read_text())}
        say(f"DIOR-RSVG test images (from reconstruction): {len(rsvg)}")


def xbd() -> None:
    section("xBD mirror train + tier3")
    s = snap("xbd")
    for split in ("train", "tier3"):
        say(f"-- {split}")
        parquet_summary(sorted((s / "data").glob(f"{split}-*.parquet")), show=3)
    events = collections.Counter()
    for f in sorted((s / "data").glob("*.parquet")):
        if f.name.startswith(("train", "tier3")):
            for nm in pq.read_table(f, columns=["image_name"]).column(0).to_pylist():
                events[re.sub(r"_\d+.*$", "", nm)] += 1
    say("scenes per event (train+tier3):", dict(sorted(events.items())))


def vhm() -> None:
    section("VHM SFT list (FitzPC/VHM_dataset_sft list_sft.json)")
    data = json.loads((snap("vhm_list") / "list_sft.json").read_text())
    say(f"type {type(data).__name__}, length {len(data)}")
    first = data[0] if isinstance(data, list) else next(iter(data.items()))
    say("first item:", json.dumps(first, ensure_ascii=False)[:800])
    items = data if isinstance(data, list) else list(data.values())
    say("keys seen:", dict(collections.Counter(k for it in items for k in it).most_common()))
    prefix = collections.Counter()
    by_prefix_samples = collections.defaultdict(list)
    pat = collections.Counter()
    hits = collections.defaultdict(list)
    for it in items:
        p = str(it.get("source_dataset", "<none>"))  # image paths carry no folder; group by source dataset
        prefix[p] += 1
        if len(by_prefix_samples[p]) < 2:
            by_prefix_samples[p].append(json.dumps(it, ensure_ascii=False)[:500])
        text = json.dumps(it.get("conversations", it), ensure_ascii=False).lower()
        for name, rx in {"honest/hnstd": r"honest|hnstd|decept",
                         "no such object": r"there is no|does not exist|no such|not present|cannot find",
                         "colour question": r"what (is the )?colou?r",
                         "presence yes/no": r"\bis there\b|\bare there\b",
                         "position 5-way": r"\(a\).*\(e\)|a\..*e\."}.items():
            if re.search(rx, text):
                pat[(p, name)] += 1
                if len(hits[name]) < 3:
                    hits[name].append(json.dumps(it, ensure_ascii=False)[:600])
    say("rows per image prefix:", dict(prefix.most_common()))
    for p, ss in by_prefix_samples.items():
        for s_ in ss:
            say(f"  [{p}] {s_}")
    say("pattern counts by (prefix, pattern):", {f"{k[0]}|{k[1]}": v for k, v in sorted(pat.items())})
    for name, ss in hits.items():
        for s_ in ss:
            say(f"  <{name}> {s_}")


def jev() -> None:
    section("jev-distill-corpus-v3 train.jsonl")
    kinds, sources, families = collections.Counter(), collections.Counter(), collections.Counter()
    n = 0
    sample = []
    with open(snap("jev") / "train.jsonl") as f:
        for line in f:
            r = json.loads(line)
            n += 1
            kinds[r.get("kind")] += 1
            sources[r.get("source")] += 1
            families[r.get("family")] += 1
            if len(sample) < 2:
                sample.append(line[:600])
    say(f"rows {n}; kinds {dict(kinds)}")
    say("sources:", dict(sources.most_common(20)))
    say("families:", dict(families.most_common(20)))
    for s_ in sample:
        say("  sample:", s_)


def aokvqa() -> None:
    section("A-OKVQA train")
    parquet_summary(sorted((snap("aokvqa") / "data").glob("train-*.parquet")))


def dm3_ids() -> None:
    section("DisasterM3 / TerraScope eval rows: recorded source ids (for xBD matching)")
    p = DATASETS / "trainset" / "blocklist" / "eval_source_ids.jsonl"
    seen = collections.defaultdict(list)
    for line in open(p):
        r = json.loads(line)
        b = r.get("benchmark") or ""
        if any(k in b for k in ("DisasterM3", "TerraScope", "xBD", "EarthMind", "BigEarthNet")) and len(seen[b]) < 3:
            seen[b].append(json.dumps(r, ensure_ascii=False)[:500])
    for b, ss in seen.items():
        for s_ in ss:
            say(f"  [{b}] {s_}")


def main() -> None:
    import sys  # noqa: PLC0415

    sections = {f.__name__: f for f in (aid, eurosat, bigearthnet, dota, dior, xbd, vhm, jev, aokvqa, dm3_ids)}
    chosen = sys.argv[1:] or list(sections)
    global OUT  # noqa: PLW0603
    if sys.argv[1:]:
        OUT = OUT.with_name(f"inspect_sources_{'_'.join(chosen)}.txt")
    for fn in (sections[c] for c in chosen):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - keep going; the report shows what failed
            say(f"!! {fn.__name__} failed: {type(exc).__name__}: {exc}")
    OUT.write_text("\n".join(LINES) + "\n")
    print(f"report: {OUT}")


if __name__ == "__main__":
    main()
