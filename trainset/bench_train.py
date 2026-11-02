"""Training rows from the benchmarks' own training splits (the eval images are kept out by trainset/check_leaks.py). One sub-command per source; each writes rows/<family>.jsonl with the
benchmark converter's question and answer format, images under $EOJEV_DATASETS/trainset/images.

- rsvqa -> fam15_rsvqa: RSVQA-LR train + val (presence, comparison, rural / urban, count bins; as C4 LR) and
  RSVQA-HR train + val (presence, comparison; as C4 HR). Answers come from OpenStreetMap / land-use data by the
  dataset's own rules (no LLM). RSVQA has ~100 questions per image, so at most PER_IMAGE questions per image are kept,
  round-robin over question types.
- vrsbench -> fam16_vrsbench: VRSBench train VQA (85,813 questions on 20,262 DOTA / DIOR train tiles; C3 uses the
  val split), converted as C3: yes / no -> noul, counts -> the 8-level score, other answers -> a choice over C3's
  closed vocabulary of the question's type (inferred from the wording, since the train file has no type field;
  answers outside the vocabulary are dropped). VRSBench's answers were drafted by GPT-4V and checked by people.
- cdvqa -> fam17_cdvqa: CDVQA train + val (C8 uses test-2), SECOND image pairs, converted as C8 (yes / no ->
  noul; largest / smallest change and change-to-what -> 6-class choice; change ratios -> the 11-bin choice).
  Answers come from SECOND's change maps by CDVQA's rules (no LLM). ~41 questions per pair, so at most
  PER_PAIR per pair, round-robin over question types.
- disasterm3 -> fam18_disasterm3: DisasterM3 Instruct (C10 uses the Bench set), its six closed-form tasks converted
  as C10 (single-choice -> choice; Disaster Bearing Bodies / Scene Recognition -> one noul per option). Items with a
  SAR image are left out (2026-10-29). DisasterM3 says its annotations were repurposed and manually verified.
- dvl -> fam20_dvl: DVL-Suite train, its two closed-form files (basic change, change speed; the same files C12
  converts from test), converted as C12 (single-choice -> choice; multi-select -> one noul per option; the dates in
  the state). Basic change has ~43 items per scene, so at most PER_SCENE per scene (round-robin over single / multi);
  change speed (~3 per scene) is kept whole. Images are downscaled to DVL_SIDE px (C12 feeds the 1024 px NAIP tiles,
  ~1,000 image tokens each; 6-7 per question would make a DVL question 5-10x the cost of the others). DVL's
  annotations were made by domain experts with GPT-4.1.
- nwpu -> fam19_nwpu: NWPU-RESISC45 train + validation (C6 uses the test split), 45-way scene labels asked as
  family 1 asks them (option count, "none of these", confusable distractors), plus family 1's "buildings" option.

    MODULE=trainset.bench_train ARGS="rsvqa" sbatch -J eojev-btrain slurm/convert_cpu.sbatch
"""

from __future__ import annotations

import collections
import io
import json
import random
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from bench.common import RAW, write_jsonl
from bench.converters import rsvqa as c4
from bench.converters.scene_cls import readable
from trainset.common import ROWS, make_row, render, store
from trainset.fam1_labels import SCENE_T, STATE_RATE, buildings_option, choice_row

PER_IMAGE = {"RSVQA-LR": 25, "RSVQA-HR": 8}


def write(name: str, rows: list[dict], rng: random.Random) -> None:
    write_jsonl(ROWS / f"{name}.jsonl", rows)
    with open(ROWS / f"{name}.txt", "w") as f:
        for r in rng.sample(rows, min(30, len(rows))):
            f.write(render(r) + "\n\n")
    yes = [a for r in rows for a in r["answer"].values() if isinstance(a, bool)]
    print(f"{name}: {len(rows)} rows; task families {dict(collections.Counter(r['task_family'] for r in rows))}; "
          f"splits {dict(collections.Counter(r['split'] for r in rows))}"
          + (f"; yes share {100 * sum(yes) / len(yes):.1f}% of {len(yes)} yes/no" if yes else ""), flush=True)


def per_image(rng, questions: list[dict], cap: int) -> list[dict]:
    """At most cap questions per image, round-robin over question types."""
    by_img = collections.defaultdict(lambda: collections.defaultdict(list))
    for q in questions:
        by_img[q["img_id"]][q["type"]].append(q)
    out = []
    for types in by_img.values():
        for v in types.values():
            rng.shuffle(v)
        keep = []
        while len(keep) < cap and any(types.values()):
            for t in sorted(types):
                if types[t] and len(keep) < cap:
                    keep.append(types[t].pop())
        out += keep
    return out


def rsvqa() -> None:
    rng = random.Random("fam15")
    rows = []
    for bench, src, prefixes, gsd, types in (
            ("RSVQA-LR", RAW / "rsvqa-lr", ("LR_split_train", "LR_split_val"), 10.0, None),
            ("RSVQA-HR", RAW / "rsvqa-hr", ("USGS_split_train", "USGS_split_val"), 0.15, ("presence", "comp"))):
        questions, gold = [], {}
        for prefix in prefixes:
            qs, g = c4.load(src, prefix)
            questions += [q for q in qs if types is None or q["type"] in types]
            gold.update(g)
        questions = per_image(rng, questions, PER_IMAGE[bench])
        need = {q["img_id"] for q in questions}
        images = {}
        if bench == "RSVQA-LR":
            with zipfile.ZipFile(src / "Images_LR.zip") as zf:
                for img_id in sorted(need):
                    images[img_id] = store(zf.read(f"Images_LR/{img_id}.tif"), f"rsvqa/lr/{img_id}")
        else:
            with tarfile.open(src / "Images.tar") as tf:
                for m in tf:
                    stem = Path(m.name).stem
                    if m.isfile() and stem.isdigit() and int(stem) in need:
                        images[int(stem)] = store(tf.extractfile(m).read(), f"rsvqa/hr/{stem}")
        missing = need - set(images)
        if missing:
            raise FileNotFoundError(f"{bench}: {len(missing)} images missing, e.g. {sorted(missing)[:5]}")
        for q in questions:
            ev = c4.make_row(bench, q, gold[q["id"]], "x.png", gsd)  # the eval row's question and answer
            if ev is None:
                continue
            rel, size, sha = images[q["img_id"]]
            rows.append(make_row(rid=f"tr:fam15:{bench}:{q['id']}", group_id=f"{bench}:img{q['img_id']}",
                                 family="fam15", source=bench, task_family=ev["task_family"], images=[rel], state="",
                                 questions=ev["questions"], answer=ev["answer"],
                                 meta={**ev["meta"], "image_sha256": sha, "orig_px": size,
                                       "licence": "RSVQA: CC BY 4.0"}))
        print(f"  {bench}: {len(questions)} questions kept on {len(need)} images", flush=True)
    write("fam15_rsvqa", rows, rng)


VRS_TYPE_WORDS = [("object color", ("color", "colour")), ("object shape", ("shape",)),
                  ("object direction", ("direction", "oriented", "orientation", "facing", "aligned")),
                  ("rural or urban", ("rural or urban", "urban or rural")),
                  ("scene type", ("scene", "type of area", "kind of area", "setting")),
                  ("object position", ("where", "located", "position", "which part", "which side", "most"))]


def vrs_type(question: str) -> str:
    q = question.lower()
    return next((t for t, words in VRS_TYPE_WORDS if any(w in q for w in words)), "object category")


def vrsbench() -> None:
    from bench.converters import vrsbench as c3  # noqa: PLC0415

    rng = random.Random("fam16")
    ev = json.load(open(c3.SRC / "VRSBench_EVAL_vqa.json"))
    vocab = {}
    for qtype in [k for k in c3.QUOTA if k not in ("yes/no", "object quantity")]:  # C3's vocabularies, as C3 builds them
        counts = collections.Counter(c3.normalize(x["ground_truth"]) for x in ev if x["type"] == qtype)
        counts.pop("yes", None), counts.pop("no", None)
        vocab[qtype] = c3.FIXED_VOCAB.get(qtype) or [a for a, c in counts.most_common(16) if c >= 15]
    items = [x for x in json.load(open(c3.SRC / "VRSBench_train.json"))
             if x["conversations"][0]["value"].startswith("<image>\n[vqa]")]
    kept, dropped = [], collections.Counter()
    for i, x in enumerate(items):
        text = x["conversations"][0]["value"].split("[vqa]", 1)[1].strip()
        text = text.replace(". A short answer to the question is", "").strip()
        gt = c3.normalize(x["conversations"][1]["value"])
        if gt in ("yes", "no"):
            kept.append((x, i, "yes/no", {"type": "noul", "instructions": text}, gt == "yes"))
        elif (level := c3.count_level(gt)) is not None:
            kept.append((x, i, "object quantity",
                         {"type": "score", "instructions": text, "criteria": c3.COUNT_LEVELS}, level))
        else:
            t = vrs_type(text)
            if gt not in vocab[t]:
                dropped[t] += 1
                continue
            opts = vocab[t][:]
            rng.shuffle(opts)
            kept.append((x, i, t, {"type": "choice", "instructions": text, "criteria": {o: None for o in opts}}, gt))
    print(f"  VRSBench train VQA {len(items)}: kept {len(kept)}; dropped (answer outside C3's vocabulary) {dict(dropped)}",
          flush=True)
    need = {x["image"] for x, *_ in kept}
    images = {}
    with zipfile.ZipFile(c3.SRC / "Images_train.zip") as zf:
        members = {Path(n).name: n for n in zf.namelist() if not n.endswith("/")}
        for name in sorted(need):
            images[name] = store(zf.read(members[name]), f"vrsbench/{Path(name).stem}")
    rows = []
    for x, i, kind, question, gold in kept:
        rel, size, sha = images[x["image"]]
        rows.append(make_row(rid=f"tr:fam16:VRSBench:{i}", group_id=f"VRSBench:{x['image']}", family="fam16",
                             source="VRSBench", task_family="vqa_" + kind.replace(" ", "_").replace("/", ""),
                             images=[rel], state="", questions={"q": question}, answer={"q": gold},
                             meta={"kind": kind, "raw_answer": x["conversations"][1]["value"], "image_sha256": sha,
                                   "orig_px": size, "licence": "VRSBench: CC BY 4.0 (GPT-4V drafted, human verified)"}))
    print(f"  kinds {dict(collections.Counter(r['meta']['kind'] for r in rows))}", flush=True)
    write("fam16_vrsbench", rows, rng)


PER_PAIR = 10


def cdvqa() -> None:
    from bench.converters import cdvqa as c8  # noqa: PLC0415

    rng = random.Random("fam17")
    samples = []
    for split in ("train", "val"):
        for tar_path in sorted((RAW / "cdvqa-wds" / split).glob(f"{split}-*.tar")):
            with tarfile.open(tar_path) as tf:
                for m in tf.getmembers():
                    if m.name.endswith(".json"):
                        rec = json.loads(tf.extractfile(m).read())
                        q = re.sub(r"^(Image \d: <image>\n)+", "", rec["conversations"][0]["value"]).strip()
                        samples.append({"tar": tar_path, "key": m.name[: -len(".json")], "meta": rec["meta"], "q": q,
                                        "a": rec["conversations"][1]["value"], "type": rec["meta"]["question_type"],
                                        "img_id": rec["meta"]["image_id"], "split": split})
    picked = per_image(rng, samples, PER_PAIR)
    print(f"  CDVQA train + val: {len(samples)} questions on {len({s['img_id'] for s in samples})} pairs; "
          f"kept {len(picked)}", flush=True)
    by_tar = collections.defaultdict(list)
    for s in picked:
        by_tar[s["tar"]].append(s)
    stored, rows = {}, []
    for tar_path, ss in sorted(by_tar.items()):
        with tarfile.open(tar_path) as tf:
            for s in ss:
                stem = s["img_id"].rsplit(".", 1)[0]
                if stem not in stored:
                    stored[stem] = [store(tf.extractfile(f"{s['key']}.{i}.img").read(), f"cdvqa/{stem}_{i}")
                                    for i in (0, 1)]
                ims, qtype, answer = stored[stem], s["type"], s["a"]
                if qtype in ("change_or_not", "increase_or_not", "decrease_or_not"):
                    q, gold = {"type": "noul", "instructions": s["q"]}, answer == "yes"
                elif qtype in ("largest_change", "smallest_change", "change_to_what"):
                    opts = list(c8.CLASSES.values())
                    rng.shuffle(opts)
                    q, gold = {"type": "choice", "instructions": s["q"], "criteria": {o: None for o in opts}}, \
                        c8.CLASSES[answer]
                else:
                    q = {"type": "choice", "instructions": s["q"],
                         "criteria": {c8.BIN_TEXT[b]: None for b in c8.BINS}}
                    gold = c8.BIN_TEXT[answer]
                rows.append(make_row(rid=f"tr:fam17:CDVQA:{s['split']}:{s['meta']['question_id']}", group_id=f"CDVQA:{stem}",
                                     family="fam17", source="CDVQA", task_family=qtype,
                                     images=[ims[0][0], ims[1][0]], state=c8.STATE, questions={"q": q},
                                     answer={"q": gold},
                                     meta={"raw_answer": answer, "image_sha256": [ims[0][2], ims[1][2]],
                                           "orig_px": ims[0][1], "licence": "CDVQA / SECOND: research use"}))
    write("fam17_cdvqa", rows, rng)


def disasterm3() -> None:
    from bench.converters import disasterm3 as c10  # noqa: PLC0415

    rng = random.Random("fam18")
    tasks = {t.lower(): t for t in c10.QUOTA}  # the train file writes "relational reasoning" in lower case
    items = json.load(open(RAW / "dm3-json" / "train_release.json"))
    keep, skipped = [], collections.Counter()
    for i, x in enumerate(items):
        task = tasks.get(x["task"].lower())
        modality = x.get("post_image_type") or x.get("image_type")
        if task is None:
            continue
        if modality == "SAR":
            skipped["SAR"] += 1
            continue
        keep.append((i, task, modality, x))
    src = RAW / "disasterm3" / "DisasterM3_Instruct"
    zips = [zipfile.ZipFile(src / z) for z in ("train_images.zip", "box_train_images.zip")]
    members = {Path(n).name: (zf, n) for zf in zips for n in zf.namelist() if not n.endswith("/")}
    stored, rows = {}, []
    for i, task, modality, x in keep:
        options = [str(o) for o in x["options_list"]]
        letters = [s.strip() for s in str(x["ground_truth_option"]).strip().rstrip(".").split(",")]
        try:
            golds = {options[ord(s) - ord("A")] for s in letters}
        except IndexError:
            skipped["bad gold letter"] += 1
            continue
        if "pre_image_path" in x:
            srcs = [c10.path(x["pre_image_path"]), c10.path(x["post_image_path"])]
            state = f"Image 1 was taken before the disaster (optical); Image 2 after it ({modality})."
        else:
            srcs = [c10.path(x["image_path"])]
            state = f"A {modality} image taken after the disaster."
        try:
            for s in srcs:
                if s not in stored:
                    zf, n = members[Path(s).name]
                    stored[s] = store(zf.read(n), f"disasterm3/{Path(s).stem}")
        except Exception:  # noqa: BLE001 - missing or unreadable image (the bench release has a zero-byte one)
            skipped["unreadable image"] += 1
            continue
        multi = task in c10.MULTI
        if multi:
            questions = {f"opt{k}": {"type": "noul", "instructions": f"{x['prompts']} Is \"{o}\" one of them?"}
                         for k, o in enumerate(options)}
            answer = {f"opt{k}": o in golds for k, o in enumerate(options)}
        else:
            if len(set(options)) != len(options) or len(golds) != 1:
                skipped["not single-answer"] += 1
                continue
            questions = {"q": {"type": "choice", "instructions": x["prompts"], "criteria": {o: None for o in options}}}
            answer = {"q": golds.pop()}
        ims = [stored[s] for s in srcs]
        rows.append(make_row(rid=f"tr:fam18:DisasterM3:{i}", group_id=f"DisasterM3:{srcs[-1]}", family="fam18",
                             source="DisasterM3", task_family=task.lower().replace(" ", "_") + ("_multi" if multi else ""),
                             images=[im[0] for im in ims], state=state, questions=questions, answer=answer,
                             meta={"task": task, "post_modality": modality, "multi_select": multi,
                                   "image_sha256": [im[2] for im in ims], "orig_px": ims[-1][1],
                                   "licence": "DisasterM3: CC BY-NC-SA 4.0"}))
    for zf in zips:
        zf.close()
    print(f"  DisasterM3 Instruct closed-form items: kept {len(rows)}, skipped {dict(skipped)}", flush=True)
    write("fam18_disasterm3", rows, rng)


PER_SCENE, DVL_SIDE = 12, 768


def dvl() -> None:
    from bench.converters import dvl as c12  # noqa: PLC0415

    rng = random.Random("fam20")
    zf = zipfile.ZipFile(RAW / "dvl" / "train.zip")
    picked = []
    for fname, family in c12.FILES.items():
        items = json.loads(zf.read(f"train/{fname}"))
        if family == "dvl_basic_change":
            by_scene = collections.defaultdict(lambda: collections.defaultdict(list))
            for k, it in enumerate(items):
                by_scene["/".join(it["image_list"][0].split("/")[1:3])][it["task_type"]].append((k, it))
            for types in by_scene.values():
                for v in types.values():
                    rng.shuffle(v)
                keep = []
                while len(keep) < PER_SCENE and any(types.values()):
                    for t in sorted(types):
                        if types[t] and len(keep) < PER_SCENE:
                            keep.append(types[t].pop())
                picked += [(family, k, it) for k, it in keep]
        else:
            picked += [(family, k, it) for k, it in enumerate(items)]
    stored, rows, skipped = {}, [], collections.Counter()
    for family, k, it in picked:
        options = [str(o) for o in it["options_list"]]
        golds = {options[ord(s.strip()) - ord("A")] for s in it["ground_truth_option"].strip().rstrip(".").split(",")}
        if len(set(options)) != len(options):
            skipped["duplicate options"] += 1
            continue
        multi = it["task_type"] == "Multiple choice"
        if multi:
            questions = {f"opt{i}": {"type": "noul", "instructions": f"{it['prompts']} Is \"{o}\" part of the correct answer?"}
                         for i, o in enumerate(options)}
            answer = {f"opt{i}": o in golds for i, o in enumerate(options)}
        elif len(golds) == 1:
            questions = {"q": {"type": "choice", "instructions": it["prompts"], "criteria": {o: None for o in options}}}
            answer = {"q": golds.pop()}
        else:
            skipped["single-choice with several answers"] += 1
            continue
        for p in it["image_list"]:
            if p not in stored:
                im = Image.open(io.BytesIO(zf.read(f"train/{p}"))).convert("RGB")
                im.thumbnail((DVL_SIDE, DVL_SIDE), Image.Resampling.BICUBIC)
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                stored[p] = store(buf.getvalue(), f"dvl/{p.removeprefix('images/').rsplit('.', 1)[0]}")
        ims = [stored[p] for p in it["image_list"]]
        timeline = ", ".join(f"Image {i + 1} = {c12.date(t)}" for i, t in enumerate(it["time_stamps"]))
        scene = "/".join(it["image_list"][0].split("/")[1:3])
        rows.append(make_row(rid=f"tr:fam20:DVL:{family}:{k}", group_id=f"DVL:{scene}", family="fam20", source="DVL",
                             task_family=family + ("_multi" if multi else ""), images=[im[0] for im in ims],
                             state=f"The images show the same area over time, in chronological order: {timeline}.",
                             questions=questions, answer=answer,
                             meta={"task": it["task"], "n_images": len(ims), "multi_select": multi,
                                   "image_sha256": [im[2] for im in ims], "orig_px": ims[0][1],
                                   "licence": "DVL-Suite: Apache-2.0 (experts + GPT-4.1)"}))
    zf.close()
    print(f"  DVL train items kept {len(rows)} of {len(picked)} picked; skipped {dict(skipped)}; "
          f"{len(stored)} images at <= {DVL_SIDE} px", flush=True)
    write("fam20_dvl", rows, rng)


NWPU_GROUPS = [("residential", "residential mobile"), ("urban", "commercial church palace"),
               ("industry", "industrial storage thermal"),
               ("sports", "baseball basketball golf ground stadium tennis"),
               ("transport", "airplane airport bridge freeway intersection overpass parking railway roundabout runway "
                             "ship harbor"),
               ("water", "beach lake river sea wetland island"),
               ("natural", "chaparral cloud desert forest meadow mountain snowberg terrace farmland")]


def nwpu() -> None:
    rng = random.Random("fam19")
    data = RAW / "resisc45" / "data"
    meta = json.loads(pq.ParquetFile(sorted(data.glob("test-*"))[0]).schema_arrow.metadata[b"huggingface"])
    names = [readable(n) for n in meta["info"]["features"]["label"]["names"]]
    classes = sorted(names)
    groups = {c: next((g for g, words in NWPU_GROUPS if any(w in c for w in words.split())), "other") for c in classes}
    rows = []
    for split in ("train", "validation"):
        for f in sorted(data.glob(f"{split}-*.parquet")):
            for i, rec in enumerate(pq.read_table(f).to_pylist()):
                label = names[rec["label"]]
                item = rec.get("image_id") or f"{split}{f.stem[-5:]}_{i:05d}"
                rel, size, sha = store(rec["image"]["bytes"], f"nwpu/{split}/{label.replace(' ', '_')}/{item}")
                state = "Google Earth image, about 0.2–30 m per pixel." if rng.random() < STATE_RATE else ""
                rows.append(buildings_option(rng, choice_row(
                    rng, source="NWPU-RESISC45", item=f"{split}:{item}", group_id=f"NWPU-RESISC45:{split}:{item}",
                    task_family="scene_cls", image_rel=rel, size=size, sha=sha, gold=label, classes=classes,
                    groups=groups, templates=SCENE_T, state=state, extra={"licence": "NWPU-RESISC45: research use"})))
    for r in rows:  # choice_row writes family 1's ids and family name
        r["id"] = r["id"].replace("tr:fam1:", "tr:fam19:", 1)
        r["benchmark"] = r["benchmark"].replace("trainset/fam1/", "trainset/fam19/", 1)
        r["meta"]["family"] = "fam19"
    print(f"  buildings option: {dict(collections.Counter(r['meta'].get('buildings') for r in rows))}", flush=True)
    write("fam19_nwpu", rows, rng)


if __name__ == "__main__":
    {"rsvqa": rsvqa, "vrsbench": vrsbench, "cdvqa": cdvqa, "disasterm3": disasterm3, "dvl": dvl, "nwpu": nwpu}[sys.argv[1]]()
