"""Family 4, damage: ~12k training rows from xBD train + tier3 (HF mirror
kshitijrajsharma/xview2-xbd, the same mirror C9 uses for its test split; test and hold are never read).

Buildings are found exactly as C9 finds them (bench/converters/xbd_damage.py): connected components of the pre-event
footprint mask (>= 100 px), with the majority post-event label inside as the level. Crops are 2x the building's box,
at least 96 px, with a red outline, so training and test buildings look identical. Only the wording differs.

Parts:
- building damage (score, 4 levels: no / minor / major damage / destroyed): 8,000 buildings, 2,000 per level.
- area damage share (score, 5 bins of the share of buildings with major damage or destroyed: none, under 10%,
  10-30%, 30-60%, over 60%): 2,000 scenes with at least 10 buildings, balanced over bins; full pre / post scenes.
- disaster type (choice over xBD's own labels: fire, flooding, wind, earthquake, tsunami, volcano): 2,000 scenes,
  balanced by type, with the option rules of trainset/common.py.

Every part is drawn at random with no event above EVENT_CAP of its rows. Drawn at random with no cap, four flood
and hurricane events held half of the rows: the model answered "flood" for 86 of CHOICE's 200 disaster images
(50 are floods) and CHOICE disaster discrimination fell from 91.0 (d3-flash) to 79.5. A strict round robin over the
19 events brought CHOICE back only to 84.0 while xBD disaster type fell from 87.0 to 73.7.

Leaks: any xBD scene named anywhere in the eval rows (e.g. DisasterM3's "lower_puna_volcano_00000121_post_disaster")
is dropped. meta.level and meta.share keep the exact targets for soft ordinal (SORD) training.

    MODULE=trainset.fam4_damage [ARGS="--sample 10"] sbatch -J eojev-fam4 slurm/convert_cpu.sbatch
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import random
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from bench.common import write_jsonl
from bench.converters.xbd_damage import DISASTER_TYPES_XBD, EVENT_TYPE_XBD, LEVELS, buildings, crop
from trainset.common import ROWS, TRAIN, choice_options, make_row, render, store

MAN = json.loads((TRAIN / "downloads.json").read_text())
TARGET = {"damage": 8000, "share": 2000, "type": 2000}
EVENT_CAP = 0.10  # largest share of one xBD event in each part
MIN_BUILDINGS_FOR_SHARE = 10
SHARE_LEVELS = ["none", "under 10%", "10–30%", "30–60%", "over 60%"]
STATES = ["Image 1 was taken before the disaster and Image 2 after it; both show the same place.",
          "Image 1: before the disaster. Image 2: the same area after the disaster.",
          "The first image shows the area before a disaster, the second the same area afterwards."]
DMG_T = [("dmg_v1", "How much damage did the building in the red box suffer?"),
         ("dmg_v2", "Rate the damage to the outlined building."),
         ("dmg_v3", "What is the damage level of the building marked in red?"),
         ("dmg_v4", "Judging by the two images, how badly was the outlined building damaged?"),
         ("dmg_v5", "Assess the damage to the building inside the red outline.")]
SHARE_T = [("share_v1", "What share of the buildings in this area were badly damaged or destroyed?"),
           ("share_v2", "Roughly what fraction of the buildings here suffered major damage or were destroyed?"),
           ("share_v3", "How much of the building stock in this area was heavily damaged?")]
TYPE_T = [("type_v1", "What kind of disaster affected this area?"),
          ("type_v2", "Which type of disaster caused the changes between the two images?"),
          ("type_v3", "What disaster struck this place?"),
          ("type_v4", "Comparing the two images, what type of event caused the damage?")]
SCENE_RE = re.compile(r"([a-z]+(?:[-_][a-z]+)+)_(\d{8})")


def blocked_scenes() -> set[str]:
    out = set()
    with open(TRAIN / "blocklist" / "eval_source_ids.jsonl") as f:
        for line in f:
            for m in SCENE_RE.finditer(line):
                event = m.group(1).replace("_", "-")
                if event in EVENT_TYPE_XBD:
                    out.add(f"{event}_{m.group(2)}")
    return out


def event_balanced(rng, items: list, n: int, scene_of=lambda x: x) -> list:
    """n items drawn at random, no xBD event above EVENT_CAP of n; if the cap leaves too few, the rest at random."""
    pool = items[:]
    rng.shuffle(pool)
    per_event, out, rest = collections.Counter(), [], []
    for it in pool:
        e = scene_of(it).rsplit("_", 1)[0]
        if len(out) < n and per_event[e] < EVENT_CAP * n:
            per_event[e] += 1
            out.append(it)
        else:
            rest.append(it)
    return out + rest[:max(0, n - len(out))]


def share_bin(share: float) -> int:
    if share == 0:
        return 0
    return 1 if share < 0.1 else 2 if share < 0.3 else 3 if share < 0.6 else 4


def png(blob: dict) -> Image.Image:
    return Image.open(io.BytesIO(blob["bytes"]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0, help="only this many rows per part, for review")
    args = p.parse_args()
    target = {k: (args.sample or v) for k, v in TARGET.items()}
    blocked = blocked_scenes()
    print(f"  xBD scenes named in eval rows (blocked): {len(blocked)}", flush=True)
    snap = Path(MAN["xbd"]["path"])
    files = sorted(f for f in (snap / "data").glob("*.parquet") if f.name.startswith(("train-", "tier3-")))

    # pass 1: buildings and scene statistics from the masks
    cand: dict[int, list] = collections.defaultdict(list)
    scenes = {}
    n_blocked = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=["image_name", "t1_mask", "t2_mask"]).to_pylist()
            for i, r in enumerate(t):
                scene = r["image_name"].rsplit("/", 1)[-1]
                if scene in blocked:
                    n_blocked += 1
                    continue
                bs = buildings(np.array(png(r["t1_mask"])), np.array(png(r["t2_mask"])))
                tier = r["image_name"].split("/", 1)[0]
                scenes[scene] = {"file": f, "group": g, "row": i, "tier": tier, "levels": [lv for _, lv in bs]}
                for box, lv in bs:
                    cand[lv].append((scene, box))
    print(f"  scenes {len(scenes)} (blocked {n_blocked}); buildings per level "
          f"{ {LEVELS[k]: len(v) for k, v in sorted(cand.items())} }", flush=True)

    rng = random.Random("fam4")
    picked_b = collections.defaultdict(list)
    for lv in range(4):
        for scene, box in event_balanced(rng, cand[lv], target["damage"] // 4, lambda x: x[0]):
            picked_b[scene].append((box, lv))
    by_bin = collections.defaultdict(list)
    for scene, s in scenes.items():
        if len(s["levels"]) >= MIN_BUILDINGS_FOR_SHARE:
            share = sum(lv >= 2 for lv in s["levels"]) / len(s["levels"])
            by_bin[share_bin(share)].append((scene, share))
    picked_share = {}
    for b in range(5):
        for scene, share in event_balanced(rng, by_bin[b], target["share"] // 5, lambda x: x[0]):
            picked_share[scene] = share
    by_type = collections.defaultdict(list)
    for scene in scenes:
        by_type[EVENT_TYPE_XBD[scene.rsplit("_", 1)[0]]].append(scene)
    picked_type = []
    for ty in DISASTER_TYPES_XBD:
        picked_type += event_balanced(rng, by_type[ty], target["type"] // len(DISASTER_TYPES_XBD))
    print(f"  share bins available { {SHARE_LEVELS[b]: len(v) for b, v in sorted(by_bin.items())} }; "
          f"types available { {k: len(v) for k, v in by_type.items()} }", flush=True)

    # pass 2: images
    need = set(picked_b) | set(picked_share) | set(picked_type)
    by_file = collections.defaultdict(list)
    for scene in need:
        s = scenes[scene]
        by_file[(s["file"], s["group"])].append(scene)
    full: dict[str, tuple] = {}
    rows = []
    for (f, g), scs in sorted(by_file.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        t = pq.ParquetFile(f).read_row_group(g, columns=["image_name", "t1_image", "t2_image"]).to_pylist()
        for scene in scs:
            r = t[scenes[scene]["row"]]
            event = scene.rsplit("_", 1)[0]
            common = {"event": event, "tier": scenes[scene]["tier"], "licence": "xBD: CC BY-NC-SA 4.0"}
            if scene in picked_share or scene in picked_type:
                pre = store(r["t1_image"]["bytes"], f"xbd/scenes/{scene}_pre")
                post = store(r["t2_image"]["bytes"], f"xbd/scenes/{scene}_post")
                full[scene] = (pre, post)
            if scene in picked_b:
                pre_img, post_img = png(r["t1_image"]).convert("RGB"), png(r["t2_image"]).convert("RGB")
                for box, lv in picked_b[scene]:
                    key = f"{scene}_{box[0].start}_{box[1].start}"
                    ims = []
                    for tag, im in (("pre", pre_img), ("post", post_img)):
                        buf = io.BytesIO()
                        crop(im, box).save(buf, format="PNG")
                        ims.append(store(buf.getvalue(), f"xbd/buildings/{key}_{tag}"))
                    tid, text = rng.choice(DMG_T)
                    rows.append(make_row(
                        rid=f"tr:fam4:xBD:{key}", group_id=f"xBD:{scene}", family="fam4", source="xBD",
                        task_family="damage_ordinal", images=[ims[0][0], ims[1][0]], state=rng.choice(STATES),
                        questions={"q": {"type": "score", "instructions": text, "criteria": LEVELS}},
                        answer={"q": lv},
                        meta={"template": tid, "level": lv, "box_yx": [box[0].start, box[0].stop, box[1].start,
                                                                      box[1].stop],
                              "image_sha256": [ims[0][2], ims[1][2]], **common}))
            if scene in picked_share:
                (pre, post), share = full[scene], picked_share[scene]
                tid, text = rng.choice(SHARE_T)
                rows.append(make_row(
                    rid=f"tr:fam4:xBD:{scene}:share", group_id=f"xBD:{scene}", family="fam4", source="xBD",
                    task_family="damage_share", images=[pre[0], post[0]], state=rng.choice(STATES),
                    questions={"q": {"type": "score", "instructions": text, "criteria": SHARE_LEVELS}},
                    answer={"q": share_bin(share)},
                    meta={"template": tid, "share": round(share, 4), "n_buildings": len(scenes[scene]["levels"]),
                          "image_sha256": [pre[2], post[2]], **common}))
            if scene in picked_type:
                pre, post = full[scene]
                gold = EVENT_TYPE_XBD[event]
                tid, text = rng.choice(TYPE_T)
                options, answer, ometa = choice_options(rng, gold, DISASTER_TYPES_XBD)
                rows.append(make_row(
                    rid=f"tr:fam4:xBD:{scene}:type", group_id=f"xBD:{scene}", family="fam4", source="xBD",
                    task_family="disaster_type", images=[pre[0], post[0]], state=rng.choice(STATES),
                    questions={"q": {"type": "choice", "instructions": text, "criteria": {o: None for o in options}}},
                    answer={"q": answer},
                    meta={"template": tid, "gold_label": gold, **ometa, "image_sha256": [pre[2], post[2]], **common}))

    suffix = ".sample" if args.sample else ""
    write_jsonl(ROWS / f"fam4_damage{suffix}.jsonl", rows)
    with open(ROWS / f"fam4_damage{suffix}.txt", "w") as f:
        shown = collections.Counter()
        for r in rows:
            if shown[r["task_family"]] < (args.sample or 8):
                f.write(render(r) + "\n\n")
                shown[r["task_family"]] += 1
    kinds = collections.Counter(r["task_family"] for r in rows)
    lv = collections.Counter(r["answer"]["q"] for r in rows if r["task_family"] == "damage_ordinal")
    print(f"fam4_damage: {len(rows)} rows; {dict(kinds)}; damage levels {dict(sorted(lv.items()))}; "
          f"splits {dict(collections.Counter(r['split'] for r in rows))}")


if __name__ == "__main__":
    main()
