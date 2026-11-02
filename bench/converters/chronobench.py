"""C21 ChronoBench (GeoChrono, arXiv 2607.15768; ACM MM'26 per its README) -> rows.

Source: HF Davidup1/GeoChrono-Data @ 58bbfebd38d5e4843418b2bc3083d3b365817371, ChronoBench/*.json (19 task
files, 17,689 QAs, Apache-2.0). Images: the DVL-Suite test split (raw/dvl/test.zip, HF weihao1115/dvl_suite @
38066bd0), 3,463 NAIP frames of 1024 x 1024 px; ChronoBench's `images/...` is `test/images/...` in the zip. The
frames are the same PNGs as converted/dvl/images (RGB of the RGB+NIR TIFF, NIR dropped exactly as the reference
harness's `convert("RGB")`); they are hard-linked from there (rendered from the zip if missing). Frames stay at
1024 x 1024 (1,048,576 px <= MAX_PIXELS), the reference harness's fixed input size.

Rows (one per QA; 12 sub-tasks, see the README task table):
  - Single Choice (OP, BCAC, OAM, OCM, OCO, CSOCO, RDC, CCME; 2-9 options) -> one `choice`, published option order.
  - Multiple Choice (BCC, LCC, BOC; 4 options, 0-4 correct) -> one `noul` per option, task_family "<abbr>_multi",
    scored by exact-set match (= the paper's multiple-choice "exact_match"). One BOC-HBB item (#600) has an empty
    gold set (gold "vegetation to vegetation", option "."): kept, all four answers False.
  - Object History Memory (OHM, 880 = 440 coordinate + 440 HBB): the answer is an ordered letter sequence with
    repeats (e.g. "E, C, E, C, D."), which no choice / noul / score question can express -> DROPPED.
  - Cross-sequence CSOCO (410): Region#1's frames, then Region#2's, in the published order; the state says which
    images belong to which region and gives each frame's year.
Prompts are the published `prompts` / `prompt` text (HBB boxes stay as text, normalised 0-999; geographic
coordinates stay as text; nothing is drawn). The reference harness interleaves "YYYY: <image>" and puts a
per-task system prompt first; here the year of every frame goes in the state ("Image k = YYYY"), followed by that
task's system-prompt context (coordinate convention, the five land-cover classes, the task definition) from
GeoChrono's lmms-eval task pack (third_party/GeoChrono @ ba537705, chronobench_system_prompts.json); its role
line, input-format paragraph, "This is a ... question" paragraph and answer-format ("IMPORTANT: Respond ...")
paragraph are left out because the runner asks the question and reads the answer.

ADAPTED (not as-published): OHM dropped (880 of 17,689; the paper's L2 / LTM average includes it); years in the
state instead of interleaved; multi-select as per-option yes/no (exact-set scoring keeps the paper's metric).
Outputs: chronobench.jsonl (all 16,809 kept QAs) and chronobench_5k.jsonl (fixed 5,000-row subset, stratified by
task file with largest-remainder proportional quotas, random.Random(0)) for checkpoints.

    python -m bench.converters.chronobench
"""

from __future__ import annotations

import io
import json
import os
import random
import zipfile
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from bench.common import CONVERTED, DATASETS, MAX_PIXELS, RAW, THIRD_PARTY, save_png, validate_row, write_jsonl

REV = "58bbfebd38d5e4843418b2bc3083d3b365817371"
SRC = DATASETS / "hf_hub" / "datasets--Davidup1--GeoChrono-Data" / "snapshots" / REV / "ChronoBench"
ZIP = RAW / "dvl" / "test.zip"
DVL_PNG = CONVERTED / "dvl" / "images"
PROMPTS = THIRD_PARTY / "GeoChrono" / "lmms_eval_tasks" / "chronobench" / "chronobench_system_prompts.json"
OUT = CONVERTED / "chronobench"
N_SUBSET = 5000

# file stem -> (sub-task abbreviation, level, localisation) from the README task table.
TASKS = {
    "Object_Classification_Coordinate": ("OP", "L0", "coord"),
    "Object_Classification_HBB_Box": ("OP", "L0", "hbb"),
    "Bi_Temporal_Class_Level_Change_Perception": ("BCC", "L1", None),
    "Bi_Temporal_Class_Level_Change_Perception_Single": ("BCAC", "L1", None),
    "Long_Temporal_Class_Level_Change_Perception": ("LCC", "L1", None),
    "Temporal_Object_Level_Change_Perception_Geo_Coordinate": ("BOC", "L1", "coord"),
    "Temporal_Object_Level_Change_Perception_HBB_Box": ("BOC", "L1", "hbb"),
    "Object_Appear_Memory_Coordinate": ("OAM", "L2", "coord"),
    "Object_Appear_Memory_HBB_Box": ("OAM", "L2", "hbb"),
    "Object_Change_Memory_Coordinate": ("OCM", "L2", "coord"),
    "Object_Change_Memory_HBB_Box": ("OCM", "L2", "hbb"),
    "Object_History_Memory_Coordinate": ("OHM", "L2", "coord"),
    "Object_History_Memory_HBB_Box": ("OHM", "L2", "hbb"),
    "Object_Development_Comparison_Coordinate": ("OCO", "L3", "coord"),
    "Object_Development_Comparison_HBB_Box": ("OCO", "L3", "hbb"),
    "Cross_View_Object_Development_Comparison_Coordinate": ("CSOCO", "L3", "coord"),
    "Cross_View_Object_Development_Comparison_HBB_Box": ("CSOCO", "L3", "hbb"),
    "Region_Comparison_HBB_Box": ("RDC", "L3", None),
    "Change_Speed_Substantial": ("CCME", "L3", None),
}
DROPPED = {"OHM": "ordered-sequence answer (letters in chronological order, with repeats): not a choice / noul / score"}
LEVEL_NAME = {"L0": "Land Cover Perception", "L1": "Temporal Recognition", "L2": "Long-Term Memory",
              "L3": "Spatio-Temporal Reasoning"}
# System-prompt paragraphs that only describe the interleaved input or the answer format.
SKIP_PARAGRAPH = ("You are an expert", "You will receive", "Region#2:", "This is a", "IMPORTANT:")


def task_context(system_prompt: str) -> str:
    keep = [p.strip() for p in system_prompt.split("\n\n") if p.strip() and not p.strip().startswith(SKIP_PARAGRAPH)]
    return "\n\n".join(keep)


def timeline(stamps: list[str], first: int) -> str:
    return ", ".join(f"Image {first + i} = {t[:4]}" for i, t in enumerate(stamps))


def frames_state(x: dict) -> str:
    if "image_list_1" in x:
        n1, n2 = len(x["image_list_1"]), len(x["image_list_2"])
        return (f"Images 1-{n1} show Region#1 and images {n1 + 1}-{n1 + n2} show Region#2, two different areas; "
                f"each region's images are a time series of that area from earliest to latest. "
                f"Region#1: {timeline(x['time_stamps_1'], 1)}. Region#2: {timeline(x['time_stamps_2'], n1 + 1)}.")
    n = len(x["image_list"])
    return f"The {n} images show the same area over time, from earliest to latest: {timeline(x['time_stamps'], 1)}."


def scene(path: str) -> str:
    return "/".join(path.split("/")[1:3])


def png_rel(path: str) -> str:
    return f"chronobench/images/{path.removeprefix('images/').rsplit('.', 1)[0]}.png"


def place_images(paths: set[str]) -> Counter:
    """Hard-link the DVL PNG of every frame (same pixels); render from the zip when it is missing."""
    how = Counter()
    with zipfile.ZipFile(ZIP) as zf:
        for k, path in enumerate(sorted(paths)):
            dest = CONVERTED / png_rel(path)
            src = DVL_PNG / (path.removeprefix("images/").rsplit(".", 1)[0] + ".png")
            if k < 3:  # spot check: the DVL PNG is exactly the RGB of the source TIFF
                tif = np.asarray(Image.open(io.BytesIO(zf.read(f"test/{path}"))).convert("RGB"))
                if not np.array_equal(tif, np.asarray(Image.open(src).convert("RGB"))):
                    raise ValueError(f"{src} differs from test/{path}")
            if dest.exists():
                how["present"] += 1
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(src, dest)
                    how["hardlinked"] += 1
                except OSError:
                    save_png(Image.open(io.BytesIO(zf.read(f"test/{path}"))), dest)
                    how["rendered"] += 1
            with Image.open(dest) as im:
                if im.size != (1024, 1024) or im.width * im.height > MAX_PIXELS:
                    raise ValueError(f"{dest}: unexpected size {im.size}")
    return how


def subset(rows: list[dict], n: int) -> list[dict]:
    """Proportional quotas per task file (largest remainder), seeded sample within each, source order kept."""
    strata = defaultdict(list)
    for r in rows:
        strata[r["meta"]["file"]].append(r)
    exact = {k: n * len(v) / len(rows) for k, v in strata.items()}
    quota = {k: int(e) for k, e in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k] - quota[k]), k))[: n - sum(quota.values())]:
        quota[k] += 1
    rng = random.Random(0)
    keep = set()
    for k in sorted(strata):
        keep.update(r["id"] for r in rng.sample(strata[k], quota[k]))
    return [r for r in rows if r["id"] in keep]


def main() -> None:
    system_prompts = json.loads(PROMPTS.read_text())
    rows, need, dropped = [], set(), Counter()
    for stem, (abbr, level, loc) in TASKS.items():
        items = json.loads((SRC / f"{stem}.json").read_text())
        if abbr in DROPPED:
            dropped[f"{stem} ({abbr}): {DROPPED[abbr]}"] += len(items)
            continue
        context = task_context(system_prompts[f"{stem}.json"])
        for k, x in enumerate(items):
            prompt = x.get("prompts") or x.get("prompt")  # four files spell the key `prompt`
            options = [str(o) for o in x["options_list"]]
            if len(set(options)) != len(options):
                raise ValueError(f"{stem} #{k}: duplicate options")
            letters = [s.strip() for s in x["ground_truth_option"].strip().rstrip(".").split(",") if s.strip()]
            golds = {options[ord(s) - ord("A")] for s in letters}
            multi = x["task_type"] == "Multiple Choice"
            if multi:
                questions = {f"opt{i}": {"type": "noul", "instructions": f"{prompt} Is \"{o}\" one of the correct answers?"}
                             for i, o in enumerate(options)}
                answer = {f"opt{i}": o in golds for i, o in enumerate(options)}
            else:
                if len(golds) != 1:
                    raise ValueError(f"{stem} #{k}: single choice with {len(golds)} answers")
                questions = {"q": {"type": "choice", "instructions": prompt, "criteria": {o: None for o in options}}}
                answer = {"q": golds.pop()}
            cross = "image_list_1" in x
            paths = x["image_list_1"] + x["image_list_2"] if cross else x["image_list"]
            stamps = x["time_stamps_1"] + x["time_stamps_2"] if cross else x["time_stamps"]
            need.update(paths)
            group = (f"ChronoBench:{scene(x['image_list_1'][0])}+{scene(x['image_list_2'][0])}" if cross
                     else f"ChronoBench:{scene(paths[0])}")
            family = abbr.lower() + (f"_{loc}" if loc else "") + ("_multi" if multi else "")
            source_loc = {key: x[key] for key in ("point", "point_1", "point_2", "HBB_Box", "HBB_Box_1", "HBB_Box_2",
                                                  "Region_HBB_Box_1", "Region_HBB_Box_2", "time_stamp_1", "time_stamp_2")
                          if key in x}
            row = {
                "id": f"ChronoBench:{stem}:{k}",
                "group_id": group,
                "benchmark": "ChronoBench",
                "split": "test",
                "task_family": family,
                "skills": ["temporal", "change"] + (["cross_sequence"] if cross else []),
                "images": [png_rel(p) for p in paths],
                "state": frames_state(x) + "\n\n" + context,
                "questions": questions,
                "answer": answer,
                "meta": {"file": stem, "index": k, "task": abbr, "level": level, "level_name": LEVEL_NAME[level],
                         "localisation": loc, "task_type": x["task_type"], "multi_select": multi,
                         "options": options, "gold_letters": letters, "empty_gold_set": multi and not letters,
                         "n_images": len(paths), "dates": stamps, "cross_region": cross,
                         "n_region1": len(x["image_list_1"]) if cross else None,
                         "source_paths": paths, "source_loc": source_loc, "orig_px": [1024, 1024], "resize": 1.0,
                         "sensor": "NAIP RGB", "gsd_m": 1.0},
            }
            validate_row(row)
            rows.append(row)
    how = place_images(need)
    n_all = write_jsonl(OUT / "chronobench.jsonl", rows)
    sub = subset(rows, N_SUBSET)
    n_sub = write_jsonl(OUT / "chronobench_5k.jsonl", sub)

    def summary(rs: list[dict]) -> dict:
        return {"rows": len(rs), "questions": dict(Counter(q["type"] for r in rs for q in r["questions"].values())),
                "task_family": dict(sorted(Counter(r["task_family"] for r in rs).items()))}

    print("ChronoBench:", n_all, "rows ->", OUT / "chronobench.jsonl")
    print("  all:", json.dumps(summary(rows)))
    print("  5k: ", n_sub, "rows ->", OUT / "chronobench_5k.jsonl", json.dumps(summary(sub)))
    print("  images:", len(need), dict(how), "| resized: 0 (all 1024x1024)")
    print("  frames per row:", dict(sorted(Counter(r["meta"]["n_images"] for r in rows).items())))
    print("  dropped:", dict(dropped))
    golds = Counter(r["meta"]["gold_letters"][0] for r in rows if not r["meta"]["multi_select"])
    print("  single-choice gold letter distribution:", dict(sorted(golds.items())))
    print("  multi-select #correct:", dict(Counter(len(r["meta"]["gold_letters"]) for r in rows if r["meta"]["multi_select"])))


if __name__ == "__main__":
    main()
