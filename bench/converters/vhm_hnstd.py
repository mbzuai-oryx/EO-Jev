"""C14 VHM honesty / hallucination probes (HnstD, AAAI'25; from FitzPC/VHM_eval_dataset) -> rows.

  position (absolute 112, relative 242 true items; 3 x 100 deceptive each: adversarial / popular / random)
      native 5-option MCQ whose options include "Sorry, the object is invisible" -> `choice`
  presence (348)                 yes/no -> `noul`
  color (200 true; 3 x 100 deceptive "object absent"; 100 panchromatic "colour undeterminable")
      open answers -> `choice` over the observed colours plus two abstain options
  image type (20)                native MCQ -> `choice`
  resolution / GSD (360)         metres per pixel -> `choice` over bins
Deceptive items test whether the model abstains instead of hallucinating (the failure RSAJEV reports for
letter-readout classifiers on unseen content).

    python -m bench.converters.vhm_hnstd
"""

from __future__ import annotations

import json
import random
import re
import zipfile
from collections import Counter

from bench.common import CONVERTED, RAW, validate_row, write_jsonl

SRC = RAW / "vhm-eval"
OUT = CONVERTED / "vhm"
OPT = re.compile(r"^([A-Z])\.\s*(.+)$")
ABSENT = "the object is not in the image"
PANCHROMATIC = "the color cannot be determined (panchromatic image)"
GSD_BINS = [(0.3, "finer than 0.3 m"), (0.6, "0.3 to 0.6 m"), (1.0, "0.6 to 1 m"), (3.0, "1 to 3 m"),
            (10.0, "3 to 10 m"), (float("inf"), "coarser than 10 m")]


def norm_color(a: str) -> str:
    """'grey' -> 'gray'; 'red,green' / 'green,red' -> 'green and red'."""
    parts = sorted({"gray" if p.strip() == "grey" else p.strip() for p in a.lower().split(",") if p.strip()})
    return " and ".join(parts)


def parse_mc(text: str) -> tuple[str, list[tuple[str, str]]]:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    question = lines[0]
    opts = [OPT.match(l) for l in lines[1:]]
    return question, [(m.group(1), m.group(2).strip()) for m in opts if m]


def family_of(subset: str) -> str:
    kind = subset.split("_")[0]
    if "_false_pan" in subset:
        return "color_panchromatic"
    if "_false_" in subset:
        mode = next(m for m in ("adversarial", "popular", "popula", "random") if m in subset)
        return f"{kind}_deceptive_{'popular' if mode == 'popula' else mode}"
    return {"abspos": "abspos_true", "relpos": "relpos_true", "color": "color_true", "presence": "presence",
            "imgType": "image_type", "gsd": "gsd"}[kind]


def main() -> None:
    subsets = {
        "abspos_dota-test_mc": "mc", "abspos_false_dota-test_adversarial_mc": "mc", "abspos_false_dota-test_popular_mc": "mc",
        "abspos_false_dota-test_random_mc": "mc", "relpos_dota-test_mc": "mc", "relpos_false_dota-test_adversarial_mc": "mc",
        "relpos_false_dota-test_popula_mc": "mc", "relpos_false_dota-test_random_mc": "mc", "imgType_mcq": "mc",
        "presence_mo_dota": "yesno", "color_dota-test_fair1m-val_open": "color", "color_false_dota-test_adversarial_open": "color",
        "color_false_dota-test_popular_open": "color", "color_false_dota-test_random_open": "color",
        "color_false_pan_dota-test": "color", "gsd_dota_fbp": "gsd",
    }
    data = {s: json.load(open(SRC / "json" / f"{s}.json")) for s in subsets}
    colors = sorted({norm_color(x["conversations"][1]["value"]) for x in data["color_dota-test_fair1m-val_open"]})
    rows, need = [], []
    for subset, kind in subsets.items():
        family = family_of(subset)
        for i, x in enumerate(data[subset]):
            prompt, answer = x["conversations"][0]["value"], x["conversations"][1]["value"]
            image_rel = f"vhm/images/{subset}/{x['image']}"
            need.append((f"datasets_eval/{x['image_path']}/{x['image']}", image_rel))
            if kind == "mc":
                question, opts = parse_mc(prompt)
                texts = [t for _, t in opts]
                gold = dict(opts)[answer.strip().rstrip(".")]
                q = {"type": "choice", "instructions": question, "criteria": {t: None for t in texts}}
            elif kind == "yesno":
                q = {"type": "noul", "instructions": prompt.replace(" Just answer yes or no.", "")}
                gold = answer.strip().lower() == "yes"
            elif kind == "color":
                question = prompt.split(" If this question is unanswerable")[0]
                a = answer.strip().lower()
                gold = (PANCHROMATIC if "panchromatic" in a
                        else ABSENT if ("invisible" in a or "not" in a and "image" in a) else norm_color(a))
                if gold not in (PANCHROMATIC, ABSENT) and gold not in colors:
                    continue
                options = colors + [ABSENT, PANCHROMATIC]
                random.Random(f"vhm:{subset}:{i}").shuffle(options)
                q = {"type": "choice", "instructions": question, "criteria": {o: None for o in options}}
            else:  # gsd
                value = float(answer)
                gold = next(label for upper, label in GSD_BINS if value < upper)
                q = {"type": "choice", "instructions": prompt, "criteria": {label: None for _, label in GSD_BINS}}
            row = {"id": f"VHM-HnstD:{subset}:{i}", "group_id": f"VHM-HnstD:{subset}:{x['image']}",
                   "benchmark": "VHM-HnstD", "split": "eval", "task_family": family, "skills": [family.split("_")[0]],
                   "images": [image_rel], "state": "", "questions": {"q": q}, "answer": {"q": gold},
                   "meta": {"subset": subset, "raw_answer": answer}}
            validate_row(row)
            rows.append(row)
    with zipfile.ZipFile(SRC / "datasets_eval.zip") as zf:
        for member, rel in need:
            dest = CONVERTED / rel
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(member))
    n = write_jsonl(OUT / "vhm_hnstd.jsonl", rows)
    print("VHM-HnstD:", n, "rows;", json.dumps(Counter(r["task_family"] for r in rows)))
    print("color vocabulary:", colors)


if __name__ == "__main__":
    main()
