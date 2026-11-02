"""C23 VLRS-Bench (arXiv 2602.07045) closed-form subset -> rows.

Source: HF thislzm/VLRS-Bench @ 1bb156ad80182022c7b29cde170bbd3eb3225d4e (VLRS-Bench.json + images/, CC-BY-4.0).
2,000 items, 500 per format, over 14 reasoning tasks (`job`: CR, CFR, MIR, SIR, ST-CCR, ST-CFR, ST-ER, ST-CR,
PR, ER, ST-CS-PR, ST-M-PR, ST-SU-PR, ST-SQ-PR) and 1-8 images per item.
  single_choice 500 (5 options "A. ..".."E. ..", one correct)  -> `choice`, published order, letters stripped
  true_false    500 (a statement, "True" / "False")             -> `noul` ("Is the following statement true?")
  multi_choice  500 (8 options, 1-8 correct)                     -> one `noul` per option, task_family "<job>_multi",
                                                                    exact-set match downstream; the paper's partial
                                                                    score (1 / 0.5 if a strict error-free subset / 0)
                                                                    is recomputable from the per-option answers
  fill_blank    500                                              -> DROPPED (free-text, scored by MiniLM similarity)
The question text is the published human turn minus the leading "<image>" token and the format line
("(Select one option. Format example: B)", "(Answer format: True or False)", "(Select all that apply)"); the
runner prints the options and reads the answer. Images go in the published order. Multi-image items: when the
item has `times` (one "YYYY_MM" per image, chronological) the state lists them ("Image 1 = 2018-01, ...");
the 101 two-image items without `times` (ST-* tasks, "earlier / later" in their questions) get "Image 1 is the
earlier and Image 2 the later image of the same area." The paper's evaluation prompt is not published, so the
date line is our wording. All 2,429 images used are 76x1024 .. 1024x1024 px, within MIN / MAX_PIXELS: stored as
published (native PNG / JPEG bytes, no resize); the code resizes (Lanczos down / bicubic up) only if one is not.

ADAPTED (not as-published): closed-form subset only (1,500 of 2,000; fill-in dropped), multi-select as per-option
yes/no, true/false as a yes/no question, dates in the state.

    python -m bench.converters.vlrs
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

from PIL import Image

from bench.common import CONVERTED, DATASETS, MAX_PIXELS, MIN_PIXELS, save_png, store_image, validate_row, write_jsonl

REV = "1bb156ad80182022c7b29cde170bbd3eb3225d4e"
SRC = DATASETS / "hf_hub" / "datasets--thislzm--VLRS-Bench" / "snapshots" / REV
OUT = CONVERTED / "vlrs"
FORMAT_LINE = {"single_choice": "(Select one option. Format example: B)", "true_false": "(Answer format: True or False)",
               "multi_choice": "(Select all that apply)"}
N_OPTIONS = {"single_choice": 5, "multi_choice": 8}
OPTION = re.compile(r"^([A-H])\. (.+)$")
LEVEL = {"CR": "Cognition", "CFR": "Cognition", "MIR": "Cognition", "SIR": "Cognition", "ST-CCR": "Cognition",
         "ST-CFR": "Cognition", "ST-ER": "Cognition", "ST-CR": "Cognition", "PR": "Decision", "ER": "Decision",
         "ST-CS-PR": "Prediction", "ST-M-PR": "Prediction", "ST-SU-PR": "Prediction", "ST-SQ-PR": "Prediction"}
SUFFIX = {"single_choice": "single", "true_false": "tf", "multi_choice": "multi"}


def place(rel: str) -> tuple[str, dict]:
    """Copy one image (native bytes when within limits); returns (path under CONVERTED, size / resize info)."""
    src = SRC / rel
    stem = OUT / "images" / rel.removeprefix("images/").rsplit(".", 1)[0]
    with Image.open(src) as im:
        w, h = im.size
        n = w * h
        if MIN_PIXELS <= n <= MAX_PIXELS:
            path, _ = store_image(src.read_bytes(), stem)
            return path, {"orig_px": [w, h], "resize": 1.0}
        if n > MAX_PIXELS:  # Lanczos down to <= MAX_PIXELS
            s = math.sqrt(MAX_PIXELS / n)
            size, method = (max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS
        else:  # bicubic up to >= MIN_PIXELS
            s = math.sqrt(MIN_PIXELS / n)
            size, method = (math.ceil(w * s), math.ceil(h * s)), Image.Resampling.BICUBIC
        out = save_png(im.convert("RGB").resize(size, method), stem.with_suffix(".png"))
        return str(out.relative_to(CONVERTED)), {"orig_px": [w, h], "resize": round(size[0] / w, 4)}


def date(t: str) -> str:
    return t.replace("_", "-")


def state(images: list[str], times: list[str]) -> str:
    n = len(images)
    if n == 1:
        return ""
    if times:
        if len(times) != n or times != sorted(times):
            raise ValueError(f"times {times} for {n} images")
        return ("The images show the same area over time, in chronological order: "
                + ", ".join(f"Image {i + 1} = {date(t)}" for i, t in enumerate(times)) + ".")
    if n == 2:
        return "Image 1 is the earlier and Image 2 the later image of the same area."
    raise ValueError(f"{n} images without times")


def main() -> None:
    items = json.loads((SRC / "VLRS-Bench.json").read_text())
    rows, dropped, placed = [], Counter(), {}
    for x in items:
        qa = x["qa_type"]
        if qa == "fill_blank":
            dropped["fill_blank: free-text answer scored by sentence-embedding similarity"] += 1
            continue
        human, gpt = x["conversations"][0]["value"], x["conversations"][1]["value"].strip()
        if not human.startswith("<image>\n") or human.count("<image>") != 1:
            raise ValueError(f"{x['id']}: unexpected image token layout")
        lines = human.removeprefix("<image>\n").split("\n")
        if lines.count(FORMAT_LINE[qa]) != 1:
            raise ValueError(f"{x['id']}: format line missing")
        cut = lines.index(FORMAT_LINE[qa])
        stem = "\n".join(lines[:cut]).strip()
        option_lines = lines[cut + 1:]
        meta = {}
        if qa == "true_false":
            if option_lines or gpt not in ("True", "False"):
                raise ValueError(f"{x['id']}: bad true/false item")
            questions = {"q": {"type": "noul", "instructions": f"Is the following statement true?\n{stem}"}}
            answer = {"q": gpt == "True"}
            meta["gold"] = gpt
        else:
            parsed = [OPTION.match(line) for line in option_lines]
            letters = [m.group(1) for m in parsed] if all(parsed) else []
            if letters != list("ABCDEFGH"[: N_OPTIONS[qa]]):
                raise ValueError(f"{x['id']}: options {option_lines}")
            options = [m.group(2).strip() for m in parsed]
            if len(set(options)) != len(options):
                raise ValueError(f"{x['id']}: duplicate options")
            gold_letters = [s.strip() for s in gpt.split(",")]
            if not gold_letters or not set(gold_letters) <= set(letters):
                raise ValueError(f"{x['id']}: answer {gpt!r}")
            golds = {options[letters.index(s)] for s in gold_letters}
            if qa == "single_choice":
                if len(golds) != 1:
                    raise ValueError(f"{x['id']}: {len(golds)} answers")
                questions = {"q": {"type": "choice", "instructions": stem, "criteria": {o: None for o in options}}}
                answer = {"q": golds.pop()}
            else:
                questions = {f"opt{i}": {"type": "noul", "instructions": f"{stem}\nIs \"{o}\" one of the correct answers?"}
                             for i, o in enumerate(options)}
                answer = {f"opt{i}": o in golds for i, o in enumerate(options)}
                meta["paper_metric"] = "1.0 exact set; 0.5 if a strict subset with no wrong option; else 0"
            meta.update(options=options, gold_letters=gold_letters)
        images, sizes = [], []
        for rel in x["image"]:
            if rel not in placed:
                placed[rel] = place(rel)
            images.append(placed[rel][0])
            sizes.append(placed[rel][1])
        row = {
            "id": f"VLRS:{x['id']}",
            "group_id": f"VLRS:{x['image'][0].rsplit('/', 1)[-1].rsplit('.', 1)[0]}",
            "benchmark": "VLRS-Bench",
            "split": "test",
            "task_family": f"{x['job'].lower().replace('-', '_')}_{SUFFIX[qa]}",
            "skills": ["reasoning", LEVEL[x["job"]].lower()] + (["temporal"] if len(images) > 1 else []),
            "images": images,
            "state": state(images, x["times"]),
            "questions": questions,
            "answer": answer,
            "meta": {"source_id": x["id"], "job": x["job"], "level": LEVEL[x["job"]], "qa_type": qa,
                     "times": x["times"], "n_images": len(images), "source_images": x["image"],
                     "orig_px": [s["orig_px"] for s in sizes], "resize": [s["resize"] for s in sizes],
                     "multi_select": qa == "multi_choice", **meta},
        }
        validate_row(row)
        rows.append(row)
    n = write_jsonl(OUT / "vlrs_closed.jsonl", rows)
    print("VLRS-Bench:", n, "rows ->", OUT / "vlrs_closed.jsonl")
    print("  questions by type:", dict(Counter(q["type"] for r in rows for q in r["questions"].values())))
    print("  by format:", dict(Counter(r["meta"]["qa_type"] for r in rows)), "| by level:", dict(Counter(r["meta"]["level"] for r in rows)))
    print("  task_family:", json.dumps(dict(sorted(Counter(r["task_family"] for r in rows).items()))))
    print("  images:", len(placed), "unique; resized:", sum(1 for v in placed.values() if v[1]["resize"] != 1.0),
          "| images per row:", dict(sorted(Counter(len(r["images"]) for r in rows).items())))
    print("  dropped:", dict(dropped))
    print("  single-choice gold:", dict(sorted(Counter(r["meta"]["gold_letters"][0] for r in rows if r["meta"]["qa_type"] == "single_choice").items())))
    print("  true/false gold:", dict(Counter(r["meta"]["gold"] for r in rows if r["meta"]["qa_type"] == "true_false")))
    print("  multi-select #correct:", dict(sorted(Counter(len(r["meta"]["gold_letters"]) for r in rows if r["meta"]["qa_type"] == "multi_choice").items())))


if __name__ == "__main__":
    main()
