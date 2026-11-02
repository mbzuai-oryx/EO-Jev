"""C12 DVL-Bench (DynamicVL, NeurIPS'25 D&B) closed-form QA -> rows.

The test split's two closed-form files: Basic Change (2,355: which land-cover transitions changed most) and
Change Speed (1,499: in which periods did change happen / stop); 5-9 options. Single-choice items (2,956) are
one `choice`; multi-select items (899) become one `noul` per option (task_family suffix "_multi"), scored by
exact-set match and per-option accuracy. Every question sees the
scene's 5-10 NAIP images (1 m, 1024 px, 2006-2021) as separate images in time order; their dates go in the
state, since the questions name years. NAIP TIFFs are RGB + NIR (read by PIL as RGBA): the NIR band is dropped.

    python -m bench.converters.dvl
"""

from __future__ import annotations

import io
import json
import zipfile
from collections import Counter

from PIL import Image

from bench.common import CONVERTED, RAW, save_png, validate_row, write_jsonl

SRC = RAW / "dvl" / "test.zip"
OUT = CONVERTED / "dvl"
FILES = {"basic_change_choice_qa.json": "dvl_basic_change", "change_speed_choice_qa.json": "dvl_change_speed"}


def date(stamp: str) -> str:
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}"


def main() -> None:
    rows, need = [], set()
    with zipfile.ZipFile(SRC) as zf:
        for fname, family in FILES.items():
            for k, item in enumerate(json.loads(zf.read(f"test/{fname}"))):
                options = [str(o) for o in item["options_list"]]  # some options are numbers (e.g. 89.28)
                letters = [s.strip() for s in item["ground_truth_option"].strip().rstrip(".").split(",")]
                golds = {options[ord(l) - ord("A")] for l in letters}
                if len(set(options)) != len(options):
                    raise ValueError(f"duplicate options in {fname} #{k}")
                multi = item["task_type"] == "Multiple choice"
                if multi:  # multi-select: one yes/no per option; exact-set match is scored downstream
                    questions = {f"opt{i}": {"type": "noul", "instructions":
                                             f"{item['prompts']} Is \"{o}\" part of the correct answer?"}
                                 for i, o in enumerate(options)}
                    answer = {f"opt{i}": o in golds for i, o in enumerate(options)}
                else:
                    if len(golds) != 1:
                        raise ValueError(f"single-choice item with {len(golds)} answers in {fname} #{k}")
                    questions = {"q": {"type": "choice", "instructions": item["prompts"],
                                       "criteria": {o: None for o in options}}}
                    answer = {"q": golds.pop()}
                images = [f"dvl/images/{p.removeprefix('images/').rsplit('.', 1)[0]}.png" for p in item["image_list"]]
                need.update(item["image_list"])
                timeline = ", ".join(f"Image {i + 1} = {date(t)}" for i, t in enumerate(item["time_stamps"]))
                scene = item["image_list"][0].split("/")[1] + "/" + item["image_list"][0].split("/")[2]
                row = {
                    "id": f"DVL:{family}:{k}",
                    "group_id": f"DVL:{scene}",
                    "benchmark": "DVL-Bench",
                    "split": "test",
                    "task_family": family + ("_multi" if multi else ""),
                    "skills": ["temporal", "change"],
                    "images": images,
                    "state": f"The images show the same area over time, in chronological order: {timeline}.",
                    "questions": questions,
                    "answer": answer,
                    "meta": {"task": item["task"], "n_images": len(images), "n_options": len(options),
                             "multi_select": multi, "options": options,
                             "cls_description": item.get("cls_description"), "sensor": "NAIP RGB", "gsd_m": 1.0},
                }
                validate_row(row)
                rows.append(row)
        for path in sorted(need):
            dest = OUT / "images" / (path.removeprefix("images/").rsplit(".", 1)[0] + ".png")
            if not dest.exists():
                save_png(Image.open(io.BytesIO(zf.read(f"test/{path}"))), dest)
    n = write_jsonl(OUT / "dvl_choice.jsonl", rows)
    print("DVL-Bench:", n, "rows,", len(need), "images", Counter(r["task_family"] for r in rows),
          "images per question", Counter(r["meta"]["n_images"] for r in rows))


if __name__ == "__main__":
    main()
