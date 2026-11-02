"""End-to-end check of a native-format adapter on the model's OWN released xBD test items (not our rows): if our
pipeline reproduces its published accuracy there, a low score on our rows is the model, not the harness.

EarthDial: akshaydudhane/EarthDial-Dataset Change_detection/xBD (img_cls_testset_1 = disaster type, published 96.4;
reg_cls_testset_1 = building damage, imbalanced, published 82.85). Questions are used verbatim; the eval appends
"Answer in single word." to the damage ones (src/earthdial/eval/rs_change_detection/rs_change_detection_test.py).

    python -m bench.specialists.check_native earthdial [n per test set]     (GPU; slurm/run_py.sbatch)
"""

from __future__ import annotations

import collections
import io
import os
import sys

import pyarrow as pa
from PIL import Image

from bench.common import DATASETS
from bench.specialists import load
from bench.specialists.prompting import Item, parse_answer

HUB = os.environ.get("EOJEV_DATASET_HUB", str(DATASETS / "hf_hub"))
SETS = {"xBD_img_cls_testset_1": ("", ["volcano", "fire", "earthquake", "flood", "tsunami", "wind"]),
        "xBD_reg_cls_testset_1": ("Answer in single word.", ["no-damage", "destroyed", "minor-damage", "major-damage"])}


def _arrow(name: str) -> pa.Table:
    for root, _, files in os.walk(os.path.join(HUB, "datasets--akshaydudhane--EarthDial-Dataset")):
        if root.endswith(name) and "data-00000-of-00001.arrow" in files:
            with pa.memory_map(os.path.join(root, "data-00000-of-00001.arrow")) as src:
                return pa.ipc.open_stream(src).read_all()
    raise FileNotFoundError(name)


def main() -> None:
    model_name, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60
    assert model_name == "earthdial", "only EarthDial's test files are wired up"
    model = load(model_name)
    for name, (suffix, labels) in SETS.items():
        t = _arrow(name)
        step = max(1, t.num_rows // n)
        rows = [t.slice(i, 1).to_pylist()[0] for i in list(range(0, t.num_rows, step))[:n]]  # take() overflows
        gen = collections.Counter()
        ok_gen = ok_lp = 0
        for r in rows:
            gold = r["groundtruth"].strip('"')
            images = [Image.open(io.BytesIO(r[k]["bytes"])).convert("RGB") for k in ("jpg_A", "jpg_B")]
            item = Item("x", "q", "choice", "names", r["question"] + suffix, labels, images=images, native=True)
            text = model.generate([item])[0]
            i = parse_answer(item, text)
            pred = labels[i] if i is not None else f"?{text[:20]}"
            gen[(gold, pred)] += 1
            ok_gen += pred == gold
            probs = model.score([item])[0]
            ok_lp += labels[max(range(len(labels)), key=probs.__getitem__)] == gold
        print(f"{name}: {len(rows)} items | written answers {ok_gen / len(rows):.3f} | option probabilities "
              f"{ok_lp / len(rows):.3f}", flush=True)
        print("   (gold, written) most common:", gen.most_common(10), flush=True)


if __name__ == "__main__":
    main()
