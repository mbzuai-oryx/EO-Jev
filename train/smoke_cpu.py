"""CPU checks of the training pipeline before it takes a GPU (no model weights are loaded).

1. Every train/ and probe module imports.
2. Targets for every training question:
   - one value per option, summing to 1;
   - the target's top option is the gold one, except replay rows with d3's own probabilities;
   - it reports ordinal / SORD / one-hot counts per family.
3. Mixer: 4,000 draws follow the family weights.
4. Prompts: 40 questions per family are rendered exactly as the trainer does and encoded by the processor, to report
   input tokens per question (this sets the micro-batch size).
5. The probe D1 rows build (bench.probes.d1_rows), into a temporary directory: the stored rows are left alone.

    MODULE=train.smoke_cpu sbatch -J eojev-smoke slurm/convert_cpu.sbatch
"""

from __future__ import annotations

import collections
import importlib
import json
import random
import statistics
from pathlib import Path

MODULES = ("train.d3model", "train.data", "train.train", "train.parity", "train.label_replay", "train.export",
           "bench.probes.extract", "bench.probes.fit", "bench.probes.d1_rows")


def main() -> None:
    for m in MODULES:
        importlib.import_module(m)
    print("imports ok:", ", ".join(MODULES), flush=True)

    from transformers import AutoTokenizer

    from train import data
    from train.d3model import D3_FLASH, gold_index, render_prompt, rt

    for split in ("train", "dev"):
        items = data.load_items(split)
        summary = {}
        for fam, its in items.items():
            kinds, bad = collections.Counter(), []
            for it in its:
                q = it.row["questions"][it.qid]
                n = 2 if q["type"] == "noul" else len(q["criteria"])
                d3 = "d3_probs" in it.row.get("meta", {})
                top = max(range(len(it.target)), key=it.target.__getitem__)
                if len(it.target) != n or abs(sum(it.target) - 1) > 1e-4:
                    bad.append((it.row["id"], "shape"))
                elif not d3 and top != gold_index(q, it.row["answer"][it.qid]):
                    bad.append((it.row["id"], "top != gold"))
                kinds["d3_probs" if d3 else "ordinal" if it.ordinal else "soft" if max(it.target) < 0.999 else "one-hot"] += 1
            summary[fam] = {"questions": len(its), "targets": dict(kinds), "bad": len(bad), "bad_examples": bad[:3]}
        print(split, json.dumps(summary, indent=1), flush=True)

    items = data.load_items("train")
    weights = data.default_weights(items)
    drawn = collections.Counter(it.family for it in data.Mixer(items, weights).sample(4000))
    print("weights", {k: round(v, 3) for k, v in weights.items()}, "drawn", {k: round(v / 4000, 3) for k, v in drawn.items()},
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(D3_FLASH))
    tokenizer.padding_side = "left"
    processor = rt.load_processor(D3_FLASH)
    codes, _ = rt.answer_codes(tokenizer)
    lengths = {}
    for fam, its in items.items():
        sample = random.Random(0).sample(its, min(40, len(its)))
        n = []
        for it in sample:
            ex = data.to_example(it, random.Random(1))
            text, _ = render_prompt(tokenizer, processor, codes, ex)
            if ex.images:
                n.append(int(processor(text=[text], images=ex.images, return_tensors="pt")["input_ids"].shape[1]))
            else:
                n.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
        lengths[fam] = {"median": statistics.median(n), "max": max(n)}
    print("input tokens per question:", json.dumps(lengths), flush=True)

    import tempfile

    from bench.probes import d1_rows

    with tempfile.TemporaryDirectory() as tmp:  # the stored rows must stay those their probe features were made from
        d1_rows.main(Path(tmp))
    print("smoke ok", flush=True)


if __name__ == "__main__":
    main()
