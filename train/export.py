"""Merge a trained adapter into d3-flash's weights and write a package that d3's runtime loads like d3-flash itself.

Every file of the d3-flash snapshot is copied unchanged (config, readout, decision_config, tokenizer, processor,
runtime), except the weight shards and MODEL_MANIFEST.json. The shards are rewritten from the merged backbone in
BF16. An optional --scale (the plan's alpha_merge, 0..1) shrinks the adapter before merging. The package is
evaluated with bench/runners/d3.py --model <package dir>.

**Export check:** before merging, the adapter model scores every dev question of the training rows, batched as the runtime batches
them. After writing, the runtime loads the package and scores the same questions.
- They must agree on the top answer for at least 99% of questions, with median max|dp| <= 2e-3. Merging into BF16
  rounds the weights, which can move near-ties (99.7%, 0.0015 in a test).
- With adapters on the vision blocks: 98% and 5e-3. Rounding the merged ViT layers moves more answers. Test: a
  small test adapter's vision part scaled x300 (about a trained adapter's size) exported at 98.6% and 0.0035, while
  that vision part itself moved answers by 0.034 at 90.2%.
- Otherwise the job exits 1, so the benchmark job queued after it does not start.

    MODULE=train.export ARGS="--adapter $EOJEV_RUNS/train/eo-jev/adapter_step20000.safetensors --out $EOJEV_RUNS/packages/eo-jev" \
        sbatch slurm/run_py.sbatch
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import torch

from train.d3model import VISION_PREFIX, D3Trainable, Example, answer_vector, load_images
from train.parity import compare, requests

SKIP = ("model.safetensors.index.json", "MODEL_MANIFEST.json")
MIN_AGREE, MAX_MEDIAN_DP = 0.99, 2e-3  # BF16 rounding of the merge alone gave 0.0015 at 99.7%
VISION_MIN_AGREE, VISION_MAX_MEDIAN_DP = 0.98, 5e-3  # with vision-block adapters (test adapter x300: 98.6%, 0.0035)


@torch.no_grad()
def dev_probabilities(model: D3Trainable, reqs) -> dict[str, list[float]]:
    """Probabilities for every dev question of the training rows, batched as the runtime batches them (8 questions per pass, in
    request order)."""
    out = {}
    for _, row, paths in reqs:
        images = load_images(paths)
        qids = list(row["questions"])
        for s in range(0, len(qids), 8):
            exs = [Example(key=q, state=row["state"], question=row["questions"][q], images=images) for q in qids[s: s + 8]]
            for ex, probs in zip(exs, model.probabilities(exs)):
                out[f"{row['id']}/{ex.key}"] = probs
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scale", type=float, default=1.0, help="alpha_merge: fraction of the adapter to merge")
    p.add_argument("--no-check", dest="check", action="store_false")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = D3Trainable()
    info = model.load_lora(args.adapter)
    vision = any(name.startswith(VISION_PREFIX + "blocks.") for name in model.lora)
    min_agree, max_median_dp = (VISION_MIN_AGREE, VISION_MAX_MEDIAN_DP) if vision else (MIN_AGREE, MAX_MEDIAN_DP)
    with torch.no_grad():
        for m in model.lora.values():
            m.lora_b.mul_(args.scale)
    model.eval()
    reqs = [r for r in requests() if r[0] != "text"] if args.check else []  # dev rows of the training families only
    before = dev_probabilities(model, reqs)
    model.merge_lora()
    model.backbone.save_pretrained(str(out), safe_serialization=True, max_shard_size="5GB")
    for f in model.root.iterdir():
        if not f.is_file() or f.name in SKIP or (f.name.startswith("model") and f.name.endswith(".safetensors")):
            continue
        shutil.copy2(f.resolve(), out / f.name)  # snapshot files are symlinks into the blob store
    root = model.root
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"package written to {out}", flush=True)

    record = {"adapter": str(Path(args.adapter).resolve()), "scale": args.scale, "adapter_meta": info, "base": str(root)}
    ok = True
    if args.check:
        from bench.runners.d3 import load_runtime

        runtime = load_runtime(str(out), batch_size=8, permutation_average=False)
        after = {}
        for _, row, paths in reqs:
            res = runtime.system_one(state=row["state"], questions=row["questions"], images=load_images(paths) or None)
            for qid, q in row["questions"].items():
                after[f"{row['id']}/{qid}"] = answer_vector(q, res["answers"][qid])
        check = compare([(before[k], after[k]) for k in before if k in after])
        ok = check["top1_agreement"] >= min_agree and check["median_max_abs_dp"] <= max_median_dp
        record["check"] = {**check, "pass": ok, "criteria": f"top-1 >= {min_agree}, median max|dp| <= {max_median_dp}"}
        print("export check:", json.dumps(record["check"]), flush=True)
    (out / "EOJEV_EXPORT.json").write_text(json.dumps(record, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
