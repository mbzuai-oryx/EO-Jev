"""Gate R0: the trainer's forward pass reproduces the shipped runtime.

The trainer runs as it will in training: train/d3model.py with LoRA attached at zero, activation checkpointing on,
train mode.

**Requests:**
- **Text:** the 6,251 phase-0 Decision Index requests (raw/decision-index-work/parity-rows.jsonl), on which our
  runtime matched the official d3-flash run (docs/results/phase0.md).
- **Images:** every dev row of the training families (trainset rows/final, split == dev). These are single- and two-image rows
  with choice and score questions of 2 to 102 options, i.e. the training distribution itself.

**Steps:**
1. **Runtime:** bench.runners.d3.load_runtime, batch size 8, one request per call. It keeps every question's
   probabilities in option order, then is freed.
2. **Trainer, scored twice:**
   - **Same batching (the R0 gate):** each request's questions in groups of 8 in request order, as the runtime runs
     them. The padded inputs are then identical, so any difference is in the computation itself.
     - Pass: at least 99.9% same top answer and median max|dp| <= 1e-3, overall and for each part.
   - **Training batching (reported only):** micro-batches of 4 that mix requests. Left padding to another
     question's length changes the Gated DeltaNet chunk boundaries and the BF16 summation order, which moves
     near-tie answers. The first run (2026-10-23) measured this:
     - same-length questions matched to 4e-7;
     - short questions padded next to long ones differed by about 1e-3;
     - overall 99.82% same top answer.
     The runtime has the same sensitivity to how a request's questions are batched.
3. **Gradients:** one backward pass on 8 dev questions.
   - Every LoRA B matrix must get a finite, non-zero gradient, which shows the adapter is on the forward path.
   - 10 AdamW steps on one fixed micro-batch must lower its loss.
   - It reports seconds per micro-batch and peak memory.
4. **Merge:** after those steps the adapters are folded into the BF16 weights. The merged model must answer like
   the adapter model (export check).

    STEPS=parity sbatch slurm/r0_gpu.sbatch     ->  $EOJEV_RUNS/train/r0/report.json

Exits 1 if R0 fails.
"""

from __future__ import annotations

import gc
import json
import random
import statistics
import sys
import time

import torch

from bench.common import DATASETS, RUNS, read_jsonl
from train import data
from train.d3model import D3Trainable, Example, answer_vector, load_images
from trainset.common import TRAIN

OUT = RUNS / "train" / "r0"
TEXT_ROWS = DATASETS / "raw" / "decision-index-work" / "parity-rows.jsonl"
MIN_AGREE, MAX_MEDIAN_DP = 0.999, 1e-3


def requests() -> list[tuple[str, dict, list]]:
    """(part, row, image paths) for every request."""
    out = [("text", r, []) for r in read_jsonl(TEXT_ROWS)]
    for fam in data.FAMILIES:
        for r in read_jsonl(data.family_file(fam)):
            if r["split"] == "dev":
                out.append(("image" if r["images"] else "text-dev", r, [TRAIN / p for p in r["images"]]))
    return out


def run_runtime(reqs) -> dict[str, list[float]]:
    from bench.runners.d3 import load_runtime

    runtime = load_runtime("d3-flash", batch_size=8, permutation_average=False)
    ref, errors, t0 = {}, 0, time.perf_counter()
    for k, (_, row, paths) in enumerate(reqs):
        try:
            res = runtime.system_one(state=row["state"], questions=row["questions"],
                                     images=load_images(paths) or None)
            for qid, q in row["questions"].items():
                ref[f"{row['id']}/{qid}"] = answer_vector(q, res["answers"][qid])
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"runtime error on {row['id']}: {type(exc).__name__}: {exc}", flush=True)
        if (k + 1) % 1000 == 0:
            print(f"  runtime {k + 1}/{len(reqs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    del runtime
    gc.collect()
    torch.cuda.empty_cache()
    print(f"runtime: {len(ref)} questions, {errors} request errors", flush=True)
    return ref


def compare(pairs: list[tuple[list[float], list[float]]]) -> dict:
    dp = [max(abs(a - b) for a, b in zip(x, y)) for x, y in pairs]
    agree = [max(range(len(x)), key=x.__getitem__) == max(range(len(y)), key=y.__getitem__) for x, y in pairs]
    s = sorted(dp)
    return {"questions": len(dp), "top1_agreement": round(sum(agree) / len(agree), 5),
            "median_max_abs_dp": statistics.median(dp), "p95_max_abs_dp": s[int(0.95 * (len(s) - 1))],
            "max_max_abs_dp": s[-1],
            "pass": sum(agree) / len(agree) >= MIN_AGREE and statistics.median(dp) <= MAX_MEDIAN_DP}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reqs = requests()
    print(f"{len(reqs)} requests: " + json.dumps({p: sum(x[0] == p for x in reqs) for p in ("text", "text-dev", "image")}),
          flush=True)
    ref = run_runtime(reqs)

    model = D3Trainable()
    n_params = model.add_lora(32, 64.0)
    model.enable_checkpointing()
    model.train()
    batches_same, flat, part_of = [], [], {}
    for part, row, paths in reqs:
        keys = [(f"{row['id']}/{qid}", row, qid) for qid in row["questions"] if f"{row['id']}/{qid}" in ref]
        for k, *_ in keys:
            part_of[k] = part if part != "text-dev" else "text"
        batches_same += [(keys[s: s + 8], paths) for s in range(0, len(keys), 8)]
        flat += [([k], paths) for k in keys]
    random.Random(0).shuffle(flat)
    batches_mixed = [([k for ks, _ in flat[s: s + 4] for k in ks], [p for _, p in flat[s: s + 4]])
                     for s in range(0, len(flat), 4)]

    def score(batches, mixed: bool, label: str) -> dict[str, list[float]]:
        out, t0 = {}, time.perf_counter()
        with torch.no_grad():
            for b, (keys, paths) in enumerate(batches):
                if mixed:  # one image list per question
                    exs = [Example(key=k, state=row["state"], question=row["questions"][qid], images=load_images(p))
                           for (k, row, qid), p in zip(keys, paths)]
                else:  # the request's images, shared by its questions
                    images = load_images(paths)
                    exs = [Example(key=k, state=row["state"], question=row["questions"][qid], images=images)
                           for k, row, qid in keys]
                for (k, *_), probs in zip(keys, model.probabilities(exs)):
                    out[k] = probs
                if (b + 1) % 500 == 0:
                    print(f"  trainer ({label}) {b + 1}/{len(batches)} batches ({time.perf_counter() - t0:.0f}s)", flush=True)
        return out

    def summarize(mine: dict[str, list[float]]) -> dict:
        by_part, by_family = {}, {}
        for key, probs in mine.items():
            by_part.setdefault(part_of[key], []).append((ref[key], probs))
            fam = key.split(":")[1] if key.startswith("tr:") else "decision-index"
            by_family.setdefault(fam, []).append((ref[key], probs))
        worst = sorted(mine, key=lambda k: -max(abs(a - b) for a, b in zip(ref[k], mine[k])))[:10]
        out = {"overall": compare([x for v in by_part.values() for x in v]),
               "parts": {k: compare(v) for k, v in by_part.items()},
               "families": {k: compare(v) for k, v in by_family.items()},
               "worst": [{"key": k, "runtime": [round(x, 4) for x in ref[k]][:12],
                          "trainer": [round(x, 4) for x in mine[k]][:12]} for k in worst]}
        out["pass"] = out["overall"]["pass"] and all(v["pass"] for v in out["parts"].values())
        return out

    report = {"lora_params": n_params, "lora_layers": len(model.lora)}
    report["same_batching"] = summarize(score(batches_same, False, "same batching"))
    report["training_batching"] = summarize(score(batches_mixed, True, "training batching"))
    r0 = report["same_batching"]["pass"]
    report["R0_pass"] = r0
    for mode in ("same_batching", "training_batching"):
        print(mode, json.dumps({k: report[mode][k] for k in ("overall", "parts", "pass")}, indent=1), flush=True)

    # Gradient check: every adapter on the forward path, finite gradients.
    dev = [it for fam, items in data.load_items("dev").items() for it in items[:2]]
    torch.cuda.reset_peak_memory_stats()
    from train.train import loss_terms

    exs = [data.to_example(it) for it in dev[:8]]
    logits, _ = model.logits(exs)
    _, _, loss = loss_terms(logits, exs, model.temperature, 0.25)
    loss.mean().backward()
    zero_b = [n for n, m in model.lora.items() if m.lora_b.grad is None or float(m.lora_b.grad.abs().sum()) == 0]
    finite = all(torch.isfinite(p.grad).all() for p in model.lora_parameters() if p.grad is not None)
    report["gradients"] = {"loss": float(loss.detach().mean()), "finite": bool(finite), "layers_without_grad": zero_b[:20],
                           "n_layers_without_grad": len(zero_b)}
    model.zero_grad(set_to_none=True)

    # Overfit check on one image micro-batch, with timing and peak memory.
    batch = [data.to_example(it) for it in [it for it in dev if it.row["images"]][:4]]
    opt = torch.optim.AdamW(model.lora_parameters(), lr=5e-4)
    losses, times = [], []
    for _ in range(10):
        torch.cuda.synchronize()
        t = time.perf_counter()
        logits, _ = model.logits(batch)
        _, _, loss = loss_terms(logits, batch, model.temperature, 0.25)
        loss.mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        losses.append(float(loss.mean()))
    report["overfit"] = {"losses": [round(x, 4) for x in losses], "decreased": losses[-1] < losses[0] * 0.8,
                         "sec_per_micro_batch_fwd_bwd": statistics.median(times[1:]),
                         "peak_gb": torch.cuda.max_memory_allocated() / 2**30}

    # Merge check: adapter model vs the same weights folded into BF16.
    probe = [data.to_example(it) for it in dev[:16]]
    model.eval()
    with torch.no_grad():
        before = model.probabilities(probe)
        model.merge_lora()
        after = model.probabilities(probe)
    report["merge"] = compare(list(zip(before, after)))

    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("gradients", "overfit", "merge")}, indent=1), flush=True)
    print("R0", "PASS" if r0 else "FAIL", flush=True)
    sys.exit(0 if r0 and finite and not zero_b else 1)


if __name__ == "__main__":
    main()
