"""d3-flash's own probabilities for the replay rows, or for another family's rows.

Reads rows/final/<family>.jsonl and writes rows/final/<family>_d3.jsonl (--family, default replay), adding
meta.d3_probs = {question id: probabilities in option order}. Rows with meta.answer_from_d3 (family 14: counting of
objects without labels) also take d3-flash's top option as their answer. Training on these targets distils d3-flash's current answers back into the adapter
(self-distillation replay), so its general decision behaviour is kept rather than relearned from gold labels.

The run is resumable: finished rows accumulate in replay_d3.partial.jsonl. It also reports how often d3-flash's top
answer matches the gold answer and, for jev rows, the corpus teacher's top answer (meta.target_probs).

    STEPS=replay sbatch slurm/r0_gpu.sbatch
    MODULE=train.label_replay ARGS="--family fam14_count_distill" sbatch slurm/run_py.sbatch
"""

from __future__ import annotations

import argparse
import collections
import json
import time

from bench.common import read_jsonl, write_jsonl
from bench.runners.d3 import load_runtime
from train.d3model import answer_vector, gold_index, load_images
from trainset.common import ROWS, TRAIN



def argmax(v):
    return max(range(len(v)), key=v.__getitem__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="replay")
    args = p.parse_args()
    SRC, DST = ROWS / "final" / f"{args.family}.jsonl", ROWS / "final" / f"{args.family}_d3.jsonl"
    PARTIAL = ROWS / "final" / f"{args.family}_d3.partial.jsonl"
    rows = list(read_jsonl(SRC))
    done = {r["id"]: r for r in read_jsonl(PARTIAL)} if PARTIAL.exists() else {}
    print(f"{len(rows)} {args.family} rows, {len(done)} already labelled", flush=True)
    runtime = load_runtime("d3-flash", batch_size=8, permutation_average=False)
    t0, errors = time.perf_counter(), 0
    with open(PARTIAL, "a") as f:
        for k, row in enumerate(rows):
            if row["id"] in done:
                continue
            try:
                res = runtime.system_one(state=row["state"], questions=row["questions"],
                                         images=load_images([TRAIN / p for p in row["images"]]) or None)
                probs = {qid: answer_vector(q, res["answers"][qid]) for qid, q in row["questions"].items()}
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"error on {row['id']}: {type(exc).__name__}: {exc}", flush=True)
                continue
            out = {**row, "meta": {**row["meta"], "d3_probs": probs}}
            if row["meta"].get("answer_from_d3"):
                out["answer"] = {qid: list(q["criteria"])[argmax(probs[qid])] for qid, q in row["questions"].items()}
            done[row["id"]] = out
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            if (k + 1) % 1000 == 0:
                f.flush()
                print(f"  {k + 1}/{len(rows)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    labelled = [done[r["id"]] for r in rows if r["id"] in done]
    write_jsonl(DST, labelled)

    stats = collections.defaultdict(collections.Counter)
    for r in labelled:
        src = r["meta"]["source"]
        for qid, q in r["questions"].items():
            p = r["meta"]["d3_probs"][qid]
            stats[src]["questions"] += 1
            stats[src]["d3_top_is_gold"] += argmax(p) == gold_index(q, r["answer"][qid])
            teacher = r["meta"].get("target_probs")
            if teacher and len(teacher) == len(p):
                stats[src]["with_teacher"] += 1
                stats[src]["d3_top_is_teacher_top"] += argmax(p) == argmax(teacher)
    print(f"wrote {len(labelled)} rows to {DST} ({errors} errors this run)")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
