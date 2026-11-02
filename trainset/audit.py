"""Audit of the final training rows (is the data balanced?), before any training run.

Per family (rows/final via train.data.family_file):
- rows and questions (train / dev), images per row, distinct images;
- question types; yes share of the yes / no questions;
- choice questions: option counts, and the share of questions whose gold is the first option (a position bias the
  model could learn; with shuffled options it is about 1 / options);
- counting questions (meta.count): share per count bin;
- rows with SAR images (bench.common.is_sar; must be 0).
With --mix (as train.train's) and --steps / --batch: each family's share of sampled questions, the epochs over its
training questions, and the yes share over all sampled yes / no questions (the model's yes / no balance follows it).

    MODULE=trainset.audit ARGS="--mix fam1_labels=0.1,... --steps 12000 --batch 16" sbatch slurm/convert_cpu.sbatch
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics

from bench.common import is_sar, read_jsonl
from train.data import FAMILIES, family_file

COUNT_BINS = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 10**9)]


def audit(fam: str) -> dict:
    rows = list(read_jsonl(family_file(fam)))
    qs = [(r, qid, q) for r in rows for qid, q in r["questions"].items()]
    train_q = [x for x in qs if x[0]["split"] == "train"]
    types = collections.Counter(q["type"] for _, _, q in train_q)
    yes = [bool(r["answer"][qid]) for r, qid, q in train_q if q["type"] == "noul"]
    choice = [(r, qid, q) for r, qid, q in train_q if q["type"] == "choice"]
    first = [list(q["criteria"])[0] == r["answer"][qid] for r, qid, q in choice]
    n_opts = [len(q["criteria"]) for _, _, q in choice]
    counts = [r["meta"]["count"] for r, _, _ in train_q if isinstance(r["meta"].get("count"), int)]
    return {
        "file": family_file(fam).name, "rows": len(rows), "questions_train": len(train_q),
        "questions_dev": len(qs) - len(train_q), "distinct_images": len({im for r in rows for im in r["images"]}),
        "images_per_row": dict(sorted(collections.Counter(len(r["images"]) for r in rows).items())),
        "types": dict(types),
        "yes_share": round(sum(yes) / len(yes), 3) if yes else None, "yes_no_questions": len(yes),
        "choice_options_median": statistics.median(n_opts) if n_opts else None,
        "gold_first_share": round(sum(first) / len(first), 3) if first else None,
        "uniform_first_share": round(statistics.mean(1 / n for n in n_opts), 3) if n_opts else None,
        "count_bins": ({f"{lo}-{hi}" if hi < 10**9 else f"{lo}+": round(sum(lo <= c <= hi for c in counts) / len(counts), 3)
                        for lo, hi in COUNT_BINS} if counts else None),
        "sar_rows": sum(map(is_sar, rows)),
        "d3_targets": sum(bool(r["meta"].get("d3_probs")) for r in rows),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mix")
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    report = {f: audit(f) for f in FAMILIES if family_file(f).exists()}
    for f, a in report.items():
        print(f, json.dumps(a), flush=True)
    if args.mix:
        w = {k: float(v) for k, v in (kv.split("=") for kv in args.mix.split(","))}
        total = sum(w.values())
        sampled = args.steps * args.batch
        yes_w = no_w = 0.0
        print(f"\nmix over {args.steps} steps x {args.batch} = {sampled} sampled questions:")
        for f, wt in sorted(w.items(), key=lambda kv: -kv[1]):
            a = report.get(f)
            if a is None:
                print(f"  {f}: NO ROWS")
                continue
            share = wt / total
            n_yn = a["yes_no_questions"] / max(1, a["questions_train"])  # share of the family's questions that are yes / no
            if a["yes_share"] is not None:
                yes_w += share * n_yn * a["yes_share"]
                no_w += share * n_yn * (1 - a["yes_share"])
            print(f"  {f:22s} share {100 * share:5.1f}%  epochs {sampled * share / max(1, a['questions_train']):5.2f}  "
                  f"yes/no {100 * n_yn:4.0f}% of its questions, yes {a['yes_share']}")
        if yes_w + no_w:
            print(f"yes share over all sampled yes / no questions: {100 * yes_w / (yes_w + no_w):.1f}%")


if __name__ == "__main__":
    main()
