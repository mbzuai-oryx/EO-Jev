"""A per-format yes/no threshold for a fine-tuned package, fitted on our own training dev rows (never on benchmark
answers), applied to the package's stored benchmark answers.

Format, read from the question text the same way on dev and benchmark questions:
- multi-select: one option of a "select every option that applies" item ("... Is "X" one of them?" / "... part of
  the correct answer?"), as fam10 and the DisasterM3 / DVL converters word it;
- single: every other yes/no question.

Steps:
1. rows: the dev split of the current training rows (trainset rows/final) that hold yes/no questions, with absolute
   image paths -> $EOJEV_DATASETS/converted/train_dev/noul_dev.jsonl
2. the package answers them (GPU):
   MODULE=bench.runners.d3 ARGS="--rows <that file> --out-dir $EOJEV_RUNS/<name>/train_dev_noul --model <package>"
3. fit: per format, the logit offset d (yes when logit(p_yes) + d >= 0) with the best dev accuracy, then every
   benchmark row with yes/no questions re-scored with those offsets (exact-set match for multi-select rows).

    MODULE=bench.analysis.yesno_devfit ARGS="rows" sbatch slurm/convert_cpu.sbatch
    MODULE=bench.analysis.yesno_devfit ARGS="fit eo-jev" sbatch slurm/convert_cpu.sbatch
    -> $EOJEV_RUNS/analysis/yesno_devfit_<name>.json and a printed table
"""

from __future__ import annotations

import argparse
import glob
import json
import re

import numpy as np

from bench.analysis.calibration_latency import ROWS, _load
from bench.analysis.yesno_threshold import GRID, Row, logit
from bench.common import CONVERTED, RUNS, read_jsonl, write_jsonl
from bench.metrics.score import question_records
from trainset.common import TRAIN

DEV_ROWS = CONVERTED / "train_dev" / "noul_dev.jsonl"
MULTI = re.compile(r"(one of them|part of the correct answer)\?\s*$")
FORMATS = ("multi-select", "single")


def fmt(question: dict) -> str:
    return "multi-select" if MULTI.search(question.get("instructions", "")) else "single"


def make_rows() -> None:
    from train import data  # the trainer's own family files (replay_d3.jsonl when present, not replay.jsonl too)

    rows = []
    for fam in data.FAMILIES:
        for r in read_jsonl(data.family_file(fam)):
            if r["split"] == "dev" and any(q["type"] == "noul" for q in r["questions"].values()):
                rows.append({**r, "images": [str(TRAIN / p) for p in r["images"]]})
    DEV_ROWS.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(DEV_ROWS, rows)
    n = {k: sum(fmt(q) == k for r in rows for q in r["questions"].values() if q["type"] == "noul") for k in FORMATS}
    print(f"{len(rows)} dev rows with yes/no questions -> {DEV_ROWS}; yes/no questions by format {n}")


def fit(name: str) -> None:
    rows = {r["id"]: r for r in read_jsonl(DEV_ROWS)}
    res = [r for f in sorted(glob.glob(str(RUNS / name / "train_dev_noul" / "chunk*.jsonl"))) for r in read_jsonl(f)]
    recs = [r for r in question_records(rows, res)[0] if r["type"] == "noul"]
    offsets, dev = {}, {}
    for k in FORMATS:
        sel = [r for r in recs if fmt(rows[r["id"]]["questions"][r["qid"]]) == k]
        z, gold = logit([r["p_yes"] for r in sel]), np.array([bool(r["gold"]) for r in sel])
        acc = [float((((z + d) >= 0) == gold).mean()) for d in GRID]
        best = max(acc)
        d = float(min((g for g, a in zip(GRID, acc) if a == best), key=abs))  # ties: the smallest shift
        offsets[k] = d
        dev[k] = {"questions": len(sel), "gold_yes": float(gold.mean()), "yes_before": float((z >= 0).mean()),
                  "yes_after": float((z + d >= 0).mean()), "acc_before": acc[list(GRID).index(0.0)], "acc_after": best,
                  "offset": d}
        print(f"dev {k:12s} n={len(sel):5d} gold yes {100 * gold.mean():5.1f}% | yes {100 * (z >= 0).mean():5.1f}% -> "
              f"{100 * (z + d >= 0).mean():5.1f}% | accuracy {100 * acc[list(GRID).index(0.0)]:5.1f} -> {100 * best:5.1f}"
              f" | offset {d:+.2f}")

    print(f"\n{'row':30s} {'score':>9s} {'format':>12s} | {'yes% gold':>9s} {'d3':>6s} {'pkg':>6s} {'+dev':>6s} | "
          f"{'d3':>6s} {'pkg':>6s} {'+dev':>6s} {'change':>7s}")
    out = []
    for label, run, _m, keep in ROWS:
        a_recs, _, qrows = _load("d3", run)
        a = {(r["id"], r["qid"]): r for r in a_recs if keep(r)}
        b = {(r["id"], r["qid"]): r for r in _load(name, run)[0] if keep(r)}
        keys = sorted(set(a) & set(b))
        if not keys or not any(a[k]["type"] == "noul" for k in keys):
            continue
        ra, rb = Row(label, keys, a), Row(label, keys, b)
        kinds = np.array([fmt(qrows[i]["questions"][q]) for i, q in keys])
        d = np.array([offsets[k] for k in kinds])[rb.noul]
        mix = sorted(set(kinds[rb.noul]))
        e = {"row": label, "metric": "exact-set" if rb.multi else "accuracy", "formats": mix,
             "yes_rate": {"gold": float(rb.gold[rb.noul].mean()), "d3": ra.yes_rate(0), "pkg": rb.yes_rate(0),
                          "pkg_dev": rb.yes_rate(d)},
             "score": {"d3": ra.score(0), "pkg": rb.score(0), "pkg_dev": rb.score(d)}}
        out.append(e)
        y, s = e["yes_rate"], e["score"]
        print(f"{label[:30]:30s} {e['metric']:>9s} {'+'.join(mix):>12s} | {100 * y['gold']:9.1f} {100 * y['d3']:6.1f} "
              f"{100 * y['pkg']:6.1f} {100 * y['pkg_dev']:6.1f} | {100 * s['d3']:6.1f} {100 * s['pkg']:6.1f} "
              f"{100 * s['pkg_dev']:6.1f} {100 * (s['pkg_dev'] - s['pkg']):+7.1f}")
    dest = RUNS / "analysis" / f"yesno_devfit_{name}.json"
    dest.write_text(json.dumps({"offsets": offsets, "dev": dev, "rows": out}, indent=1))
    print("\nwrote", dest)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["rows", "fit"])
    p.add_argument("name", nargs="?")
    args = p.parse_args()
    if args.step == "rows":
        make_rows()
    else:
        fit(args.name)


if __name__ == "__main__":
    main()
