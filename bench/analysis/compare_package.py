"""A fine-tuned package against d3-flash on the benchmark rows, from run records.

**Rows:**
- the 27 benchmark-guide rows (bench/analysis/calibration_latency.ROWS; SAR rows left out, bench.common.is_sar);
- the three counting sets;
- the Decision Index general-decision check: 6,251 text questions, pooled and per source benchmark. This is the
  regression test for forgetting.

Every comparison uses the questions both models answered.

**Per row,** with the paired difference package minus d3-flash and a 95% bootstrap CI over images (1,000
resamples):
- the benchmark's own score: accuracy, except
  - exact-set match for multi-select rows (every per-option yes/no right);
  - all-rotations-correct for LHRS-Bench (strict CircularEval);
- question accuracy, ECE (15 bins) and Brier score.

A positive score difference, and a negative ECE or Brier difference, favour the package.

    MODULE=bench.analysis.compare_package ARGS="eo-jev" sbatch slurm/convert_cpu.sbatch
    -> $EOJEV_RUNS/analysis/compare_<name>.json and a printed table
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.analysis.calibration_latency import ROWS, _all, _bench, _load
from bench.analysis.calibration_study import Arrays, ci
from bench.common import RUNS

DI = ["BANKING77", "ARC-Challenge", "ARC-Easy", "MMLU", "WinoGrande", "HellaSwag"]
EXTRA = [("Counting, NWPU VHR-10", "det_count_nwpu_vhr10", _all),
         ("Counting, DOTA-v2 val", "det_count_dota_v2_val", _all),
         ("Counting, absent classes", "det_count_vhr10_absent", _all),
         ("Decision Index, all (general text)", "decision_index", _all)] + \
        [(f"Decision Index, {b}", "decision_index", _bench(b)) for b in DI]
CHUNK = 100


def own_items(name: str, keys: list, rec: dict):
    """Items the benchmark scores as a whole, or None when its score is question accuracy."""
    if "multi-select" in name:
        by = lambda k: k[0]  # noqa: E731 - one multi-select item = one row id, its options are the questions
    elif name == "LHRS-Bench":
        by = lambda k: rec[k]["group"]  # noqa: E731 - the four option rotations of an item share its group
    else:
        return None
    items = defaultdict(list)
    for i, k in enumerate(keys):
        items[by(k)].append(i)
    return list(items.values())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("name", help="package run name: runs are in $EOJEV_RUNS/<name>/<d3 run dir>")
    p.add_argument("--baseline", default="d3", help="d3, or another package run name (e.g. eo-jev)")
    p.add_argument("--boot", type=int, default=1000)
    args = p.parse_args()
    base_name = "d3-flash" if args.baseline == "d3" else args.baseline

    spec = [(n, run, keep) for n, run, _models, keep in ROWS] + EXTRA
    table = []
    for name, run, keep in spec:
        a = {(r["id"], r["qid"]): r for r in _load(args.baseline, run)[0] if keep(r)}
        b = {(r["id"], r["qid"]): r for r in _load(args.name, run)[0] if keep(r)}
        keys = sorted(set(a) & set(b))
        if not keys:
            print(f"{name}: no common questions ({base_name} {len(a)}, {args.name} {len(b)}), skipped", flush=True)
            continue
        table.append((name, run, a, b, keys))
    groups = sorted({(run, a[k]["group"]) for _, run, a, _, keys in table for k in keys})
    gidx = {g: i for i, g in enumerate(groups)}
    rng = np.random.default_rng(0)
    C = np.stack([np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
                  for _ in range(args.boot)]).astype(np.float32)

    out = []
    for name, run, a, b, keys in table:
        g = np.array([gidx[(run, a[k]["group"])] for k in keys])
        arrs = {s: Arrays(np.array([d[k]["conf"] for k in keys], dtype=float),
                          np.array([float(d[k]["correct"]) for k in keys]),
                          np.array([d[k]["brier"] for k in keys], dtype=float)) for s, d in (("d3", a), ("pkg", b))}
        point = {s: arrs[s].metrics(np.ones((1, len(keys)))) for s in arrs}
        draws = {s: defaultdict(list) for s in arrs}
        for start in range(0, len(C), CHUNK):
            W = C[start: start + CHUNK][:, g]
            for s in arrs:
                for k, v in arrs[s].metrics(W).items():
                    draws[s][k].append(v)
        draws = {s: {k: np.concatenate(v) for k, v in d.items()} for s, d in draws.items()}
        entry = {"row": name, "run": run, "questions": len(keys), "groups": int(len(np.unique(g)))}
        for k in ("accuracy", "ece", "brier"):
            entry[k] = {"d3": ci(point["d3"][k][0], draws["d3"][k]), "pkg": ci(point["pkg"][k][0], draws["pkg"][k]),
                        "diff": ci(point["pkg"][k][0] - point["d3"][k][0], draws["pkg"][k] - draws["d3"][k])}
        items = own_items(name, keys, a)
        if items is None:
            entry["score"] = {"name": "accuracy", **entry["accuracy"]}
        else:
            item_g = np.array([g[ix[0]] for ix in items])
            allc = {s: np.array([all(arrs[s].correct[i] == 1 for i in ix) for ix in items], dtype=float) for s in arrs}
            Wi = C[:, item_g]
            est = {s: float(allc[s].mean()) for s in arrs}
            dr = {s: (Wi @ allc[s]) / Wi.sum(1) for s in arrs}
            entry["score"] = {"name": "exact-set match" if "multi-select" in name else "all rotations correct",
                              "items": len(items), "d3": ci(est["d3"], dr["d3"]), "pkg": ci(est["pkg"], dr["pkg"]),
                              "diff": ci(est["pkg"] - est["d3"], dr["pkg"] - dr["d3"])}
        out.append(entry)

    def sig(c):  # +1 CI wholly above 0, -1 wholly below, 0 otherwise
        return 1 if c[1] is not None and c[1] > 0 else -1 if c[2] is not None and c[2] < 0 else 0

    print(f"\n{args.name} vs {base_name} (difference = {args.name} - {base_name}, 95% CI; column 'd3' = {base_name})")
    print(f"{'row':38s} {'n':>6s} {'score':>22s} {'d3':>6s} {'pkg':>6s} {'diff [CI]':>24s}   {'ECE d3':>6s} {'pkg':>6s} {'diff [CI]':>24s}")
    for e in out:
        s, ec = e["score"], e["ece"]
        print(f"{e['row'][:38]:38s} {e['questions']:6d} {s['name'][:22]:>22s} {100 * s['d3'][0]:6.1f} {100 * s['pkg'][0]:6.1f} "
              f"{100 * s['diff'][0]:+6.1f} [{100 * (s['diff'][1] or 0):+5.1f},{100 * (s['diff'][2] or 0):+5.1f}]   "
              f"{ec['d3'][0]:6.3f} {ec['pkg'][0]:6.3f} {ec['diff'][0]:+7.3f} [{ec['diff'][1] or 0:+.3f},{ec['diff'][2] or 0:+.3f}]")
    eo = [e for e in out if not e["row"].startswith("Decision Index")]
    summary = {"rows": len(eo),
               "score_up": sum(sig(e["score"]["diff"]) > 0 for e in eo), "score_down": sum(sig(e["score"]["diff"]) < 0 for e in eo),
               "ece_down": sum(sig(e["ece"]["diff"]) < 0 for e in eo), "ece_up": sum(sig(e["ece"]["diff"]) > 0 for e in eo)}
    print("\nEO rows: score significantly up", summary["score_up"], "/ down", summary["score_down"],
          "| ECE significantly down", summary["ece_down"], "/ up", summary["ece_up"])
    dest = RUNS / "analysis" / (f"compare_{args.name}.json" if args.baseline == "d3"
                                else f"compare_{args.name}_vs_{args.baseline}.json")
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"name": args.name, "baseline": base_name, "summary": summary, "rows": out}, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main()
