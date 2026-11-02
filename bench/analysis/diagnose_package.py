"""Where a fine-tuned package's answers differ from d3-flash's, from run records only (follows compare_package.py).

On the questions both models answered, for every row of compare_package (the 27 guide rows, counting, Decision Index):
1. Churn: right-to-wrong and wrong-to-right counts, and mean confidence minus accuracy (> 0: overconfident).
2. Yes/no questions: the gold yes rate, each model's yes rate, and its accuracy on gold-yes and gold-no questions.
   A shifted prior shows as a yes rate moving away from gold, with accuracy falling on one side only. For
   multi-select rows, also the mean number of options answered yes per item.
3. Per sub-task (task_family): accuracy, or exact-set match per item for multi-select rows.
4. Multiple choice: the share of predictions on the first / last option against the gold share. Where a
   "none of these"-style option is offered, how often each model picks it.
5. Counting: mean signed error of the predicted count (predicted minus true).
6. fMoW-WILDS country: how concentrated each model's predictions are, and the most frequent new confusions.

    MODULE=bench.analysis.diagnose_package ARGS="eo-jev" sbatch -J eojev-diag slurm/convert_cpu.sbatch
    -> printed report and $EOJEV_RUNS/analysis/diagnose_<name>.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

import numpy as np

from bench.analysis.calibration_latency import ROWS, _load
from bench.analysis.compare_package import EXTRA
from bench.common import RUNS

NONE_RE = re.compile(r"\bnone\b|\bnot (in|present|visible|shown)\b|\bno such\b|\bcannot\b|\bis no\b|\bare no\b", re.I)


def mean(xs) -> float | None:
    xs = list(xs)
    return float(np.mean(xs)) if xs else None


def pct(v) -> str:
    return "  —  " if v is None else f"{100 * v:5.1f}"


def num(s) -> float | None:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", str(s))
    return float(m.group(1)) if m else None


def analyse(name: str, row: str, run: str, keep) -> dict | None:
    ra, _, rows = _load("d3", run)
    rb, _, _ = _load(name, run)
    a = {(r["id"], r["qid"]): r for r in ra if keep(r)}
    b = {(r["id"], r["qid"]): r for r in rb if keep(r)}
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    multi = "multi-select" in row
    out = {"row": row, "n": len(keys),
           "right_to_wrong": sum(a[k]["correct"] and not b[k]["correct"] for k in keys),
           "wrong_to_right": sum(b[k]["correct"] and not a[k]["correct"] for k in keys),
           "overconf": {m: mean(d[k]["conf"] for k in keys) - mean(float(d[k]["correct"]) for k in keys)
                        for m, d in (("d3", a), ("pkg", b))}}

    nk = [k for k in keys if a[k]["type"] == "noul"]
    if nk:
        yn = {"n": len(nk), "gold_yes": mean(float(a[k]["gold"]) for k in nk)}
        for m, d in (("d3", a), ("pkg", b)):
            yn[m] = {"yes_rate": mean(float(d[k]["pred"]) for k in nk),
                     "mean_p_yes": mean(d[k]["p_yes"] for k in nk),
                     "acc_gold_yes": mean(float(d[k]["correct"]) for k in nk if d[k]["gold"]),
                     "acc_gold_no": mean(float(d[k]["correct"]) for k in nk if not d[k]["gold"])}
        if multi:
            items = defaultdict(list)
            for k in nk:
                items[k[0]].append(k)
            yn["per_item_yes"] = {"gold": mean(sum(a[k]["gold"] for k in ks) for ks in items.values()),
                                  "d3": mean(sum(a[k]["pred"] for k in ks) for ks in items.values()),
                                  "pkg": mean(sum(b[k]["pred"] for k in ks) for ks in items.values()),
                                  "options": mean(len(ks) for ks in items.values())}
        out["yes_no"] = yn

    fam = defaultdict(list)
    for k in keys:
        fam[a[k]["task_family"]].append(k)
    sub = {}
    for f, ks in sorted(fam.items()):
        if multi:
            items = defaultdict(list)
            for k in ks:
                items[k[0]].append(k)
            sub[f] = {"n": len(items), "metric": "exact-set",
                      "d3": mean(all(a[k]["correct"] for k in v) for v in items.values()),
                      "pkg": mean(all(b[k]["correct"] for k in v) for v in items.values())}
        else:
            sub[f] = {"n": len(ks), "metric": "accuracy", "d3": mean(float(a[k]["correct"]) for k in ks),
                      "pkg": mean(float(b[k]["correct"]) for k in ks)}
    out["sub_tasks"] = sub

    ck = [k for k in keys if a[k]["type"] == "choice"]
    if ck:
        pos = Counter()
        none_rows = none_gold = 0
        none_pick = Counter()
        count_err = defaultdict(list)
        for k in ck:
            opts = list(rows[k[0]]["questions"][k[1]]["criteria"])
            last = len(opts) - 1
            for m, d in (("gold", a), ("d3", a), ("pkg", b)):
                v = d[k]["gold"] if m == "gold" else d[k]["pred"]
                i = opts.index(v) if v in opts else -1
                pos[(m, "first")] += i == 0
                pos[(m, "last")] += i == last
            nones = [o for o in opts if NONE_RE.search(str(o))]
            if nones:
                none_rows += 1
                none_gold += a[k]["gold"] in nones
                none_pick["d3"] += a[k]["pred"] in nones
                none_pick["pkg"] += b[k]["pred"] in nones
            g = num(a[k]["gold"])
            if row.startswith("Counting") and g is not None:
                for m, d in (("d3", a), ("pkg", b)):
                    p = num(d[k]["pred"])
                    if p is not None:
                        count_err[m].append(p - g)
        out["choice"] = {"n": len(ck), **{f"{m}_{w}": pos[(m, w)] / len(ck) for m in ("gold", "d3", "pkg")
                                           for w in ("first", "last")}}
        if none_rows:
            out["choice"]["none_offered"] = none_rows
            out["choice"]["none_gold"] = none_gold / none_rows
            out["choice"]["none_d3"] = none_pick["d3"] / none_rows
            out["choice"]["none_pkg"] = none_pick["pkg"] / none_rows
        if count_err:
            out["count_signed_error"] = {m: mean(v) for m, v in count_err.items()}

    if row == "fMoW-WILDS, country":
        preds = {m: Counter(d[k]["pred"] for k in keys) for m, d in (("d3", a), ("pkg", b))}
        gold = Counter(a[k]["gold"] for k in keys)
        out["fmow_country"] = {
            "top5_share": {m: sum(c for _, c in cnt.most_common(5)) / len(keys) for m, cnt in
                           (("gold", gold), *preds.items())},
            "most_predicted_pkg": preds["pkg"].most_common(6), "most_predicted_d3": preds["d3"].most_common(6),
            "new_confusions": Counter(f"{a[k]['gold']} -> {b[k]['pred']}" for k in keys
                                      if a[k]["correct"] and not b[k]["correct"]).most_common(10)}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("name")
    args = p.parse_args()
    spec = [(n, run, keep) for n, run, _m, keep in ROWS] + EXTRA
    report = [r for r in (analyse(args.name, *s) for s in spec) if r]

    print(f"\n1. CHURN AND CONFIDENCE ({args.name} vs d3-flash)")
    print(f"{'row':38s} {'n':>6s} {'right->wrong':>13s} {'wrong->right':>13s} {'conf-acc d3':>12s} {'pkg':>7s}")
    for r in report:
        print(f"{r['row'][:38]:38s} {r['n']:6d} {r['right_to_wrong']:13d} {r['wrong_to_right']:13d} "
              f"{r['overconf']['d3']:+12.3f} {r['overconf']['pkg']:+7.3f}")

    print("\n2. YES/NO QUESTIONS (rates in %)")
    print(f"{'row':38s} {'n':>6s} {'gold yes':>8s} | {'yes d3':>6s} {'pkg':>6s} | {'acc on yes d3':>13s} {'pkg':>6s} | "
          f"{'acc on no d3':>12s} {'pkg':>6s} | per-item yes: gold / d3 / pkg (of options)")
    for r in report:
        y = r.get("yes_no")
        if not y:
            continue
        it = y.get("per_item_yes")
        extra = f"{it['gold']:.2f} / {it['d3']:.2f} / {it['pkg']:.2f} (of {it['options']:.1f})" if it else ""
        print(f"{r['row'][:38]:38s} {y['n']:6d} {pct(y['gold_yes']):>8s} | {pct(y['d3']['yes_rate']):>6s} "
              f"{pct(y['pkg']['yes_rate']):>6s} | {pct(y['d3']['acc_gold_yes']):>13s} {pct(y['pkg']['acc_gold_yes']):>6s} | "
              f"{pct(y['d3']['acc_gold_no']):>12s} {pct(y['pkg']['acc_gold_no']):>6s} | {extra}")

    print("\n3. SUB-TASKS (rows with more than one; d3 -> pkg, % ; n)")
    for r in report:
        if len(r["sub_tasks"]) < 2:
            continue
        print(f"  {r['row']}")
        for f, s in r["sub_tasks"].items():
            diff = 100 * (s["pkg"] - s["d3"])
            flag = "  <<" if diff <= -5 and s["n"] >= 30 else "  >>" if diff >= 5 and s["n"] >= 30 else ""
            print(f"    {f[:44]:44s} {s['metric']:>9s} {pct(s['d3'])} -> {pct(s['pkg'])}  ({diff:+5.1f}, n={s['n']}){flag}")

    print("\n4. MULTIPLE-CHOICE POSITION AND 'NONE' OPTIONS (%)")
    print(f"{'row':38s} {'first: gold':>11s} {'d3':>6s} {'pkg':>6s} | {'last: gold':>10s} {'d3':>6s} {'pkg':>6s} | "
          f"none offered / gold / d3 / pkg")
    for r in report:
        c = r.get("choice")
        if not c:
            continue
        nn = (f"{c['none_offered']} / {pct(c['none_gold'])} / {pct(c['none_d3'])} / {pct(c['none_pkg'])}"
              if "none_offered" in c else "")
        print(f"{r['row'][:38]:38s} {pct(c['gold_first']):>11s} {pct(c['d3_first']):>6s} {pct(c['pkg_first']):>6s} | "
              f"{pct(c['gold_last']):>10s} {pct(c['d3_last']):>6s} {pct(c['pkg_last']):>6s} | {nn}")

    print("\n5. COUNTING: mean signed error of the predicted count (pred - true)")
    for r in report:
        if "count_signed_error" in r:
            e = r["count_signed_error"]
            print(f"  {r['row']:38s} d3 {e.get('d3', 0):+.2f}   pkg {e.get('pkg', 0):+.2f}")

    print("\n6. fMoW-WILDS COUNTRY")
    for r in report:
        if "fmow_country" in r:
            print(json.dumps(r["fmow_country"], indent=1))

    dest = RUNS / "analysis" / f"diagnose_{args.name}.json"
    dest.write_text(json.dumps(report, indent=1))
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
