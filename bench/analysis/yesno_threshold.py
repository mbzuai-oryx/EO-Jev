"""Does a shifted yes/no threshold explain a fine-tuned package's losses on yes/no-scored rows? From run records only.

On every benchmark row with yes/no (`noul`) questions, using the questions both models answered, the package's yes/no
answers are re-decided with a logit offset d: yes when logit(p_yes) + d >= 0. The other questions keep their answers.

**Offsets reported:**
- **0:** as run.
- **global:** one offset for all rows, chosen without labels. It is the median of logit(p_pkg) - logit(p_d3) over
  every yes/no question, negated, so it undoes the average shift the training introduced.
- **match:** per row, the offset that gives the package the yes rate d3-flash has on that row (no labels;
  transductive).
- **oracle:** per row, the best offset using the gold answers. This is an upper bound, also computed for d3-flash so
  the two are compared on equal terms.

Also reported is the AUROC of p_yes, which is threshold-free: if the package's AUROC is no worse than d3-flash's, the
loss is a threshold (prior) problem, not lost discrimination.

Row scores are those of compare_package: exact-set match per item for multi-select rows, accuracy otherwise.

    MODULE=bench.analysis.yesno_threshold ARGS="eo-jev" sbatch -J eojev-yesno slurm/convert_cpu.sbatch
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from bench.analysis.calibration_latency import ROWS, _load
from bench.common import RUNS
from bench.metrics.score import auroc

GRID = np.round(np.arange(-4, 4.0001, 0.05), 2)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Row:
    def __init__(self, name: str, keys: list, recs: dict):
        self.multi = "multi-select" in name
        self.keys = keys
        self.noul = np.array([recs[k]["type"] == "noul" for k in keys])
        self.correct = np.array([bool(recs[k]["correct"]) for k in keys])
        self.gold = np.array([bool(recs[k]["gold"]) if recs[k]["type"] == "noul" else False for k in keys])
        self.z = np.where(self.noul, logit([recs[k].get("p_yes", 0.5) for k in keys]), 0.0)
        items = defaultdict(list)
        for i, k in enumerate(keys):
            items[k[0] if self.multi else k].append(i)
        self.items = list(items.values())

    def score(self, d: float) -> float:
        c = self.correct.copy()
        c[self.noul] = ((self.z[self.noul] + d) >= 0) == self.gold[self.noul]
        return float(np.mean([c[ix].all() for ix in self.items])) if self.multi else float(c.mean())

    def yes_rate(self, d: float) -> float:
        return float(((self.z[self.noul] + d) >= 0).mean())

    def auroc(self) -> float:
        return auroc(self.z[self.noul], self.gold[self.noul])

    def oracle(self) -> tuple[float, float]:
        scores = [self.score(d) for d in GRID]
        i = int(np.argmax(scores))
        return float(GRID[i]), scores[i]

    def match(self, rate: float) -> float:
        """Offset giving this yes rate."""
        z = np.sort(self.z[self.noul])
        k = int(round((1 - rate) * len(z)))
        if k <= 0:
            return float(-z[0] + 1e-6)
        if k >= len(z):
            return float(-z[-1] - 1e-6)
        return float(-(z[k - 1] + z[k]) / 2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("name")
    args = p.parse_args()
    rows = []
    for name, run, _m, keep in ROWS:
        a = {(r["id"], r["qid"]): r for r in _load("d3", run)[0] if keep(r)}
        b = {(r["id"], r["qid"]): r for r in _load(args.name, run)[0] if keep(r)}
        keys = sorted(set(a) & set(b))
        if keys and any(a[k]["type"] == "noul" for k in keys):
            rows.append((name, Row(name, keys, a), Row(name, keys, b)))
    shift = np.concatenate([rb.z[rb.noul] - ra.z[ra.noul] for _, ra, rb in rows])
    d_global = float(-np.median(shift))
    print(f"label-free global offset: {d_global:+.3f} (median logit shift package - d3 = {np.median(shift):+.3f}, "
          f"mean {shift.mean():+.3f}, over {len(shift)} yes/no questions)\n")

    hdr = (f"{'row':30s} {'score':>9s} | {'AUROC d3':>8s} {'pkg':>6s} | {'yes% d3':>7s} {'pkg':>6s} {'+glob':>6s} | "
           f"{'d3':>6s} {'pkg':>6s} {'+glob':>6s} {'+match':>6s} | {'oracle d3':>9s} {'pkg':>6s}")
    print(hdr)
    out = []
    for name, ra, rb in rows:
        e = {"row": name, "metric": "exact-set" if ra.multi else "accuracy", "yes_no_questions": int(ra.noul.sum()),
             "auroc": {"d3": ra.auroc(), "pkg": rb.auroc()},
             "yes_rate": {"gold": float(ra.gold[ra.noul].mean()), "d3": ra.yes_rate(0), "pkg": rb.yes_rate(0),
                          "pkg_global": rb.yes_rate(d_global)},
             "score": {"d3": ra.score(0), "pkg": rb.score(0), "pkg_global": rb.score(d_global)}}
        dm = rb.match(ra.yes_rate(0))
        e["score"]["pkg_match"], e["match_offset"] = rb.score(dm), dm
        (od, os_), (opd, ops) = ra.oracle(), rb.oracle()
        e["oracle"] = {"d3": {"offset": od, "score": os_}, "pkg": {"offset": opd, "score": ops}}
        out.append(e)
        s, y = e["score"], e["yes_rate"]
        print(f"{name[:30]:30s} {e['metric']:>9s} | {e['auroc']['d3']:8.3f} {e['auroc']['pkg']:6.3f} | "
              f"{100 * y['d3']:7.1f} {100 * y['pkg']:6.1f} {100 * y['pkg_global']:6.1f} | "
              f"{100 * s['d3']:6.1f} {100 * s['pkg']:6.1f} {100 * s['pkg_global']:6.1f} {100 * s['pkg_match']:6.1f} | "
              f"{100 * os_:9.1f} {100 * ops:6.1f}")
    print("\n(yes% gold per row is in the JSON; oracle offsets: "
          + ", ".join(f"{e['row'][:18]} d3 {e['oracle']['d3']['offset']:+.2f}/pkg {e['oracle']['pkg']['offset']:+.2f}"
                      for e in out) + ")")
    dest = RUNS / "analysis" / f"yesno_threshold_{args.name}.json"
    dest.write_text(json.dumps({"global_offset": d_global, "rows": out}, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main()
