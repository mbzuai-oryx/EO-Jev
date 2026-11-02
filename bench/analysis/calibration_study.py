"""Calibration study for d3-flash ("JEV"), Qwen3.5-9B and the open specialists, from run records alone (no new
inference). Three experiments:
1. **Reliability:** accuracy against confidence (15 equal-width bins on the top answer's probability).
2. **ECE difference per row:** d3-flash minus each baseline on each of the 29 benchmark-guide rows. Negative favours
   d3-flash.
3. **Selective prediction:** questions sorted by confidence; the least confident are discarded step by step, and the
   error rate on the rest is plotted against the share kept (the risk–coverage curve).

**Rows and systems:** the 29 rows and their specialists are those of bench/analysis/calibration_latency.ROWS (the
benchmark guide). Specialists are scored in their option-probability mode on the common question format. Written
answers carry no confidence, and own-format (native) runs are a format check only. A comparison always uses the
questions that every system in it answered.

**Metrics.** Each comes with a 95% bootstrap CI: 1,000 resamples of image groups (group_id). The same draws serve every
system, row and pooled set, so differences are paired.
- **ECE:** 15 equal-width bins (as bench/metrics/score.py).
- **Adaptive ECE:** 15 equal-mass bins, each holding 1/15 of the questions.
- **Brier:** squared error of the full probability vector (two-class for yes/no), as score.py.
- **AURC:** mean error rate over all coverages, the most confident questions kept first. Tied confidences enter
  together. Lower is better.
- **E-AURC:** AURC minus that of a perfect ordering with the same accuracy (all mistakes last; Geifman et al. 2019:
  r + (1 - r) ln(1 - r)). It removes the part of AURC that is just the error rate, leaving how well confidence picks
  out mistakes.

**Pooled results** (experiments 1 and 3, and the overall metrics) weight every row equally: each question gets weight
1 / its row's size, so large rows such as BigEarthNet or scene classification do not dominate.

    MODULE=bench.analysis.calibration_study sbatch -J eojev-calib slurm/convert_cpu.sbatch
    -> $EOJEV_RUNS/analysis/calibration_study.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np

from bench.analysis.calibration_latency import ROWS, _load
from bench.common import RUNS

BINS = 15
GRID = np.round(np.linspace(0.02, 1.0, 50), 4)
METRICS = ("accuracy", "ece", "aece", "brier", "aurc", "eaurc")
CHUNK = 50
NAMES = {"d3": "d3-flash", "qwen": "Qwen3.5-9B"}


class Arrays:
    """One system's answers on a question set, with the orderings the weighted metrics reuse."""

    def __init__(self, conf: np.ndarray, correct: np.ndarray, brier: np.ndarray):
        self.conf, self.correct, self.brier = conf, correct, brier
        self.err = 1.0 - correct
        n = len(conf)
        self.bin = np.clip(np.digitize(conf, np.linspace(0, 1, BINS + 1)[1:-1]), 0, BINS - 1)
        self.asc = np.argsort(conf, kind="stable")
        self.desc = np.argsort(-conf, kind="stable")
        c = conf[self.desc]
        last = np.flatnonzero(np.r_[c[1:] != c[:-1], True])  # last position of each run of equal confidence
        self.block_end = last[np.searchsorted(last, np.arange(n))]

    def _binned(self, W: np.ndarray, idx: np.ndarray, conf: np.ndarray, correct: np.ndarray):
        """Per-bin weight, confidence sum and correct sum ([B, BINS] each); idx is [n] or [B, n]."""
        B = W.shape[0]
        flat = (idx + BINS * np.arange(B)[:, None]).ravel()
        size = B * BINS
        sw = np.bincount(flat, W.ravel(), size).reshape(B, BINS)
        sc = np.bincount(flat, (W * conf).ravel(), size).reshape(B, BINS)
        sa = np.bincount(flat, (W * correct).ravel(), size).reshape(B, BINS)
        return sw, sc, sa

    def _ranked(self, W: np.ndarray):
        Wd = W[:, self.desc]
        cw = np.cumsum(Wd, 1)
        ce = np.cumsum(Wd * self.err[self.desc], 1)
        cw_b, ce_b = cw[:, self.block_end], ce[:, self.block_end]
        risk = ce_b / np.maximum(cw_b, 1e-12)
        return Wd, cw_b / cw[:, -1:], risk

    def metrics(self, W: np.ndarray) -> dict[str, np.ndarray]:
        tot = W.sum(1)
        out = {"accuracy": (W @ self.correct) / tot, "brier": (W @ self.brier) / tot}
        _, sc, sa = self._binned(W, np.broadcast_to(self.bin, W.shape), self.conf, self.correct)
        out["ece"] = np.abs(sc - sa).sum(1) / tot
        Ws = W[:, self.asc]
        mid = (np.cumsum(Ws, 1) - Ws / 2) / tot[:, None]
        idx = np.minimum((mid * BINS).astype(np.int64), BINS - 1)
        _, sc, sa = self._binned(Ws, idx, self.conf[self.asc], self.correct[self.asc])
        out["aece"] = np.abs(sc - sa).sum(1) / tot
        Wd, _, risk = self._ranked(W)
        out["aurc"] = (Wd * risk).sum(1) / tot
        r = np.clip(1 - out["accuracy"], 0, 1)
        out["eaurc"] = out["aurc"] - (r + (1 - r) * np.log(np.clip(1 - r, 1e-12, None)))
        return out

    def reliability(self, W: np.ndarray):
        sw, sc, sa = self._binned(W, np.broadcast_to(self.bin, W.shape), self.conf, self.correct)
        with np.errstate(invalid="ignore", divide="ignore"):
            return sw / W.sum(1, keepdims=True), sc / sw, sa / sw

    def risk_coverage(self, W: np.ndarray) -> np.ndarray:
        _, cov, risk = self._ranked(W)
        out = np.empty((W.shape[0], len(GRID)))
        for b in range(W.shape[0]):
            i = np.minimum(np.searchsorted(cov[b], GRID - 1e-9), cov.shape[1] - 1)
            out[b] = risk[b, i]
        return out


def ci(point: float, draws: np.ndarray) -> list[float]:
    d = draws[np.isfinite(draws)]
    if not len(d):
        return [round(float(point), 5), None, None]
    return [round(float(point), 5), round(float(np.percentile(d, 2.5)), 5), round(float(np.percentile(d, 97.5)), 5)]


def evaluate(parts: list[dict], systems: list[str], C: np.ndarray, curves: bool = False) -> dict:
    """Metrics of each system on the concatenated parts, the differences d3 - other (paired bootstrap), and optionally
    reliability bins and risk–coverage curves."""
    g = np.concatenate([p["g"] for p in parts])
    base = np.concatenate([p["w"] for p in parts])
    arr = {s: Arrays(*(np.concatenate([p[k][s] for p in parts]) for k in ("conf", "correct", "brier"))) for s in systems}
    point_w = base[None, :]
    point = {s: {k: float(v[0]) for k, v in arr[s].metrics(point_w).items()} for s in systems}
    draws = {s: defaultdict(list) for s in systems}
    rel_draws = {s: [] for s in systems}
    rc_draws = {s: [] for s in systems}
    for start in range(0, len(C), CHUNK):
        W = C[start: start + CHUNK][:, g] * base[None, :]
        for s in systems:
            for k, v in arr[s].metrics(W).items():
                draws[s][k].append(v)
            if curves:
                rel_draws[s].append(arr[s].reliability(W)[2])
                rc_draws[s].append(arr[s].risk_coverage(W))
    draws = {s: {k: np.concatenate(v) for k, v in d.items()} for s, d in draws.items()}
    out = {"n": int(len(g)), "groups": int(len(np.unique(g))),
           "systems": {s: {k: ci(point[s][k], draws[s][k]) for k in METRICS} for s in systems}, "delta": {}}
    for s in systems:
        if s == "d3":
            continue
        out["delta"][s] = {}
        for k in METRICS:
            d = draws["d3"][k] - draws[s][k]
            out["delta"][s][k] = ci(point["d3"][k] - point[s][k], d) + [round(float(np.mean(d < 0)), 4)]
    if curves:
        out["reliability"], out["risk_coverage"] = {}, {}
        for s in systems:
            share, conf, acc = arr[s].reliability(point_w)
            accd = np.concatenate(rel_draws[s])
            with np.errstate(invalid="ignore"):
                lo, hi = np.nanpercentile(accd, 2.5, axis=0), np.nanpercentile(accd, 97.5, axis=0)
            out["reliability"][s] = [
                {"bin": b, "share": round(float(share[0, b]), 5),
                 "conf": None if not np.isfinite(conf[0, b]) else round(float(conf[0, b]), 4),
                 "acc": None if not np.isfinite(acc[0, b]) else round(float(acc[0, b]), 4),
                 "acc_lo": None if not np.isfinite(lo[b]) else round(float(lo[b]), 4),
                 "acc_hi": None if not np.isfinite(hi[b]) else round(float(hi[b]), 4)} for b in range(BINS)]
            rc = arr[s].risk_coverage(point_w)[0]
            rcd = np.concatenate(rc_draws[s])
            out["risk_coverage"][s] = {"risk": np.round(rc, 4).tolist(),
                                       "lo": np.round(np.percentile(rcd, 2.5, axis=0), 4).tolist(),
                                       "hi": np.round(np.percentile(rcd, 97.5, axis=0), 4).tolist()}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--out", default=str(RUNS / "analysis" / "calibration_study.json"))
    args = p.parse_args()
    t0 = time.time()

    # every system's records per row: (id, qid) -> (conf, correct, brier, group)
    table = []
    for name, run, models, keep in ROWS:
        recs = {}
        for s in ["d3", "qwen", *models]:
            got = {(r["id"], r["qid"]): (r["conf"], float(r["correct"]), r["brier"], r["group"])
                   for r in _load(s, run)[0] if keep(r)}
            if got:
                recs[s] = got
        table.append((name, run, recs))
        print(f"{name}: " + ", ".join(f"{s} {len(v)}" for s, v in recs.items()), flush=True)
    groups = sorted({(run, v[3]) for _, run, recs in table for v in recs["d3"].values()})
    gidx = {k: i for i, k in enumerate(groups)}
    rng = np.random.default_rng(0)
    C = np.stack([np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
                  for _ in range(args.boot)]).astype(np.float32)
    print(f"{len(groups)} image groups, {args.boot} bootstrap draws ({time.time() - t0:.0f}s)", flush=True)

    def part(name, run, recs, systems, row_weight: bool) -> dict | None:
        keys = sorted(set.intersection(*(set(recs[s]) for s in systems)))
        if not keys:
            return None
        n = len(keys)
        return {"row": name, "n": n, "g": np.array([gidx[(run, recs["d3"][k][3])] for k in keys]),
                "w": np.full(n, 1.0 / n if row_weight else 1.0),
                "conf": {s: np.array([recs[s][k][0] for k in keys]) for s in systems},
                "correct": {s: np.array([recs[s][k][1] for k in keys]) for s in systems},
                "brier": {s: np.array([recs[s][k][2] for k in keys]) for s in systems}}

    report = {"meta": {"bins": BINS, "boot": args.boot, "grid": GRID.tolist(), "names": NAMES,
                       "metrics": list(METRICS)}, "rows": [], "pooled": {}}
    specialists = sorted({s for _, _, recs in table for s in recs} - {"d3", "qwen"})
    for name, run, recs in table:
        entry = {"row": name, "run": run, "comparisons": {}}
        for s in recs:
            if s == "d3":
                continue
            pt = part(name, run, recs, ["d3", s], row_weight=False)
            if pt is None:
                continue
            res = evaluate([pt], ["d3", s], C, curves=(s == "qwen"))
            if s == "qwen":
                entry["risk_coverage"] = {k: v["risk"] for k, v in res.pop("risk_coverage").items()}
                res.pop("reliability")
            entry["comparisons"][s] = res
        report["rows"].append(entry)
        q = entry["comparisons"].get("qwen", {}).get("delta", {}).get("qwen", {}).get("ece")
        print(f"  {name}: dECE vs Qwen {q} ({time.time() - t0:.0f}s)", flush=True)

    parts = [part(n, r, recs, ["d3", "qwen"], row_weight=True) for n, r, recs in table]
    report["pooled"]["qwen"] = {"rows": [pt["row"] for pt in parts if pt],
                                **evaluate([pt for pt in parts if pt], ["d3", "qwen"], C, curves=True)}
    print(f"pooled vs Qwen done ({time.time() - t0:.0f}s)", flush=True)
    for s in specialists:
        parts = [part(n, r, recs, ["d3", "qwen", s], row_weight=True) for n, r, recs in table if s in recs]
        parts = [pt for pt in parts if pt]
        if parts:
            report["pooled"][s] = {"rows": [pt["row"] for pt in parts], **evaluate(parts, ["d3", "qwen", s], C, curves=True)}
            print(f"pooled vs {s} on {len(parts)} rows done ({time.time() - t0:.0f}s)", flush=True)

    # consistency of the per-row differences
    summary = {}
    for s in ["qwen", *specialists]:
        per = [e["comparisons"][s]["delta"][s] for e in report["rows"] if s in e["comparisons"]]
        if not per:
            continue
        summary[s] = {"rows": len(per)}
        for k in ("ece", "aece", "brier", "aurc", "eaurc"):
            vals = [d[k] for d in per]
            summary[s][k] = {"d3_lower": sum(v[0] < 0 for v in vals),
                             "ci_below_0": sum(v[2] is not None and v[2] < 0 for v in vals),
                             "ci_above_0": sum(v[1] is not None and v[1] > 0 for v in vals),
                             "median_delta": round(float(np.median([v[0] for v in vals])), 5)}
    report["summary"] = summary
    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report))
    print(json.dumps(summary, indent=1))
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
