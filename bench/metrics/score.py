"""Score one results file against its rows (docs/03 §7).

choice: accuracy, macro-F1, balanced accuracy, top-3; noul: AUROC, accuracy@0.5, balanced accuracy;
score: quadratic-weighted kappa (argmax level), MAE of the expected level, exact accuracy.
Every type: ECE (15 equal-width bins, top-label confidence), Brier, NLL. 95% CIs by bootstrap over group_id.

    python -m bench.metrics.score --rows R.jsonl --results RES.jsonl [--out OUT.json] [--by task_family]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.common import read_jsonl

EPS = 1e-12


def question_records(rows: dict[str, dict], results: list[dict]) -> tuple[list[dict], int]:
    """One record per scored question: type, correctness, confidence, label probability, etc."""
    recs, errors = [], 0
    for res in results:
        row = rows.get(res["id"])
        if row is None:
            continue
        if res.get("status") != "ok":
            errors += len(row["questions"])
            continue
        for qid, q in row["questions"].items():
            ans = res["answers"].get(qid, {})
            if "error" in ans:
                errors += 1
                continue
            gold = row["answer"][qid]
            base = {"id": row["id"], "qid": qid, "group": row["group_id"], "benchmark": row["benchmark"],
                    "task_family": row["task_family"], "type": q["type"], "meta": row.get("meta", {})}
            if q["type"] == "choice":
                probs = ans["probabilities"]
                keys = list(probs)
                p = np.array([probs[k] for k in keys], dtype=float)
                order = np.argsort(-p)
                pred = keys[order[0]]
                y = np.array([k == gold for k in keys], dtype=float)
                recs.append({**base, "gold": gold, "pred": pred, "correct": pred == gold,
                             "top3": gold in {keys[i] for i in order[:3]}, "conf": float(p[order[0]]),
                             "p_gold": float(probs[gold]), "brier": float(((p - y) ** 2).sum())})
            elif q["type"] == "noul":
                p_yes = float(ans["noul"])
                label = bool(gold)
                pred = p_yes >= 0.5
                p_gold = p_yes if label else 1 - p_yes
                recs.append({**base, "gold": label, "pred": pred, "correct": pred == label, "p_yes": p_yes,
                             "conf": max(p_yes, 1 - p_yes), "p_gold": p_gold,
                             "brier": 2 * (p_yes - float(label)) ** 2})  # two-class Brier, same scale as choice
            else:  # score
                probs = ans["probabilities"]
                levels = sorted(probs, key=int)
                p = np.array([probs[k] for k in levels], dtype=float)
                pred = int(levels[int(np.argmax(p))])
                y = np.zeros(len(levels))
                y[int(gold)] = 1.0
                recs.append({**base, "gold": int(gold), "pred": pred, "correct": pred == int(gold),
                             "expected": float(ans["score"]), "conf": float(p.max()), "p_gold": float(p[int(gold)]),
                             "brier": float(((p - y) ** 2).sum())})
    return recs, errors


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = np.empty(len(scores))
    order = np.argsort(scores, kind="mergesort")
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties
    for v in np.unique(scores):
        m = scores == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def macro_f1(gold: list, pred: list) -> tuple[float, float]:
    classes = sorted(set(gold) | set(pred), key=str)
    f1s, recalls = [], []
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        if tp + fn:
            recalls.append(rec)
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)), float(np.mean(recalls))


def qwk(gold: list[int], pred: list[int]) -> float:
    k = max(max(gold), max(pred)) + 1
    o = np.zeros((k, k))
    for g, p in zip(gold, pred):
        o[g, p] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / max(1, (k - 1) ** 2)
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    return float(1 - (w * o).sum() / max(EPS, (w * e).sum()))


def metrics(recs: list[dict]) -> dict:
    out = {"n": len(recs)}
    if not recs:
        return out
    correct = np.array([r["correct"] for r in recs], dtype=float)
    conf = np.array([r["conf"] for r in recs])
    out.update(accuracy=float(correct.mean()), ece=ece(conf, correct),
               brier=float(np.mean([r["brier"] for r in recs])),
               nll=float(np.mean([-math.log(max(EPS, r["p_gold"])) for r in recs])),
               mean_conf=float(conf.mean()))
    types = {r["type"] for r in recs}
    if types == {"choice"}:
        out["macro_f1"], out["balanced_acc"] = macro_f1([r["gold"] for r in recs], [r["pred"] for r in recs])
        out["top3"] = float(np.mean([r["top3"] for r in recs]))
    elif types == {"noul"}:
        labels = np.array([r["gold"] for r in recs], dtype=bool)
        out["auroc"] = auroc(np.array([r["p_yes"] for r in recs]), labels)
        out["balanced_acc"] = macro_f1([r["gold"] for r in recs], [r["pred"] for r in recs])[1]
        out["pos_rate"] = float(labels.mean())
    elif types == {"score"}:
        gold, pred = [r["gold"] for r in recs], [r["pred"] for r in recs]
        out["qwk"] = qwk(gold, pred)
        out["mae_expected"] = float(np.mean([abs(r["expected"] - r["gold"]) for r in recs]))
    return out


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    hits = labels[order].astype(float)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def multilabel(recs: list[dict], min_pos: int = 10) -> dict:
    """Rows asking one `noul` per class (e.g. BigEarthNet): per-class AP, macro mAP, micro / sample F1."""
    per_class = defaultdict(list)
    per_row = defaultdict(list)
    for r in recs:
        per_class[r["qid"]].append(r)
        per_row[r["id"]].append(r)
    ap = {}
    for qid, rs in sorted(per_class.items()):
        labels = np.array([r["gold"] for r in rs], dtype=bool)
        ap[qid] = {"ap": average_precision(np.array([r["p_yes"] for r in rs]), labels), "positives": int(labels.sum())}
    tp = sum(r["gold"] and r["pred"] for r in recs)
    fp = sum((not r["gold"]) and r["pred"] for r in recs)
    fn = sum(r["gold"] and not r["pred"] for r in recs)
    sample_f1 = []
    for rs in per_row.values():
        t = sum(r["gold"] and r["pred"] for r in rs)
        denom = sum(r["gold"] for r in rs) + sum(r["pred"] for r in rs)
        sample_f1.append(2 * t / denom if denom else 1.0)
    valid = [v["ap"] for v in ap.values() if not math.isnan(v["ap"])]
    common = [v["ap"] for v in ap.values() if v["positives"] >= min_pos]
    exact = float(np.mean([all(r["correct"] for r in rs) for rs in per_row.values()]))
    return {"macro_map": float(np.mean(valid)) if valid else float("nan"),
            f"macro_map_classes_ge{min_pos}pos": float(np.mean(common)) if common else float("nan"),
            "micro_f1": 2 * tp / max(1, 2 * tp + fp + fn), "sample_f1": float(np.mean(sample_f1)),
            "exact_set_match": exact, "per_class": ap}


def bootstrap(recs: list[dict], keys=("accuracy",), n: int = 1000, seed: int = 0) -> dict:
    groups = defaultdict(list)
    for r in recs:
        groups[r["group"]].append(r)
    names = list(groups)
    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    for _ in range(n):
        pick = rng.integers(0, len(names), len(names))
        m = metrics([r for i in pick for r in groups[names[i]]])
        for k in keys:
            if k in m and not (isinstance(m[k], float) and math.isnan(m[k])):
                samples[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in samples.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", required=True)
    p.add_argument("--results", required=True, nargs="+")
    p.add_argument("--out")
    p.add_argument("--by", default="benchmark", help="row field to slice by (benchmark, task_family, ...)")
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()

    rows = {r["id"]: r for r in read_jsonl(args.rows)}
    # One result per row: a later successful record (e.g. from a retry pass) replaces an earlier error.
    latest: dict[str, dict] = {}
    for r in (r for f in args.results for r in read_jsonl(f)):
        if r["id"] not in latest or r.get("status") == "ok":
            latest[r["id"]] = r
    results = list(latest.values())
    recs, errors = question_records(rows, results)
    report = {"rows": len(rows), "results": len(results), "question_errors": errors, "slices": {}}
    slices = defaultdict(list)
    for r in recs:
        slices[str(r.get(args.by) if args.by in r else r["meta"].get(args.by))].append(r)
    for name, rs in sorted(slices.items()):
        m = metrics(rs)
        ci_keys = [k for k in ("accuracy", "auroc", "qwk", "ece") if k in m]
        m["ci95"] = bootstrap(rs, ci_keys, n=args.bootstrap) if args.bootstrap else {}
        if {r["type"] for r in rs} == {"noul"} and len({r["qid"] for r in rs}) > 1 and args.by != "qid":
            m["multilabel"] = multilabel(rs)
        for key in ("rotation", "permutation"):  # CircularEval (LHRS-Bench) / option-permutation variants
            if all(key in r["meta"] for r in rs):
                groups = defaultdict(list)
                for r in rs:
                    groups[r["group"]].append(r)
                m["consistency"] = {
                    "variant": key,
                    "items": len(groups),
                    "all_variants_correct": float(np.mean([all(r["correct"] for r in g) for g in groups.values()])),
                    "first_variant": float(np.mean([r["correct"] for r in rs if r["meta"][key] == 0])),
                    "answer_flip_rate": float(np.mean([len({str(r["pred"]) for r in g}) > 1 for g in groups.values()])),
                }
        report["slices"][name] = m
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
