"""Phase-0 parity: compare our d3-flash results with the official Decision Index run, request by request.

Rows are joined on run_id and kept only when the payload hashes match (so both sides scored the same input).
For every question we compare the full option-probability vectors.

    python bench/parity_compare.py <official results.jsonl.gz> <our results.jsonl> <out.json>

Pass criteria (docs/03_benchmark_plan.md, Phase 0): argmax agreement >= 99.5% and median max|dp| <= 0.005.
"""

from __future__ import annotations

import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def read(path: str) -> dict[str, dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        rows = (json.loads(line) for line in f)
        return {r["run_id"]: r for r in rows if r.get("status") == "ok"}


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> None:
    official, ours, out = read(sys.argv[1]), read(sys.argv[2]), Path(sys.argv[3])
    per_dataset: dict[str, dict[str, list]] = defaultdict(lambda: {"dp": [], "agree": []})
    hash_mismatch = 0
    for run_id, mine in ours.items():
        ref = official.get(run_id)
        if ref is None:
            continue
        if ref.get("payload_sha256") != mine.get("payload_sha256"):
            hash_mismatch += 1
            continue
        for qid, answer in mine["response"]["answers"].items():
            ref_probs = ref["response"]["answers"][qid]["probabilities"]
            probs = answer["probabilities"]
            per_dataset[ref["dataset"]]["dp"].append(max(abs(probs[k] - ref_probs[k]) for k in ref_probs))
            per_dataset[ref["dataset"]]["agree"].append(
                max(probs, key=probs.get) == max(ref_probs, key=ref_probs.get)
            )

    def summarize(dp: list[float], agree: list[bool]) -> dict:
        return {
            "questions": len(dp),
            "argmax_agreement": round(sum(agree) / len(agree), 5),
            "median_max_abs_dp": round(statistics.median(dp), 6),
            "p95_max_abs_dp": round(quantile(dp, 0.95), 6),
            "max_max_abs_dp": round(max(dp), 6),
        }

    all_dp = [x for d in per_dataset.values() for x in d["dp"]]
    all_agree = [x for d in per_dataset.values() for x in d["agree"]]
    overall = summarize(all_dp, all_agree)
    report = {
        "our_requests": len(ours),
        "matched_requests": sum(1 for r in ours if r in official) - hash_mismatch,
        "payload_hash_mismatches": hash_mismatch,
        "overall": overall,
        "by_dataset": {name: summarize(d["dp"], d["agree"]) for name, d in sorted(per_dataset.items())},
        "pass": overall["argmax_agreement"] >= 0.995 and overall["median_max_abs_dp"] <= 0.005,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
