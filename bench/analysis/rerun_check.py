"""A re-run of a model against its stored run on the same rows, question by question: does this machine reproduce
the stored outputs (e.g. after moving the project to new hardware)?

For each pair of run directories, over the questions both runs answered (rows with status "ok"):
- top-1 agreement: the same most probable option (a noul answer counts as the two options no / yes);
- max|dp|: the largest absolute difference between the two runs' option probabilities, as median, 95th percentile
  and maximum;
- the share of questions whose probabilities are identical.

Pass, per pair, as gate R0 (train/parity.py): top-1 agreement >= 99.9% and median max|dp| <= 1e-3, over at least 99%
of the stored run's questions.

    python -m bench.analysis.rerun_check --out <report.json> <stored run dir>:<new run dir> ...
Exits 1 if any pair fails.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

from bench.common import read_jsonl

MIN_AGREE, MAX_MEDIAN_DP, MIN_COVERAGE = 0.999, 1e-3, 0.99


def vector(answer: dict) -> dict[str, float] | None:
    if "probabilities" in answer:
        return answer["probabilities"]
    if answer.get("type") == "noul":
        return {"no": 1 - answer["noul"], "yes": answer["noul"]}
    return None


def answers(run_dir: str) -> dict[str, dict[str, float]]:
    """Option probabilities per '<row id>/<question id>' from a run's ok rows (a retry/ record wins)."""
    out = {}
    for f in sorted(glob.glob(f"{run_dir}/chunk*.jsonl")) + sorted(glob.glob(f"{run_dir}/retry/chunk*.jsonl")):
        for r in read_jsonl(f):
            if r.get("status") != "ok":
                continue
            for qid, a in r["answers"].items():
                if (v := vector(a)) is not None:
                    out[f"{r['id']}/{qid}"] = v
    return out


def compare(old: dict, new: dict) -> dict:
    keys = [k for k in old if k in new and old[k].keys() == new[k].keys()]
    res = {"stored_questions": len(old), "new_questions": len(new), "compared": len(keys)}
    if not keys:
        return {**res, "pass": False}
    dp = sorted(max(abs(old[k][o] - new[k][o]) for o in old[k]) for k in keys)
    agree = sum(max(old[k], key=old[k].get) == max(new[k], key=new[k].get) for k in keys)
    res.update({"top1_agreement": round(agree / len(keys), 5), "identical": round(dp.count(0.0) / len(keys), 5),
                "median_max_abs_dp": statistics.median(dp), "p95_max_abs_dp": dp[int(0.95 * (len(dp) - 1))],
                "max_max_abs_dp": dp[-1]})
    res["pass"] = (res["top1_agreement"] >= MIN_AGREE and res["median_max_abs_dp"] <= MAX_MEDIAN_DP
                   and len(keys) >= MIN_COVERAGE * len(old))
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pairs", nargs="+", help="<stored run dir>:<new run dir>")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    report = {}
    for pair in args.pairs:
        old, new = pair.split(":")
        report[Path(new).name] = {"stored": old, "new": new, **compare(answers(old), answers(new))}
        print(Path(new).name, json.dumps({k: v for k, v in report[Path(new).name].items() if k not in ("stored", "new")}),
              flush=True)
    ok = all(v["pass"] for v in report.values())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"pairs": report, "pass": ok}, indent=1))
    print("RERUN CHECK", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
