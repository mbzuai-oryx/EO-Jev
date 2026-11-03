"""Derive robustness / ablation row files from a converted benchmark.

    python -m bench.variants sample  IN OUT --n 1500 [--by task_family]   # stratified, seeded subset
    python -m bench.variants permute IN OUT --k 3                         # k seeded option orders per row (flip rate)
    python -m bench.variants options IN OUT --k 8                         # choice -> GT + (k-1) seeded distractors (A5)
    python -m bench.variants state   IN OUT                               # metadata into the state (A2)

Variant rows keep the original group_id, so per-item consistency (all permutations right, flip rate) can be
scored by grouping; ids get a suffix. Image transforms (rotations, flips, downscaling) are runner flags instead
(bench/runners/d3.py --image-transform).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from bench.common import read_jsonl, validate_row, write_jsonl


def sample(rows: list[dict], n: int, by: str) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(by) if by in r else r["meta"].get(by)].append(r)
    rng = random.Random(0)
    share = n / len(rows)
    out = []
    for key in sorted(groups, key=str):
        g = groups[key]
        out += rng.sample(g, min(len(g), max(1, round(share * len(g)))))
    return out


def permute(rows: list[dict], k: int) -> list[dict]:
    out = []
    for r in rows:
        for i in range(k):
            v = json.loads(json.dumps(r))
            for qid, q in v["questions"].items():
                if q["type"] == "choice":
                    keys = list(q["criteria"])
                    random.Random(f"perm:{r['id']}:{qid}:{i}").shuffle(keys)
                    q["criteria"] = {key: q["criteria"][key] for key in keys}
            v["id"] = f"{r['id']}:perm{i}"
            v["group_id"] = r["id"]  # group the permutations of one item
            v["meta"] = {**v["meta"], "permutation": i, "source_group": r["group_id"]}
            out.append(v)
    return out


def options(rows: list[dict], k: int) -> list[dict]:
    out = []
    for r in rows:
        v = json.loads(json.dumps(r))
        keep = True
        for qid, q in v["questions"].items():
            if q["type"] != "choice":
                continue
            keys = list(q["criteria"])
            if len(keys) <= k:
                continue
            gold = r["answer"][qid]
            rng = random.Random(f"opts{k}:{r['id']}:{qid}")
            chosen = [gold] + rng.sample([o for o in keys if o != gold], k - 1)
            rng.shuffle(chosen)
            q["criteria"] = {o: q["criteria"][o] for o in chosen}
        if keep:
            v["id"] = f"{r['id']}:{k}way"
            v["meta"] = {**v["meta"], "n_options": k}
            out.append(v)
    return out


def with_state(rows: list[dict]) -> list[dict]:
    """P1: put what is known about the image (sensor, ground sampling distance, place, date) into the state."""
    out = []
    for r in rows:
        m = r.get("meta", {})
        facts = {k: m[k] for k in ("sensor", "gsd_m", "country", "date") if m.get(k) not in (None, "")}
        if "gsd_m" in facts:
            facts["ground_sampling_distance"] = f"{facts.pop('gsd_m')} m per pixel"
        v = json.loads(json.dumps(r))
        if facts:
            prefix = r["state"] + "\n" if r["state"] else ""
            v["state"] = prefix + "Image metadata: " + json.dumps(facts, ensure_ascii=False)
        v["id"] = f"{r['id']}:state"
        out.append(v)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("op", choices=["sample", "permute", "options", "state"])
    p.add_argument("inp")
    p.add_argument("out")
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--by", default="task_family")
    args = p.parse_args()
    rows = list(read_jsonl(args.inp))
    result = {"sample": lambda: sample(rows, args.n, args.by), "permute": lambda: permute(rows, args.k),
              "options": lambda: options(rows, args.k), "state": lambda: with_state(rows)}[args.op]()
    for r in result:
        validate_row(r)
    print(f"{args.op}: {len(rows)} -> {write_jsonl(args.out, result)} rows -> {args.out}")


if __name__ == "__main__":
    main()
