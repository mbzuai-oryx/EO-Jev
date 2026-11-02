"""General replay pool: ~15k rows, mixed into ~20% of every training batch so that
fine-tuning keeps d3's general decision skills.

- 10,000 text-only decision rows from jev-distill-corpus-v3 (SargeDev @75111a13, Apache-2.0; JEV-9B's distillation
  corpus). The placeholder stream yuri_v1 (uniform [0.5, 0.5] targets) is dropped. Its eval files (test, ood,
  calibration, test_set_30k) are never read.
- 5,000 natural-image four-way questions from A-OKVQA train (Apache-2.0; COCO images). A-OKVQA is not in our
  retention panel; its images still get hashed against the panel's once the panel is fetched.

The corpus's own target distribution is kept in meta.target_probs (options in order). The plan's replay target is
d3-flash's own probabilities (self-distillation): a GPU job fills meta.d3_probs later; until then the trainer may
use target_probs. The stored answer is the argmax, so the row validates.

    MODULE=trainset.replay [ARGS="--sample 10"] sbatch -J eojev-replay slurm/convert_cpu.sbatch
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

from bench.common import write_jsonl
from trainset.common import ROWS, TRAIN, make_row, render, store

MAN = json.loads((TRAIN / "downloads.json").read_text())
TARGET = {"jev": 10000, "aokvqa": 5000}


def jev_rows(rng, n) -> list[dict]:
    path = Path(MAN["jev"]["path"]) / "train.jsonl"
    keep = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("source") == "yuri_v1":
                continue
            keep.append(line)
    picked = [json.loads(l) for l in rng.sample(keep, min(n, len(keep)))]
    rows = []
    for r in picked:
        kind, opts, tgt = r["kind"], [str(o) for o in r["options"]], r["target"]
        best = max(range(len(tgt)), key=lambda i: tgt[i])
        if kind == "noul":
            yes = next((i for i, o in enumerate(opts) if o.lower() in ("true", "yes")), 1)
            q = {"type": "noul", "instructions": r["question"], "criteria": None}
            ans = tgt[yes] >= 0.5
        elif kind == "score":
            q = {"type": "score", "instructions": r["question"], "criteria": opts}
            ans = best
        else:
            q = {"type": "choice", "instructions": r["question"], "criteria": {o: None for o in opts}}
            ans = opts[best]
        rows.append(make_row(rid=f"tr:replay:jev:{r['id']}", group_id=f"jev:{r['id']}", family="replay", source="jev",
                             task_family=f"text_{r.get('family', 'general')}", images=[], state=r.get("state") or "",
                             questions={"q": q}, answer={"q": ans},
                             meta={"target_probs": tgt, "options": opts, "domain": r.get("domain"),
                                   "stream": r.get("source"), "licence": "Apache-2.0 (openjev_v2 also CC0)"}))
    return rows


def aokvqa_rows(rng, n) -> list[dict]:
    files = sorted((Path(MAN["aokvqa"]["path"]) / "data").glob("train-*.parquet"))
    idx = [(f, i) for f in files for i in range(pq.ParquetFile(f).metadata.num_rows)]
    picked = sorted(rng.sample(idx, min(n, len(idx))), key=lambda x: (str(x[0]), x[1]))
    rows = []
    for f in files:
        want = [i for ff, i in picked if ff == f]
        if not want:
            continue
        t = pq.read_table(f).take(want).to_pylist()
        for r in t:
            choices = list(r["choices"])
            gold = choices[r["correct_choice_idx"]]
            order = choices[:]
            rng.shuffle(order)
            rel, size, sha = store(r["image"]["bytes"], f"aokvqa/{r['question_id']}")
            rows.append(make_row(rid=f"tr:replay:aokvqa:{r['question_id']}", group_id=f"aokvqa:{r['question_id']}",
                                 family="replay", source="A-OKVQA", task_family="general_vqa", images=[rel], state="",
                                 questions={"q": {"type": "choice", "instructions": r["question"],
                                                  "criteria": {o: None for o in order}}},
                                 answer={"q": gold},
                                 meta={"image_sha256": sha, "orig_px": size, "licence": "Apache-2.0 (COCO images)"}))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0)
    args = p.parse_args()
    rows = jev_rows(random.Random("replay:jev"), args.sample or TARGET["jev"])
    rows += aokvqa_rows(random.Random("replay:aokvqa"), args.sample or TARGET["aokvqa"])
    suffix = ".sample" if args.sample else ""
    write_jsonl(ROWS / f"replay{suffix}.jsonl", rows)
    with open(ROWS / f"replay{suffix}.txt", "w") as f:
        for r in (rows if args.sample else random.Random(0).sample(rows, 30)):
            f.write(render(r) + "\n\n")
    print(f"replay: {len(rows)} rows; {dict(collections.Counter(r['meta']['source'] for r in rows))}; "
          f"types {dict(collections.Counter(q['type'] for r in rows for q in r['questions'].values()))}")


if __name__ == "__main__":
    main()
