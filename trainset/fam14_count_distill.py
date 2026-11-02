"""Family 14, counting objects we have no labels for: ~2,000 questions whose targets are d3-flash's own
answer probabilities (train/label_replay.py --family fam14_count_distill), as for replay.

Why: fine-tuning overwrote d3-flash's counting of unlabelled objects. EarthMind "paths" fell from 42.4 (d3-flash) to
22.5 (fine-tuned without this family): the counting family's answers were 36% "1", and the model fell back to small counts. Without labels
this family cannot teach these counts; it keeps d3-flash's behaviour on them while the other families change.

- Images: the AID training images that family 1 already stores (its non-test images), so no new image is added.
- Objects: nouns no family labels (paths, roads, trees, fields, ponds), each asked on AID scene classes where it is
  plausible (NOUNS_BY_CLASS), with family 2's counting templates.
- Options: 50% a run of 4-6 consecutive counts starting at 0-12, 30% the full list 0..K (K 10, 15 or 20), 20% count
  ranges (family 2's schemes), all independent of the unknown count. The stored answer is a placeholder until the
  labelling job replaces it with d3-flash's top option (meta.answer_from_d3).

    MODULE=trainset.fam14_count_distill sbatch -J eojev-fam14 slurm/convert_cpu.sbatch
"""

from __future__ import annotations

import argparse
import collections
import random

from bench.common import read_jsonl, write_jsonl
from trainset.common import ROWS, make_row, render
from trainset.fam2_9_objects import CNT_T, RANGE_SCHEMES

FAMILY = "fam14"
TARGET = 2000
NOUNS_BY_CLASS = {
    "paths": {"park", "resort", "square", "school", "playground", "farmland", "medium residential", "sparse residential"},
    "roads": {"dense residential", "medium residential", "sparse residential", "commercial", "center", "industrial",
              "farmland", "viaduct", "railway station", "church"},
    "trees": {"park", "resort", "medium residential", "sparse residential", "meadow", "school", "church", "pond"},
    "fields": {"farmland", "meadow", "sparse residential", "bare land", "playground", "baseball field"},
    "ponds": {"pond", "park", "farmland", "resort", "meadow", "sparse residential"},
}


def options(rng) -> tuple[list[str], str]:
    u = rng.random()
    if u < 0.5:
        start, m = rng.randint(0, 12), rng.choice([4, 5, 6])
        return [str(v) for v in range(start, start + m)], "run"
    if u < 0.8:
        k = rng.choice([10, 15, 20])
        return [str(i) for i in range(k + 1)] + [f"more than {k}"], "full"
    lo, *rest = rng.choice(RANGE_SCHEMES)
    bins, start = [], lo
    for edge in rest:
        bins.append(str(start) if start == edge else f"{start} to {edge}")
        start = edge + 1
    return bins + [f"more than {start - 1}"], "ranges"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0)
    args = p.parse_args()
    n_target = args.sample or TARGET
    rng = random.Random("fam14")
    aid = [r for r in read_jsonl(ROWS / "fam1_labels.jsonl") if r["meta"]["source"] == "AID"]
    by_noun = {noun: [r for r in aid if r["meta"]["gold_label"] in classes] for noun, classes in NOUNS_BY_CLASS.items()}
    rows, used = [], set()
    per_noun = n_target // len(by_noun)
    for noun, pool in sorted(by_noun.items()):
        rng.shuffle(pool)
        for src in pool[:per_noun]:
            opts, form = options(rng)
            tid, text = rng.choice(CNT_T)
            rid = f"tr:{FAMILY}:AID:{src['id'].rsplit(':', 1)[-1]}:{noun}"
            if rid in used:
                continue
            used.add(rid)
            rows.append(make_row(
                rid=rid, group_id=src["group_id"], family=FAMILY, source="AID", task_family="count_distill",
                images=src["images"], state="",
                questions={"q": {"type": "choice", "instructions": text.format(p=noun),
                                 "criteria": {o: None for o in opts}}},
                answer={"q": opts[0]},  # placeholder: train/label_replay.py sets d3-flash's top option
                meta={"template": tid, "noun": noun, "scene_class": src["meta"]["gold_label"], "options_form": form,
                      "answer_from_d3": True, "image_sha256": src["meta"]["image_sha256"],
                      "orig_px": src["meta"]["orig_px"], "licence": "AID: research use (no licence stated)"}))
    suffix = ".sample" if args.sample else ""
    write_jsonl(ROWS / f"fam14_count_distill{suffix}.jsonl", rows)
    with open(ROWS / f"fam14_count_distill{suffix}.txt", "w") as fh:
        for r in (rows if args.sample else rng.sample(rows, min(30, len(rows)))):
            fh.write(render(r) + "\n\n")
    print(f"fam14_count_distill: {len(rows)} rows; nouns {dict(collections.Counter(r['meta']['noun'] for r in rows))}; "
          f"forms {dict(collections.Counter(r['meta']['options_form'] for r in rows))}; "
          f"splits {dict(collections.Counter(r['split'] for r in rows))}")


if __name__ == "__main__":
    main()
