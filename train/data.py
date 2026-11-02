"""Training examples from the EO training rows: targets, family mix and augmentation.

Rows come from $EOJEV_DATASETS/trainset/rows/final/<family>.jsonl. Replay is read from replay_d3.jsonl when
train/label_replay.py has written it. Image paths are relative to $EOJEV_DATASETS/trainset.

Targets are a probability vector over the question's options, in option order:
- **Replay rows:** d3-flash's own probabilities (meta.d3_probs), so its general decision behaviour is distilled back
  unchanged. Without them the gold option is used.
- **Counting rows** (meta.count, numeric options): SORD soft labels over the option values,
  t_i ∝ exp(-s (v_i - c)^2 / (c + 1)). An off-by-one at 10 costs less than at 1, and "more than K" stands for any
  count above K.
- **Score rows** (ordinal levels): SORD over level indices, t_i ∝ exp(-s (i - g)^2).
- **Everything else:** the gold option.

Ordinal questions also get the RPS term (train/train.py): score questions, and counting questions whose options
increase.

Augmentation: rows of families 1, 2 and 4 (scene labels, counting, damage) get a random rotation or flip, the
same one for every image of a row. It never changes their answer for nadir imagery. Family 9 asks about positions
and replay is not EO, so neither is augmented.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass

from bench.common import read_jsonl, transform_image
from trainset.common import ROWS, TRAIN
from train.d3model import Example, gold_index, load_images, n_options

FAMILIES = ("fam1_labels", "fam2_counting", "fam4_damage", "fam9_abstention", "replay",
            "fam10_multiselect", "fam11_geo", "fam12_presence_lr",
            "fam13_buildings", "fam14_count_distill", "fam15_rsvqa", "fam16_vrsbench", "fam17_cdvqa",
            "fam18_disasterm3", "fam19_nwpu", "fam20_dvl")
# Share of each family in the batches of the EO-Jev model (train/train.py --mix); replay is 20% of every batch.
DEFAULT_MIX = ("replay=0.20,fam15_rsvqa=0.11,fam18_disasterm3=0.115,fam1_labels=0.075,fam19_nwpu=0.07,"
               "fam16_vrsbench=0.06,fam2_counting=0.05,fam9_abstention=0.05,fam10_multiselect=0.05,fam17_cdvqa=0.05,"
               "fam20_dvl=0.04,fam4_damage=0.04,fam12_presence_lr=0.03,fam11_geo=0.025,fam13_buildings=0.02,"
               "fam14_count_distill=0.015")
# rotation / flip never changes these answers; not 9 (positions), 14 (d3-flash's answers on the unrotated image),
# 16 / 18 / 20 (VRSBench, DisasterM3 and DVL ask about positions and directions), replay
AUGMENT_FAMILIES = {"fam1", "fam2", "fam4", "fam10", "fam11", "fam12", "fam13", "fam15", "fam17", "fam19"}
TRANSFORMS = (None, "rot90", "rot180", "rot270", "hflip", "vflip")


def family_file(name: str):
    """rows/final/<name>.jsonl, or <name>_d3.jsonl once train/label_replay.py has added d3-flash's probabilities."""
    final = ROWS / "final"
    if (final / f"{name}_d3.jsonl").exists():
        return final / f"{name}_d3.jsonl"
    return final / f"{name}.jsonl"


def sord(distances: list[float], scale: float) -> list[float]:
    m = min(distances)
    w = [math.exp(-scale * (d - m)) for d in distances]
    total = sum(w)
    return [x / total for x in w]


def count_interval(option: str) -> tuple[float, float] | None:
    """The counts an option stands for: "7" -> [7, 7], "3 to 5" -> [3, 5], "more than 20" / "21 or more" -> [21, inf]."""
    m = re.fullmatch(r"(\d+)", option) or re.fullmatch(r"(\d+) to (\d+)", option)
    if m:
        return float(m.group(1)), float(m.groups()[-1])
    m = re.fullmatch(r"more than (\d+)", option)
    if m:
        return float(m.group(1)) + 1, math.inf
    m = re.fullmatch(r"(\d+) or more", option)
    return (float(m.group(1)), math.inf) if m else None


def count_values(keys: list[str], count: int) -> list[float] | None:
    """Each counting option's value for SORD: the true count clamped to the option's interval, so an option that
    contains the count is at distance 0. None if any option is not a count."""
    out = []
    for k in keys:
        iv = count_interval(k)
        if iv is None:
            return None
        out.append(min(max(float(count), iv[0]), iv[1]))
    return out


@dataclass
class Item:
    row: dict
    qid: str
    family: str
    target: list[float]
    ordinal: bool


def make_item(row: dict, qid: str, family: str, sord_count: float, sord_level: float) -> Item:
    q, gold, meta = row["questions"][qid], row["answer"][qid], row.get("meta", {})
    n = n_options(q)
    d3 = (meta.get("d3_probs") or {}).get(qid)
    if d3 is not None and len(d3) == n:
        return Item(row, qid, family, list(d3), False)
    if q["type"] == "score":
        g = int(gold)
        return Item(row, qid, family, sord([(i - g) ** 2 for i in range(n)], sord_level), True)
    if q["type"] == "choice" and isinstance(meta.get("count"), int):
        values = count_values(list(q["criteria"]), meta["count"])
        if values is not None:
            c = meta["count"]
            target = sord([(v - c) ** 2 / (c + 1) for v in values], sord_count)
            return Item(row, qid, family, target, all(a < b for a, b in zip(values, values[1:])))
    target = [0.0] * n
    target[gold_index(q, gold)] = 1.0
    return Item(row, qid, family, target, False)


def load_items(split: str, families=FAMILIES, limit_per_family: int | None = None, seed: int = 0,
               sord_count: float = 4.0, sord_level: float = 2.0) -> dict[str, list[Item]]:
    """Items per family for one split. limit_per_family keeps a fixed random subset of rows (the quick first model).
    A family without a final rows file is skipped."""
    out = {}
    for fam in families:
        if not family_file(fam).exists():
            continue
        rows = [r for r in read_jsonl(family_file(fam)) if r["split"] == split]
        random.Random(f"{seed}:{fam}").shuffle(rows)
        if limit_per_family:
            rows = rows[:limit_per_family]
        out[fam] = [make_item(r, qid, fam, sord_count, sord_level) for r in rows for qid in r["questions"]]
    return out


def to_example(item: Item, rng: random.Random | None = None) -> Example:
    """Load the item's images (with a random rotation / flip when rng is given and the family allows it)."""
    row = item.row
    images = load_images([TRAIN / p for p in row["images"]])
    if rng is not None and row.get("meta", {}).get("family") in AUGMENT_FAMILIES:
        t = rng.choice(TRANSFORMS)
        images = [transform_image(im, t) for im in images]
    return Example(key=f"{row['id']}/{item.qid}", state=row["state"], question=row["questions"][item.qid],
                   images=images, target=item.target, ordinal=item.ordinal, group=item.family)


class Mixer:
    """Endless weighted sampling over families; each family is walked in a fresh shuffle every pass."""

    def __init__(self, items: dict[str, list[Item]], weights: dict[str, float], seed: int = 0):
        self.items = {k: v for k, v in items.items() if v and weights.get(k, 0) > 0}
        self.names = sorted(self.items)
        self.weights = [weights[k] for k in self.names]
        self.rng = random.Random(seed)
        self.order = {k: [] for k in self.names}
        self.passes = {k: 0 for k in self.names}

    def _next(self, fam: str) -> Item:
        if not self.order[fam]:
            idx = list(range(len(self.items[fam])))
            random.Random(f"{fam}:{self.passes[fam]}:{self.rng.random()}").shuffle(idx)
            self.order[fam], self.passes[fam] = idx, self.passes[fam] + 1
        return self.items[fam][self.order[fam].pop()]

    def sample(self, n: int) -> list[Item]:
        return [self._next(f) for f in self.rng.choices(self.names, self.weights, k=n)]

    def state(self) -> dict:
        return {"rng": self.rng.getstate(), "order": self.order, "passes": self.passes}

    def load_state(self, state: dict) -> None:  # saved with torch.save (pickle), so the RNG tuple comes back as is
        self.rng.setstate(state["rng"])
        self.order, self.passes = state["order"], state["passes"]


def default_weights(items: dict[str, list[Item]], replay_share: float = 0.2) -> dict[str, float]:
    """Task families in proportion to their size, replay a fixed share."""
    task = {k: len(v) for k, v in items.items() if k != "replay"}
    total = sum(task.values()) or 1
    weights = {k: (1 - replay_share) * n / total for k, n in task.items()}
    if items.get("replay"):
        weights["replay"] = replay_share
    return weights
