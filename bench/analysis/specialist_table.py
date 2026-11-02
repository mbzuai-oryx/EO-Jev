"""Specialist results per website table row (docs/05_specialist_models.md): for every row of the benchmark-guide
tables, each assigned specialist's score in both answer modes, from runs/specialists/<model>/<mode>/<rows>/score_*.json
(scripts/score_specialists.sh). Exact-set match for multi-select items is computed from the raw results, as for d3.

    python -m bench.analysis.specialist_table [--json out.json]
(prints, per model: option probabilities / written / option probabilities, own format / written, own format)
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from bench.analysis.calibration_latency import _load
from bench.common import CONVERTED, RUNS, is_sar

SPEC = RUNS / "specialists"
MODES = ("logprob", "generate", "logprob-native", "generate-native")  # -native: the model's own prompt format


def _slices(model: str, mode: str, run: str, by: str) -> dict | None:
    p = SPEC / model / mode / run / f"score_{by}.json"
    return json.load(open(p))["slices"] if p.exists() else None


def _pooled(sl: dict, keep=lambda k: True) -> float | None:
    ks = [k for k in sl if keep(k)]
    n = sum(sl[k]["n"] for k in ks)
    return 100 * sum(sl[k]["accuracy"] * sl[k]["n"] for k in ks) / n if n else None


def accuracy(model, mode, run, by="benchmark", key=None, keep=None):
    sl = _slices(model, mode, run, by)
    if sl is None:
        return None
    if key is not None:
        return 100 * sl[key]["accuracy"] if key in sl else None
    return _pooled(sl, keep or (lambda k: True))


def field(model, mode, run, key, path, scale=1.0, by="benchmark"):
    sl = _slices(model, mode, run, by)
    if sl is None or key not in sl:
        return None
    v = sl[key]
    for p in path:
        if v is None:
            return None
        v = v.get(p) if isinstance(p, str) else v[p]
    return None if v is None else scale * v


def lhrs_strict(model, mode):
    sl = _slices(model, mode, "lhrs_bench_circular", "benchmark")
    if sl is None:
        return None
    c = sl["LHRS-Bench"].get("consistency") or sl["LHRS-Bench"].get("circular") or {}
    v = c.get("all_variants_correct", c.get("strict_all_rotations"))
    return None if v is None else 100 * v


def raw_accuracy(model, mode, run, keep=lambda r: True):
    """Question accuracy from the raw results without SAR rows (calibration_latency._load), for runs that hold SAR
    rows: their score_*.json slices include them."""
    if len(glob.glob(str(SPEC / model / mode / run / "chunk*.done"))) < 64:
        return None  # still running: as scripts/score_specialists.sh, never report a partial run
    recs = [r for r in _load(model, run, mode)[0] if keep(r)]
    return 100 * sum(r["correct"] for r in recs) / len(recs) if recs else None


def exact_set(model, mode, run, rows_rel):
    """Share of multi-select items (task_family ending in _multi) where every per-option yes/no is right; SAR items
    are left out."""
    files = glob.glob(str(SPEC / model / mode / run / "chunk*.jsonl"))
    if not files or len(glob.glob(str(SPEC / model / mode / run / "chunk*.done"))) < 64:
        return None  # still running: as scripts/score_specialists.sh, never report a partial run
    rows = {}
    for line in open(CONVERTED / rows_rel):
        r = json.loads(line)
        if r["task_family"].endswith("_multi") and not is_sar(r):
            rows[r["id"]] = r
    latest = {}
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r["id"] in rows and (r["id"] not in latest or r["status"] == "ok"):
                latest[r["id"]] = r
    tot = ok = 0
    for i, r in latest.items():
        if r["status"] != "ok":
            continue
        row = rows[i]
        tot += 1
        ok += all((r["answers"][q]["noul"] >= 0.5) == row["answer"][q] for q in row["questions"])
    return 100 * ok / tot if tot else None


def _scene(name):
    return lambda m, mode: accuracy(m, mode, "scene_all", key=name)


ROWS = [  # (website row, metric label, models, getter(model, mode))
    ("UC Merced", "Accuracy (%)", ["rsthinker", "geozero"], _scene("UCM")),
    ("AID", "Accuracy (%)", ["rsthinker", "geozero"], _scene("AID")),
    ("WHU-RS19", "Accuracy (%)", ["rsthinker", "geozero"], _scene("WHU-RS19")),
    ("SIRI-WHU", "Accuracy (%)", ["rsthinker", "geozero"], _scene("SIRI-WHU")),
    ("EuroSAT (RGB)", "Accuracy (%)", ["rsthinker", "geozero"], _scene("EuroSAT-RGB")),
    ("NWPU-RESISC45", "Accuracy (%)", ["rsthinker", "geozero"], _scene("NWPU-RESISC45")),
    ("BigEarthNet v2", "Micro-F1 (0–1)", ["earthmind"],
     lambda m, mode: field(m, mode, "bigearthnet_v2_test_1k", "BigEarthNet-v2", ["multilabel", "micro_f1"])),
    ("GEOBench-VLM", "Accuracy (%)", ["geozero"], lambda m, mode: accuracy(m, mode, "geobench_vlm_single")),
    ("CHOICE", "Accuracy (%)", ["geozero"], lambda m, mode: raw_accuracy(m, mode, "choice_mc")),
    ("LHRS-Bench", "Strict accuracy (%)", ["scorers-r1"], lhrs_strict),
    ("VRSBench (converted)", "Accuracy (%)", ["rsthinker"], lambda m, mode: accuracy(m, mode, "vrsbench_vqa_mc_5k")),
    ("RSVQA-LR", "Accuracy (%)", ["earthdial"], lambda m, mode: accuracy(m, mode, "rsvqa_lr")),
    ("RSVQA-HR (test 2)", "Accuracy (%)", ["geozero"], lambda m, mode: accuracy(m, mode, "rsvqa_hr_test2_5k")),
    ("CDVQA (test 2)", "Accuracy (%)", ["geochrono"], lambda m, mode: accuracy(m, mode, "cdvqa_test2_5k")),
    ("DVL-Bench, single-choice", "Accuracy (%)", ["geochrono"],
     lambda m, mode: accuracy(m, mode, "dvl_choice", by="task_family", keep=lambda k: not k.endswith("_multi"))),
    ("DVL-Bench, multi-select", "Exact-set match (%)", ["geochrono"],
     lambda m, mode: exact_set(m, mode, "dvl_choice", "dvl/dvl_choice.jsonl")),
    ("xBD building damage", "Accuracy (%)", ["terrascope", "earthdial", "teochat"],
     lambda m, mode: accuracy(m, mode, "xbd_damage_3k")),
    ("xBD building damage", "QWK (0–1)", ["terrascope", "earthdial", "teochat"],
     lambda m, mode: field(m, mode, "xbd_damage_3k", "xBD-damage", ["qwk"])),
    # xBD's own disaster-type labels (fire / flooding / wind / earthquake / tsunami / volcano), the answer format
    # xBD-trained models use; the earlier event-label rows (xbd_disaster_type: hurricane, tornado, ...) are superseded
    ("xBD disaster type", "Accuracy (%)", ["terrascope", "earthdial", "teochat", "waybob-xview2"],
     lambda m, mode: accuracy(m, mode, "xbd_disaster_type_xbdlabels")),
    ("DisasterM3, single-choice", "Accuracy (%)", ["terrascope"],
     lambda m, mode: raw_accuracy(m, mode, "disasterm3_mc_5k", keep=lambda r: not r["task_family"].endswith("_multi"))),
    ("DisasterM3, multi-select", "Exact-set match (%)", ["terrascope"],
     lambda m, mode: exact_set(m, mode, "disasterm3_mc_5k", "disasterm3/disasterm3_mc_5k.jsonl")),
    ("EarthMind, optical only", "Accuracy (%)", ["earthmind"], lambda m, mode: accuracy(m, mode, "earthmind_mc_optical")),
    # EarthMind SAR only and optical + SAR: dropped with SAR (2026-10-29)
    ("fMoW-WILDS, continent", "Accuracy (%)", ["geo-r1", "geozero"],
     lambda m, mode: accuracy(m, mode, "fmow_wilds_ood_2k", by="qid", key="region")),
    ("fMoW-WILDS, country", "Accuracy (%)", ["geo-r1", "geozero"],
     lambda m, mode: accuracy(m, mode, "fmow_wilds_ood_2k", by="qid", key="country")),
    ("fMoW-WILDS, facility type", "Accuracy (%)", ["geozero", "geo-r1"],
     lambda m, mode: accuracy(m, mode, "fmow_wilds_ood_2k", by="qid", key="land_use")),
    ("UrBench geo-localization", "Accuracy (%)", ["geo-r1"], lambda m, mode: accuracy(m, mode, "urbench_geo")),
    ("VHM, honest questions", "Accuracy (%)", ["vhm", "geozero"],
     lambda m, mode: accuracy(m, mode, "vhm_hnstd", by="task_family",
                              keep=lambda k: k in ("abspos_true", "relpos_true", "color_true", "presence"))),
    ("VHM, deceptive questions", "Correct abstention (%)", ["vhm", "geozero"],
     lambda m, mode: accuracy(m, mode, "vhm_hnstd", by="task_family", keep=lambda k: "deceptive" in k)),
]


def table() -> list[dict]:
    out = []
    for name, metric, models, get in ROWS:
        entry = {"row": name, "metric": metric, "models": {}}
        for m in models:
            entry["models"][m] = {mode: get(m, mode) for mode in MODES}
        out.append(entry)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json")
    args = p.parse_args()
    t = table()
    for e in t:
        cells = []
        for m, v in e["models"].items():
            f = (lambda x: "—" if x is None else (f"{x:.3f}" if x <= 1.0 and "0–1" in e["metric"] else f"{x:.1f}"))
            cells.append(f"{m} " + " / ".join(f(v[k]) for k in MODES))
        print(f"{e['row']:28s} {e['metric']:22s} " + " | ".join(cells))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(t, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
