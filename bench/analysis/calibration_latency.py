"""Calibration (ECE) and inference time per benchmark-guide row, for d3-flash, Qwen3.5-9B and the open specialists, from
the run records alone (no new inference).

ECE: 15 equal-width bins on the top answer's probability (bench/metrics/score.py), recomputed here on exactly the same
questions for every system; only for probability outputs (d3-flash, Qwen and the specialists' option-probability mode).
A written answer has no confidence (it is one-hot), so its ECE would only be 1 - accuracy and is not reported.

Time: from each record's wall_ms. d3-flash and Qwen time one row (image set) per call, with all its questions in one
batch; images are decoded before the clock starts. Specialists time groups of rows and store the group mean per row
(stock-transformers models batch 8 rows for option probabilities and 16 for written answers; models run through their
authors' code take one question at a time); image decoding is inside the clock. GPUs were shared with other users'
jobs, so times are indicative. Reported: GPU seconds per 1,000 questions and the median per row.

    python -m bench.analysis.calibration_latency [--json out.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from bench.common import CONVERTED, RUNS, is_sar, read_jsonl
from bench.metrics.score import ece, question_records

SPEC = RUNS / "specialists"

# specialist run name -> (rows file under converted/, d3 / Qwen run dir name)
RUNS_MAP = {
    "scene_all": ("scene/scene_all", "scene_nway"),
    "bigearthnet_v2_test_1k": ("bigearthnet/bigearthnet_v2_test_1k", "bigearthnet"),
    "geobench_vlm_single": ("geobench_vlm/geobench_vlm_single", "geobench_vlm"),
    "choice_mc": ("choice/choice_mc", "choice"),
    "lhrs_bench_circular": ("lhrs_bench/lhrs_bench_circular", "lhrs_bench"),
    "vrsbench_vqa_mc_5k": ("vrsbench/vrsbench_vqa_mc_5k", "vrsbench_vqa_mc"),
    "rsvqa_lr": ("rsvqa/rsvqa_lr", "rsvqa_lr"),
    "rsvqa_hr_test2_5k": ("rsvqa/rsvqa_hr_test2_5k", "rsvqa_hr"),
    "cdvqa_test2_5k": ("cdvqa/cdvqa_test2_5k", "cdvqa"),
    "dvl_choice": ("dvl/dvl_choice", "dvl"),
    "xbd_damage_3k": ("xbd/xbd_damage_3k", "xbd_damage"),
    "xbd_disaster_type": ("xbd/xbd_disaster_type", "xbd_type"),
    "xbd_disaster_type_xbdlabels": ("xbd/xbd_disaster_type_xbdlabels", "xbd_type_xbd"),
    "disasterm3_mc_5k": ("disasterm3/disasterm3_mc_5k", "disasterm3"),
    "earthmind_mc_optical": ("earthmind/earthmind_mc_optical", "earthmind_optical"),
    "earthmind_mc_sar": ("earthmind/earthmind_mc_sar", "earthmind_sar"),
    "earthmind_mc_fused": ("earthmind/earthmind_mc_fused", "earthmind_fused"),
    "fmow_wilds_ood_2k": ("fmow/fmow_wilds_ood_2k", "fmow_wilds"),
    "urbench_geo": ("urbench/urbench_geo", "urbench_geo"),
    "vhm_hnstd": ("vhm/vhm_hnstd", "vhm_hnstd"),
    # not benchmark-guide rows; used by bench/analysis/compare_package.py
    "det_count_nwpu_vhr10": ("det_count/det_count_nwpu_vhr10", "det_count_nwpu_vhr10"),
    "det_count_dota_v2_val": ("det_count/det_count_dota_v2_val", "det_count_dota_v2_val"),
    "det_count_vhr10_absent": ("det_count/det_count_vhr10_absent", "det_count_vhr10_absent"),
    "decision_index": ("decision_index/parity_rows", "decision_index"),
}

HONEST = {"abspos_true", "relpos_true", "color_true", "presence"}


def _all(r):
    return True


def _bench(name):
    return lambda r: r["benchmark"] == name


def _single(r):
    return not r["task_family"].endswith("_multi")


def _multi(r):
    return r["task_family"].endswith("_multi")


def _qid(q):
    return lambda r: r["qid"] == q


# website row -> (specialist run name, specialists, question filter); same rows as bench/analysis/specialist_table.py
ROWS = [
    ("UC Merced", "scene_all", ["rsthinker", "geozero"], _bench("UCM")),
    ("AID", "scene_all", ["rsthinker", "geozero"], _bench("AID")),
    ("WHU-RS19", "scene_all", ["rsthinker", "geozero"], _bench("WHU-RS19")),
    ("SIRI-WHU", "scene_all", ["rsthinker", "geozero"], _bench("SIRI-WHU")),
    ("EuroSAT (RGB)", "scene_all", ["rsthinker", "geozero"], _bench("EuroSAT-RGB")),
    ("NWPU-RESISC45", "scene_all", ["rsthinker", "geozero"], _bench("NWPU-RESISC45")),
    ("BigEarthNet v2", "bigearthnet_v2_test_1k", ["earthmind"], _all),
    ("GEOBench-VLM", "geobench_vlm_single", ["geozero"], _all),
    ("CHOICE", "choice_mc", ["geozero"], _all),
    ("LHRS-Bench", "lhrs_bench_circular", ["scorers-r1"], _all),
    ("VRSBench (converted)", "vrsbench_vqa_mc_5k", ["rsthinker"], _all),
    ("RSVQA-LR", "rsvqa_lr", ["earthdial"], _all),
    ("RSVQA-HR (test 2)", "rsvqa_hr_test2_5k", ["geozero"], _all),
    ("CDVQA (test 2)", "cdvqa_test2_5k", ["geochrono"], _all),
    ("DVL-Bench, single-choice", "dvl_choice", ["geochrono"], _single),
    ("DVL-Bench, multi-select", "dvl_choice", ["geochrono"], _multi),
    ("xBD building damage", "xbd_damage_3k", ["terrascope", "earthdial", "teochat"], _all),
    ("xBD disaster type", "xbd_disaster_type_xbdlabels", ["terrascope", "earthdial", "teochat", "waybob-xview2"], _all),
    ("DisasterM3, single-choice", "disasterm3_mc_5k", ["terrascope"], _single),
    ("DisasterM3, multi-select", "disasterm3_mc_5k", ["terrascope"], _multi),
    ("EarthMind, optical only", "earthmind_mc_optical", ["earthmind"], _all),  # SAR variants dropped 2026-10-29
    ("fMoW-WILDS, continent", "fmow_wilds_ood_2k", ["geo-r1", "geozero"], _qid("region")),
    ("fMoW-WILDS, country", "fmow_wilds_ood_2k", ["geo-r1", "geozero"], _qid("country")),
    ("fMoW-WILDS, facility type", "fmow_wilds_ood_2k", ["geozero", "geo-r1"], _qid("land_use")),
    ("UrBench geo-localization", "urbench_geo", ["geo-r1"], _all),
    ("VHM, honest questions", "vhm_hnstd", ["vhm", "geozero"], lambda r: r["task_family"] in HONEST),
    ("VHM, deceptive questions", "vhm_hnstd", ["vhm", "geozero"], lambda r: "deceptive" in r["task_family"]),
]


def _results(run_dir) -> list[dict]:
    """One record per row id: the last ok record if any, else the last one. Reruns are appended to the chunk files or,
    for d3 / Qwen, written to retry/ (as in scripts/score_all.sh)."""
    latest = {}
    for f in sorted(glob.glob(str(run_dir / "chunk*.jsonl"))) + sorted(glob.glob(str(run_dir / "retry" / "chunk*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            prev = latest.get(r["id"])
            if prev is None or r.get("status") == "ok" or prev.get("status") != "ok":
                latest[r["id"]] = r
    return list(latest.values())


_cache: dict = {}


def _load(system: str, run: str, mode: str = "logprob"):
    """(question records, row results) for a system on a specialist run name. system is d3, qwen, a specialist, or
    the name of a fine-tuned package whose runs are in $EOJEV_RUNS/<name>/<d3 run dir name> (compare_package.py).
    Rows with SAR images are left out (bench.common.is_sar)."""
    key = (system, run, mode)
    if key not in _cache:
        rows_rel, base_name = RUNS_MAP[run]
        if "rows" not in _cache.setdefault(run, {}):
            _cache[run]["rows"] = {r["id"]: r for r in read_jsonl(CONVERTED / f"{rows_rel}.jsonl") if not is_sar(r)}
        rows = _cache[run]["rows"]
        d = {"d3": RUNS / "d3" / base_name, "qwen": RUNS / "qwen3.5-9b" / base_name}.get(system)
        if d is None:
            d = RUNS / system / base_name if (RUNS / system / base_name).is_dir() else SPEC / system / mode / run
        res = [r for r in _results(d) if r["id"] in rows]
        recs, _ = question_records(rows, res) if res else ([], 0)
        _cache[key] = (recs, res, rows)
    return _cache[key]


def calib(system, run, keep, mode="logprob"):
    recs, _, _ = _load(system, run, mode)
    rs = [r for r in recs if keep(r)]
    if not rs:
        return None
    conf = np.array([r["conf"] for r in rs])
    correct = np.array([r["correct"] for r in rs], dtype=float)
    return {"n": len(rs), "ece": ece(conf, correct), "accuracy": float(correct.mean()), "mean_conf": float(conf.mean())}


def timing(system, run, keep, mode="logprob"):
    """GPU seconds per 1,000 questions of this website row, and the median ms per row (image set). A row's time is
    split over its questions in proportion, so rows that mix kept and dropped questions count partly."""
    recs, res, rows = _load(system, run, mode)
    kept = defaultdict(int)
    for r in recs:
        if keep(r):
            kept[r["id"]] += 1
    total_ms = n_q = 0.0
    per_row = []
    for r in res:
        if r.get("status") != "ok" or r["id"] not in kept or "wall_ms" not in r:
            continue
        share = kept[r["id"]] / len(rows[r["id"]]["questions"])
        total_ms += r["wall_ms"] * share
        n_q += kept[r["id"]]
        per_row.append(r["wall_ms"])
    if not n_q:
        return None
    return {"questions": int(n_q), "s_per_1k_q": total_ms / n_q, "median_ms_per_row": float(np.median(per_row))}


def table() -> list[dict]:
    out = []
    for name, run, models, keep in ROWS:
        e = {"row": name, "run": run,
             "ece": {"d3": calib("d3", run, keep), "qwen": calib("qwen", run, keep)},
             "time": {"d3": timing("d3", run, keep), "qwen": timing("qwen", run, keep)}}
        for m in models:
            e["ece"][m] = calib(m, run, keep)
            if (SPEC / m / "logprob-native" / run).exists():  # the model's own prompt format (native.py)
                e["ece"][f"{m}:native"] = calib(m, run, keep, "logprob-native")
            for mode in ("logprob", "generate", "logprob-native", "generate-native"):
                if (SPEC / m / mode / run).exists():
                    e["time"][f"{m}:{mode}"] = timing(m, run, keep, mode)
        out.append(e)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json")
    args = p.parse_args()
    t = table()
    f = lambda c, k, s="{:.3f}": "—" if not c else s.format(c[k])  # noqa: E731
    print(f"{'row':28s} {'ECE d3':>7s} {'qwen':>6s}  specialists (ECE, n)")
    for e in t:
        spec = " | ".join(f"{m} {f(c, 'ece')}" for m, c in e["ece"].items() if m not in ("d3", "qwen"))
        ns = {m: (c or {}).get("n") for m, c in e["ece"].items()}
        print(f"{e['row']:28s} {f(e['ece']['d3'], 'ece'):>7s} {f(e['ece']['qwen'], 'ece'):>6s}  {spec}   n={ns}")
    print(f"\n{'row':28s} seconds per 1,000 questions (median ms per row)")
    for e in t:
        cells = [f"{k} {f(c, 's_per_1k_q', '{:.0f}')} ({f(c, 'median_ms_per_row', '{:.0f}')})" for k, c in e["time"].items()]
        print(f"{e['row']:28s} " + " | ".join(cells))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        json.dump(t, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
