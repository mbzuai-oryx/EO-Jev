"""Flatten $EOJEV_RUNS/analysis/calibration_study.json (bench/analysis/calibration_study.py) into the row tables the
calibration dashboard reads. Values are fractions, as computed; the page formats them.

- **row_deltas.json:** one row per (benchmark row, baseline, metric). It holds d3-flash's value, the baseline's value
  and the difference d3 - baseline, each with its 95% CI, plus the share of bootstrap draws where the difference is
  below 0.
- **pooled.json:** the same, for each pooled set. The "Qwen3.5-9B" set is all 29 rows; each specialist set is the
  rows that specialist answered, with every row weighted equally.
- **reliability.json:** per pooled set, system and confidence bin: mean confidence, accuracy with a 95% CI, and the
  bin's share of the weight.
- **risk_coverage.json:** error rate of the kept questions at 50 coverage levels. Pooled sets have 95% bands;
  scope "row" gives the curves of d3-flash and Qwen3.5-9B on each benchmark row.

    MODULE=bench.analysis.calibration_tables sbatch -J eojev-calibt slurm/convert_cpu.sbatch  -> reports/calibration/
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.common import RUNS

SRC = RUNS / "analysis" / "calibration_study.json"
OUT = Path(__file__).resolve().parents[2] / "reports" / "calibration"
NAMES = {"d3": "d3-flash", "qwen": "Qwen3.5-9B", "rsthinker": "RSThinker", "geozero": "GeoZero", "earthmind": "EarthMind",
         "earthmind-multi": "EarthMind (optical + SAR)", "earthdial": "EarthDial", "geochrono": "GeoChrono",
         "terrascope": "TerraScope", "teochat": "TEOChat", "waybob-xview2": "Qwen3-VL-8B xView2 LoRA", "geo-r1": "Geo-R1",
         "vhm": "VHM-7B", "scorers-r1": "scorers-r1"}
METRICS = ("accuracy", "ece", "aece", "brier", "aurc", "eaurc")


def name(s: str) -> str:
    return NAMES.get(s, s)


def comparison_rows(label: dict, comp: dict, d3: str = "d3") -> list[dict]:
    out = []
    for base, deltas in comp["delta"].items():
        for m in METRICS:
            a, b, d = comp["systems"][d3][m], comp["systems"][base][m], deltas[m]
            out.append({**label, "baseline": name(base), "metric": m, "n": comp["n"], "groups": comp["groups"],
                        "d3": a[0], "d3_lo": a[1], "d3_hi": a[2], "base": b[0], "base_lo": b[1], "base_hi": b[2],
                        "delta": d[0], "delta_lo": d[1], "delta_hi": d[2], "p_below0": d[3]})
    return out


def main() -> None:
    rep = json.loads(SRC.read_text())
    grid = rep["meta"]["grid"]
    OUT.mkdir(parents=True, exist_ok=True)

    row_deltas = []
    for i, e in enumerate(rep["rows"]):
        for comp in e["comparisons"].values():
            row_deltas += comparison_rows({"row": e["row"], "order": i}, comp)

    pooled, reliability, rc = [], [], []
    for key, p in rep["pooled"].items():
        label = "All 29 rows" if key == "qwen" else f"{name(key)} rows"
        pooled += comparison_rows({"set": label, "set_rows": len(p["rows"]), "rows_list": "; ".join(p["rows"])}, p)
        for sys, bins in p["reliability"].items():
            reliability += [{"set": label, "system": name(sys), **b} for b in bins]
        for sys, c in p["risk_coverage"].items():
            rc += [{"scope": "pooled", "set": label, "system": name(sys), "coverage": g, "risk": r, "lo": lo, "hi": hi}
                   for g, r, lo, hi in zip(grid, c["risk"], c["lo"], c["hi"])]
    for e in rep["rows"]:
        for sys, risks in e.get("risk_coverage", {}).items():
            rc += [{"scope": "row", "set": e["row"], "system": name(sys), "coverage": g, "risk": r, "lo": None, "hi": None}
                   for g, r in zip(grid, risks)]

    for fname, rows in (("row_deltas", row_deltas), ("pooled", pooled), ("reliability", reliability),
                        ("risk_coverage", rc)):
        (OUT / f"{fname}.json").write_text(json.dumps(rows))
        print(fname, len(rows), "rows")


if __name__ == "__main__":
    main()
