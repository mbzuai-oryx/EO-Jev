"""The 6,251 phase-0 Decision Index requests (BANKING77, ARC-Challenge, ARC-Easy, MMLU, WinoGrande, HellaSwag; text
only) as benchmark rows, so bench/runners/d3.py can run them on a fine-tuned package. This is the general-decision
regression check: Earth-observation training must not cost d3-flash its general decisions.

    MODULE=bench.converters.decision_index_rows sbatch slurm/convert_cpu.sbatch
    -> $EOJEV_DATASETS/converted/decision_index/parity_rows.jsonl
"""

from __future__ import annotations

from bench.common import CONVERTED, DATASETS, read_jsonl, validate_row, write_jsonl

SRC = DATASETS / "raw" / "decision-index-work" / "parity-rows.jsonl"
OUT = CONVERTED / "decision_index" / "parity_rows.jsonl"


def main() -> None:
    rows = []
    for r in read_jsonl(SRC):
        row = {"id": r["id"], "group_id": r["_evaluation"]["group_id"], "benchmark": r["benchmark"], "split": r["split"],
               "task_family": r["family"], "skills": ["general"], "images": [], "state": r["state"],
               "questions": r["questions"], "answer": r["gold"], "meta": {"run_id": r["_evaluation"]["run_id"]}}
        validate_row(row)
        rows.append(row)
    print("Decision Index rows:", write_jsonl(OUT, rows))


if __name__ == "__main__":
    main()
