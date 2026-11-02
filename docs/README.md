# eo-jev: benchmarking and fine-tuning `vllm-sr/d3-flash` on Earth observation

| Doc | What's in it |
|---|---|
| [01_literature_review.md](01_literature_review.md) | What d3-flash is (architecture, prompt contract, limits), the Jev / JEV / Decision-model ecosystem and its training recipes, the EO-VLM landscape, the closest prior work (RSJEV / RSAJEV), and known EO bottlenecks |
| [02_benchmarks.md](02_benchmarks.md) | Catalog of ~80 EO benchmarks with d3-fit ratings, conversion recipes, reference scores, licenses and leakage notes; core and extended suites |
| [03_benchmark_plan.md](03_benchmark_plan.md) | The benchmarking plan: parity, harness, baselines, metrics, robustness, ablations, probes, contamination controls (public benchmarks only), decision gate |
| **[04_benchmark_report.md](04_benchmark_report.md)** | **Results: 21 benchmarks / ~145k questions; d3-flash vs Qwen3.5-9B (same prompt and readout) vs no-image; robustness, ablations, probes, capability map, bottlenecks → fine-tuning levers** |
| [05_specialist_models.md](05_specialist_models.md) | Open-weight EO specialist models per benchmark (checked 2026-10-17), train-split overlap, proposed SOTA re-measurement run set |
| **[06_final_benchmark_suite.md](06_final_benchmark_suite.md)** | **Freeze proposal for the evaluation suite (eval-v1.0): community-aligned categories, keep / add per benchmark with adoption and difficulty evidence, freeze and training-exclusion rules, decisions D1–D7 for the user** |
| [results/phase0.md](results/phase0.md) | A100 parity with the official run, latency, cluster issues |
| [results/summary_tables.md](results/summary_tables.md) | Generated tables (`python -m bench.analysis.summarize`) |
| [results/d3_flash_core.md](results/d3_flash_core.md) | Per-benchmark detail and running notes |

**Key facts**
- "vllm-sr d3 8B" = `vllm-sr/d3-flash`:
  - Qwen3.5-9B base; full-fine-tuned text tower; frozen vision tower; 255-code readout.
  - **No text generation**; inputs ≤1.6 MP RGB.
- **Code:**
  - `bench/converters/` (one per benchmark), `bench/runners/d3.py` (resumable, chunk-claiming), `bench/metrics/score.py`, `bench/probes/`, `bench/analysis/summarize.py`.
  - `slurm/run_d3.sbatch` (multi-GPU job with watchdog), `scripts/` (downloads, env, scoring, repairs).
- **Storage** (paths set per machine in `env.local.sh`, see the top-level README):
  - Datasets: `$EOJEV_DATASETS` (`raw/`, `converted/`).
  - Models, runs and third-party code: `$EOJEV_MODELS_ROOT` (`hf_cache/`, `runs/`, `third_party/`).
  - Python envs: `$EOJEV_ENVS/eojev` and `$EOJEV_ENVS/eo-*` for the specialists.
