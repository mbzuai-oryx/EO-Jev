# Phase 0 — d3-flash runs correctly on our A100s (2026-10-16)

**Verdict: pass.** On our A100-40GB nodes, d3-flash reproduces the official d3-flash Decision Index run (made on AMD MI325X) request by request. Benchmark numbers produced on this cluster are therefore the real model's numbers.

## Setup

| Item | Value |
|---|---|
| Model | `vllm-sr/d3-flash` @ v3.1.0 (`521525effa`); weights identity `74b71d37…` matches the official run |
| Stack | torch 2.13.0+cu126, transformers 5.17.0, flash-linear-attention 0.5.2, Python 3.12; env `$EOJEV_ENVS/eojev` (`scripts/setup_env.sh`) |
| Kernels | fla `chunk_gated_delta_rule` for Gated DeltaNet; PyTorch reference `causal_conv1d`. Same set as the official run. The AMD-only fast path (`d3_fast.py`) switches itself off on CUDA |
| GPU | NVIDIA A100-SXM4-40GB, driver 570.195 |

## Parity with the official results (`bench/parity_compare.py`, `slurm/phase0_parity.sbatch`)

- **Rows:** Decision Index rows rebuilt with the kit at the official commit `9eb2dbe`.
  - All of BANKING77 (3,080 requests, **77 options**, so two-letter answer codes) and ARC-Challenge (1,172).
  - 500 each of MMLU, ARC-Easy, WinoGrande and HellaSwag.
- **Engine:** the official one (`d3_engine:D3Engine`).
- **Matching:** joined on `run_id`, keeping only requests whose payload hashes match.

| | Questions | Same top answer | Median max \|Δp\| | p95 | Max |
|---|---:|---:|---:|---:|---:|
| **All** | **6,251** | **99.94%** | **0.0002** | 0.007 | 0.037 |
| BANKING77 (77 options) | 3,080 | 99.94% | 0.0006 | 0.007 | 0.025 |
| ARC-Challenge | 1,172 | 100% | 0.0001 | 0.005 | 0.034 |
| MMLU | 499 | 99.60% | 0.0008 | 0.012 | 0.032 |
| ARC-Easy / HellaSwag / WinoGrande | 1,500 | 100% | ≤0.0003 | ≤0.008 | 0.037 |

- **Pass criteria:** ≥99.5% same top answer and median max|Δp| ≤ 0.005. Both are met.
- **Payload-hash mismatches:** 0.
- **Unmatched rows:** one MMLU row is not in the official 0.3 set.
- **The 4 disagreements:** all near-ties.

## Smoke test (a one-off script, not kept in the release)

- The model card's requests give sensible answers: route "returns" 0.998; receipt yes 0.999; payment "card" 1.000.
- 45- and 150-option choices return complete distributions.
- Peak GPU memory is **18.3 GB**, so it fits a 40 GB A100 with room to spare.

**Measured latency on one A100** (wall time per request):

| Image | 1 question | 8 questions | per question at 8 |
|---|---:|---:|---:|
| none (3 text questions) | 0.05 s | — | — |
| 256×256 | 0.07 s | 0.18 s | 0.02 s |
| 512×512 | 0.08 s | 0.39 s | 0.05 s |
| 1024×1024 | 0.18 s | 1.29 s | 0.16 s |
| 1264×1264 (1.6 MP cap) | 0.30 s | 2.12 s | 0.27 s |
| 4 × 1024×1024 | 0.66 s | — | — |

- **Questions do not share the image encoding.** Each question is its own forward pass over the full prompt (bidirectional attention rules out prefix caching), so cost grows linearly with the number of questions per image. Many-`noul` designs (BigEarthNet: 19 per image) pay for the image 19 times; that is affordable only because the images are small.
- The compute estimates in [03_benchmark_plan.md](../03_benchmark_plan.md) §11 hold.

## Cluster issues found (and handled)

- **QoS:** per user, 2 running and 4 submitted jobs, and 4 GPUs. Runs therefore use one job with 4 single-GPU tasks (`slurm/run_d3.sbatch`), which can land on different nodes.
- **Broken GPUs:** some GPUs are listed by NVML but fail CUDA initialization.
  - `gpu-55`: every GPU.
  - `gpu-05`: only 3 of its 4 GPUs are present.
  - `gpu-52`: one GPU.
  - Fix: `gpu-55` is excluded. The runner loads the model before claiming any work and claims work in 64 chunks through lock files, so a task on a dead GPU exits and the healthy ones finish the file.
- **Submit filter:** the cluster's submit filter rejects scripts containing `tail` / `ls` ("idle/monitoring commands").
