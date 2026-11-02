# 03 — Initial plan: benchmarking `vllm-sr/d3-flash` on Earth observation

*Draft v0, 2026-10-16. The benchmark list it refers to lives in [02_benchmarks.md](02_benchmarks.md); background is in [01_literature_review.md](01_literature_review.md).*

## 1. Goals

| # | Goal | Concrete output |
|---|---|---|
| G1 | **Show what d3-flash can do on EO**, zero-shot, on well-known benchmarks | Leaderboard table vs. strong baselines, with 95% CIs |
| G2 | **Find the bottlenecks** and *attribute* each one to a cause:<br>(a) perception / vision representation<br>(b) decision readout / label binding<br>(c) input pipeline (resolution, bands, time)<br>(d) domain knowledge / vocabulary | Bottleneck table ranked by size, each tied to an ablation or probe that isolates it |
| G3 | **De-risk fine-tuning** | A converter + row format that doubles as the training-data pipeline; a go/no-go on d3-flash vs. alternatives; a first fine-tuning recipe |

**Non-goals for this phase:**
- Captioning, free-form VQA and coordinate output. d3-flash has no generative head, so it can do these only through conversion to choices.
- Training.

## 2. Ground rules from the literature (why the plan looks the way it does)

1. **Strong zero-shot baselines on the same backbone are mandatory.** "Releasing the Inner Jev" shows that plain instruction-tuned models read through letter-token logits are already near Jev level. Without the **Qwen3.5-9B base** under the same prompt, we cannot tell what d3's decision training bought (or cost) on EO.
2. **Score calibration, not just accuracy.** Calibrated probabilities are d3's selling point, and no RS-VLM paper reports calibration.
3. **Test the known failure modes of letter readouts:** option-order flips (5.1% measured independently), vocabulary binding and open-set "silent hallucination" (RSAJEV). Also rotation invariance, which matters especially for nadir imagery.
4. **Treat classic RS sets as possibly contaminated.** UCM, AID and RSVQA are in many instruction mixes, and d3's training data is unknown. Weight conclusions toward newly collected 2025–26 public benchmarks, and run perturbation and blind controls (§9). Public benchmarks only; no private test set.
5. **Respect the input contract:** RGB 8-bit only, ≤1.6 MP per image, ≥65k px. Every converter makes its tiling / rendering / resizing explicit and logs the resize factor.

## 3. Phase 0 — Environment and parity check (≈2–3 days)

The upstream runtime is validated **only on AMD/ROCm**, and our cluster is A100-40GB. So before any EO number, prove that our setup reproduces the official numbers.

1. **Environment.**
   - Conda env with torch (CUDA build), `transformers==5.17.0`, `flash-linear-attention==0.5.2`, `accelerate`, `torchvision`, `pillow`, `opencv-python-headless`.
   - **Put `HF_HOME` and datasets on the large volumes (`$EOJEV_MODELS_ROOT`, `$EOJEV_DATASETS` in `env.local.sh`)**, because `/home` is 97% full.
2. **Weights** (pin revisions):
   - `vllm-sr/d3-flash` @ **v3.1.0** (`521525effa`, the revision the boards evaluated), ~17 GB.
   - `Qwen/Qwen3.5-9B` @ `c202236…` (d3's exact base).
   - `Qwen/Qwen3-VL-8B-Instruct`.
3. **Smoke test on 1×A100-40GB.** Run text-only, 1 image at 1.6 MP, and 4 images at 1.6 MP. Log peak memory and latency per question.
4. **Parity test.**
   - Pull ~500 public vision-board rows with the [decision-index kit](https://github.com/apolinario/decision-index).
   - Compare our probabilities to `results.jsonl.gz` in [`vllm-sr/d3-flash-decision-index`](https://huggingface.co/datasets/vllm-sr/d3-flash-decision-index).
   - **Pass criteria:** argmax agreement ≥ 99.5% and median |Δp| ≤ 0.005. (Upstream reports ~0.0025 between vLLM and AutoModel.)
   - If it fails, try the fla Triton kernels vs. the PyTorch reference path, check SDPA vs. eager, and check bf16 vs. fp32 readout.
5. **Batch-invariance check:** the same 200 questions at batch sizes 1 / 8 / 32. Upstream engine batching flipped 0.075% of decisions.

**Exit criterion:** parity passes and latency is measured. If d3-flash can't be made to match on CUDA, fall back to running on an AMD allocation or to the vLLM nightly once d3-flash support lands.

## 4. Phase 1 — The harness (≈1 week)

### 4.1 Layout

```
eo-jev/
  bench/
    converters/      # one module per benchmark → JSONL rows + rendered PNGs
    runners/         # d3.py, letter_logits.py (Qwen3.5 / Qwen3-VL), generative.py, clip_zs.py
    metrics/         # accuracy, macro-F1, AUROC/AP, QWK/MAE/RPS, ECE/Brier/NLL, AURC, flip rates
    analysis/        # slicing, bootstrap CIs, reliability diagrams, confusion matrices
  configs/           # one YAML per (benchmark × variant)
  slurm/             # array jobs; 1 GPU per shard
  data -> $EOJEV_MODELS_ROOT
```

### 4.2 Row schema

A superset of the decision-index row and the Decision-2.0 training contract, so **the same rows later become training rows**:

```json
{
  "id": "aid/test/airport_123#q0",
  "group_id": "aid/airport_123",          // scene/tile; split-leakage guard
  "benchmark": "AID", "split": "test",
  "task_family": "scene_cls", "skills": ["lulc", "fine_grained"],
  "images": ["aid/airport_123.png"],
  "state": {"sensor": "Google Earth RGB", "gsd_m": 0.5},   // only in the +meta variant
  "questions": {"q": {"type": "choice",
                       "instructions": "Which scene category best describes this image?",
                       "criteria": {"airport": "...", "bare land": "...", "...": "..."}}},
  "answer": {"q": "airport"},
  "meta": {"orig_px": [600, 600], "resize_factor": 1.0, "n_options": 30,
           "object_area_frac": null, "license": "research-only"}
}
```

### 4.3 Conventions

- **Options.**
  - Use human-readable keys (class names), with descriptions where the original dataset has definitions.
  - For K > 16, run the **full list** as primary (d3 supports 255 codes) and a **16-way sampled** variant (GT + 15 random distractors, fixed seed) as a secondary.
  - Never let the GT position correlate with anything: shuffle with a fixed seed per item.
- **Multi-label** (BigEarthNet, presence): one `noul` per label.
- **Ordinal** (damage level, count bins, density): `score` with level texts.
- **Prompt variants:**
  - **P0:** minimal (the default).
  - **P1:** P0 + metadata `state` (sensor, GSD, date, place).
  - **P2:** 3 paraphrases of the question, for consistency.
- **Large scenes:** tile at native GSD to ≤1280² px, or give the full downscaled view plus zoomed tiles as extra images, depending on the benchmark.
- **Non-RGB:**
  - Multispectral: true-colour and false-colour composites; NDVI/NDWI as grayscale PNGs; a band description in `state`.
  - SAR: VV / VH / ratio rendered as an RGB composite or a grayscale dB stretch (percentile 2–98).
- **Temporal:** multiple images in time order. State says "Image 1 = 2019-05-03, Image 2 = …".

### 4.4 Runner

- Use `model.system_one(state, questions, images)`, grouping all questions about the same image into one call.
- Store the **full probability vector**, the latency and the visual-token count.
- Run as a Slurm array: one A100 per shard, 4 shards per node; resumable via the JSONL output.

## 5. Models to compare

| Role | Model | Readout | Why |
|---|---|---|---|
| **Subject** | `vllm-sr/d3-flash` v3.1.0 | native `system_one` | The only d3 size we evaluate (user decision 2026-10-16: no d3-nano / mini / 27B) |
| Same-backbone control | `Qwen/Qwen3.5-9B` | letter-logit readout under the *identical* d3 prompt ("Inner-Jev" style) | Isolates the effect of decision training |
| Strong general 8B VLM | `Qwen/Qwen3-VL-8B-Instruct` | letter-logit; generative + parse on a subset | Current open 8B SOTA; the 2026 survey shows general models ≥ RS specialists |
| RS specialist(s) | EarthDial-4B, GeoChat-7B (legacy reference); GeoZero (Qwen3-VL-8B RS) if weights are available | generative + answer parsing | Domain-specific reference points; many published numbers |
| Contrastive | GeoRSCLIP-H, RemoteCLIP-L, SkyCLIP-L (classification and retrieval only) | cosine softmax | Cheap, strong, calibrated-ish classification baselines |
| Floors | chance, majority class, **blind** (d3-flash with no image) | — | Detects language-prior shortcuts in benchmarks |
| Published | GPT-4o / Gemini / GeoChat / EarthDial numbers from papers | — | Quoted, not re-run; flagged when protocols differ |

## 6. Benchmark suite

*Full catalog (~70 benchmarks), conversion recipes, reference scores and licenses: [02_benchmarks.md](02_benchmarks.md).*

**Core suite (Tier 1, ≈115k question-forwards for d3-flash):**

| # | Benchmark | Axis | d3 type |
|---|---|---|---|
| C1 | GEOBench-VLM (3,211 MCQ) | broad, incl. 1,054 counting | `choice` |
| C2 | CHOICE (9,407 MCQ) | perception / reasoning, **low leakage** | `choice` |
| C3 | VRSBench VQA (5k converted) + VG (2k box-choice) | VQA, grounding | `noul` / `choice` |
| C4 | RSVQA-HR test-2 (5k) + RSVQA-LR | canonical VQA, count bins | `noul` / `choice` |
| C5 | LHRS-Bench (690 × 4 orders) | sanity, order robustness | `choice` |
| C6 | UCM, AID, WHU-RS19, SIRI-WHU, EuroSAT-RGB, NWPU (flagged) | scene classification (RSJEV-comparable) | `choice` (N-way + 16-way) |
| C7 | BigEarthNet v2 RGB (1k images × 19 labels) | multi-label LULC | `noul` |
| C8 | CDVQA test2 (5k) | bi-temporal change | `noul` / `choice` |
| C9 | xBD building damage (3k crops) + disaster type; data from the HF mirror `kshitijrajsharma/xview2-xbd` (test + hold parquet, CC-BY-NC-SA; user decision 2026-10-16) | **ordinal** pre/post | `score` / `choice` |
| C10 | DisasterM3 MC (5k, optical + SAR) | disaster, optical → SAR | `choice` / `noul` |
| C11 | EarthMind-Bench MC (2,841) | SAR, optical–SAR fusion | `choice` |
| C12 | DVL-Bench (5,854) or VLRS-Bench | multi-temporal reasoning | `choice` / `noul` |
| C13 | XLRS-Bench-lite (downscale vs. tiles) | ultra-high-resolution stress | `choice` |
| C14 | VHM HnstD (1,642) | hallucination / deceptive questions | `noul` / `choice` |
| C15 | UrBench Geo-Localization (4.5k) + fMoW-WILDS continent / country (2k) | geo-localisation, cross-view | multi-image `choice` |

**Build order:**
1. Week 1: C1, C6, C4, C5. These are native or near-native, give fast first numbers, and have the most published baselines.
2. Week 2: C2, C7, C8, C9, C3.
3. Week 3: C10–C15, which need multi-image, SAR or tiling.

**Extended suite (Tier 2):** resolution (MME-RW-RS, RSHR, LRS-VQA); broad MCQ (OmniEarth, VLRS, UrBench); SAR / MS (SARLANG-1M, BigEarthNet.txt, EuroSAT-MS / SAR, GEO-Bench and Copernicus-Bench classification); fine-grained (FAIR1M crops, SkyScript, fMoW-WILDS OOD); converted spatial (DIOR-RSVG box-choice, RSICD / RSITMD caption-choice, University-1652); robustness (REOBench); UAV / video (MM-UAVBench, UrbanVideo-Bench).

## 7. Metrics and statistics

| Question type | Primary | Secondary |
|---|---|---|
| `choice` | top-1 accuracy | macro-F1 / balanced accuracy, top-3 |
| `noul` | AUROC | AP, accuracy@0.5, balanced accuracy; for multi-label: micro/macro-F1, mAP |
| `score` | quadratic weighted κ | MAE of expected value, exact accuracy, RPS |
| **all** | **ECE** (15 bins, top-label), **Brier**, **NLL** | classwise-ECE; ECE *after* temperature scaling on a 20% split ("calibratability"); reliability diagrams; selective-prediction AURC and accuracy@80% coverage |

- **Robustness metrics:**
  - **Permutation flip rate:** 3 random permutations + reversed order.
  - **Rotation consistency:** 0/90/180/270° and flips; "all-4-correct" accuracy as in TDBench.
  - **Paraphrase consistency.**
  - **Blind accuracy:** no image.
  - **Corruption drop** on a subset with haze, cloud, JPEG and noise, REOBench-style.
- **Open-set / vocabulary binding (RSAJEV test):**
  - Hold out classes, then score in/out AUROC of max-prob, and how often a "none of these" option gets chosen.
  - Cross-dataset label transfer on shared classes (AID↔UCM↔NWPU).
- **Efficiency:** ms per question and per image, and visual tokens per question, on A100.
- **Statistics:**
  - Bootstrap 95% CIs (1,000 resamples, clustered by `group_id`).
  - Paired bootstrap or McNemar vs. each baseline.
  - Fixed seeds for all distractor and permutation sampling.
  - Release the converted question files so the numbers are reproducible.

## 8. Phase 3 — Bottleneck diagnostics (the core of G2)

### 8.1 Slicing (free once the runs exist)

- task family
- GSD bin
- object size (area fraction) and count magnitude, for detection-derived questions
- number of options
- resize factor
- modality (RGB / MS composite / SAR)
- number of images (temporal)
- class frequency
- source dataset

### 8.2 Controlled ablations (on ~1–2k-item subsets)

| ID | Ablation | Question it answers |
|---|---|---|
| A1 | Resolution: native → ½ → ¼, vs. tiles / zoom crops as extra images | How much is lost to the 1.6 MP cap? Is tiling worth building into training? |
| A2 | Metadata `state` on/off (GSD, sensor, date, location) | Does the model use EO context? (A cheap win if so) |
| A3 | MS rendering: true-colour / false-colour / index images / multi-image stack | Best way to feed non-RGB without changing the model |
| A4 | Temporal: multi-image vs. video; image order; explicit dates | How to encode change for d3 |
| A5 | Option count: 4 / 8 / 16 / 32 / full | Does accuracy collapse with large label sets (NWPU 45, fMoW 62)? |
| A6 | `permutation_average` on/off | Is order bias a meaningful error source? |
| A7 | Label wording: bare class names vs. names + definitions | Vocabulary binding vs. concept understanding |

### 8.3 Representation-vs-readout probes (decides *what* to fine-tune)

- **Features:** extract (i) d3-flash's last-token hidden state (pre-readout) and (ii) mean-pooled vision-tower features. The vision tower is bit-identical to Qwen3.5-9B, so it is shared with the base.
- **Probes:** on each benchmark's *train* split, fit (a) a linear probe and (b) an RSAJEV-style prototype / Mahalanobis readout with K=16 per class.
- **Interpretation:**
  - If the probe on the hidden state ≫ zero-shot readout: **the bottleneck is the readout / label binding.** Fix with cheap LoRA + head training on EO decision data.
  - If the probe on vision features is also low: **the bottleneck is perception.** Fix with higher resolution / tiling, and consider unfreezing the merger or the last ViT blocks.
  - If the probe on vision features ≫ the probe on the hidden state: **the LM loses information.** Fine-tune the text tower (LoRA or full at low LR).

### 8.4 Same-backbone comparison

- **d3-flash ≈ Qwen3.5-9B base** on EO: decision training neither helped nor hurt; EO skill is a backbone question.
- **d3-flash < base:** decision training *hurt* EO (domain forgetting). Fine-tuning should start from the base with a JEV-style adapter, not from d3.
- **d3-flash > base:** decision training transfers. Fine-tune d3-flash.

### 8.5 Error taxonomy

Hand-label ~200 errors into:
- perception / small object
- counting
- spatial relation
- fine-grained confusion
- label ambiguity / annotation noise
- temporal
- modality

This gives the qualitative story and cleans noisy benchmark items.

## 9. Phase 4 — Contamination checks (public benchmarks only)

*Decision 2026-10-16: evaluate on public benchmarks only; no private / self-collected test set.* Contamination is therefore estimated, not ruled out, using:

- **Perturbation gap:** for UCM / AID / NWPU / RSVQA, compare accuracy on original vs. rotated + cropped + colour-jittered images. A large gap relative to newer benchmarks hints at memorisation.
- **Blind and shuffled-image controls:** run every benchmark with no image, and with images shuffled across items. Accuracy well above chance points to answerable-from-text questions or memorised question–answer pairs.
- **Old vs. new benchmark comparison:** compare the same capability on classic sets (UCM, AID, RSVQA) and newly collected 2025–26 sets (CHOICE, EarthMind-Bench, DVL-Bench, VLRS-Bench, BigEarthNet v2 held-out test, fMoW-WILDS OOD). Report both, and draw conclusions mainly from the newer ones.

## 10. Phase 5 — Report and decision gate

**Deliverables:**
1. Leaderboard: d3-flash vs. baselines per benchmark and per capability axis (radar plot).
2. Calibration panel (reliability diagrams, ECE before/after temperature scaling).
3. Bottleneck ranking with the evidence for each (§8).
4. Fine-tuning recommendation.

**Decision rules (draft):**

| Finding | Fine-tuning implication |
|---|---|
| Readout-probe gap large | JEV-9B recipe: LoRA r16–64 on text tower + readout, CE/KL + Brier/RPS, 30% option permutation, rotation/flip aug, general replay |
| Big gains from tiling (A1) | Build multi-image zoom/tile inputs into training *and* inference |
| MS/SAR ≪ RGB | Invest in rendered-composite training data; consider unfreezing the vision merger |
| Poor calibration / open-set AUROC < 0.7 | Temperature scaling per type + Brier loss + explicit abstain options; prototype readout as a fallback |
| d3-flash < Qwen3.5-9B base on EO | Fine-tune from the base with a JEV-style separate adapter (keeps System 2), not from d3 |
| d3-flash ≪ Qwen3-VL-8B | Consider a Jev-style head on Qwen3-VL-8B instead |

## 11. Compute, storage and timeline

- **Compute.** The core suite is ≈115k questions.
  - d3-flash plus 3–4 baselines (CLIP only on classification), plus ablations and robustness variants on 1–2k subsets: ≈500k question-forwards.
  - At an assumed 0.3–0.5 s each on A100 (to be measured in Phase 0), that is **~40–70 GPU-hours**: 1–2 days on two 4×A100 nodes.
  - BigEarthNet dominates (19 questions per image). Cut to 500 images if needed.
- **Probes.** Feature extraction is one forward pass per image (cached); probes are CPU-cheap.
- **Storage.** Raw datasets total ~0.5–1 TB (BigEarthNet ~66 GB, xBD ~50 GB, fMoW-RGB 200 GB, so use a subset). Everything goes under `$EOJEV_DATASETS` (set in `env.local.sh`).
- **Timeline (≈4 weeks):**

| Week | Work |
|---|---|
| 1 | Phase 0 parity; harness skeleton; converters for scene classification, GEOBench-VLM and RSVQA; first numbers |
| 2 | Remaining core converters; baseline runners (Qwen3.5 letter-logit, Qwen3-VL, CLIP) |
| 3 | Full core runs; ablations A1–A7; robustness metrics |
| 4 | Probes; contamination checks; report + fine-tuning decision |

## 12. Risks

| Risk | Mitigation |
|---|---|
| d3-flash runtime misbehaves on CUDA (fla kernels, bidirectional mask) | Phase-0 parity test; fla PyTorch reference fallback; AMD allocation or vLLM nightly as backup |
| No prefix caching (bidirectional attention), so each question re-encodes the image | Group ≤8 questions per call (runtime default); budget is still small |
| Option counts > ~40 are out-of-distribution | Report full-list and 16-way variants separately (A5) |
| Licenses (many NC / academic-only, RescueNet ND) | Research use only; don't redistribute converted images; release converters, not data |
| Benchmark protocol mismatch with published numbers (e.g. grounding Acc@0.5 vs. K-way choice) | Always report chance level; label converted tasks as "-MC" variants; never put them in the same column as the original metric |
| Qwen3.5 letter-logit baseline set up unfairly | Same prompt, same images, chat template with thinking disabled; validate on the decision-index public rows first |
