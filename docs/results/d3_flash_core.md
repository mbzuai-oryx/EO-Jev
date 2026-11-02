# d3-flash on the EO core suite: running results

*Living document; updated as runs finish. Model `vllm-sr/d3-flash` v3.1.0, zero-shot, prompt variant P0 (empty state, benchmark question as instructions), on A100 (parity-checked: [phase0.md](phase0.md)). 95% CIs are bootstraps over image groups. Reference numbers are copied from the papers in [02_benchmarks.md](../02_benchmarks.md) and often use different protocols, so they are context, not head-to-head comparisons. The same-backbone control (Qwen3.5-9B under the identical prompt and readout) and the no-image runs are pending; they decide what d3's decision training contributed and how much each benchmark is answerable without looking.*

## Results so far (2026-10-16)

| Benchmark (core id) | Questions | d3-flash | Notes / published reference points |
|---|---:|---|---|
| **Scene classification, full class list** (C6) | 20,205 | UCM 82.9, AID 78.1, WHU-RS19 92.1, SIRI-WHU 69.1, EuroSAT-RGB 57.3, NWPU-RESISC45 73.3 | GPT-4o: UCM 88.8, AID 74.7, WHU-RS19 ≈90, EuroSAT 56.8. GeoChat: UCM 84.4, AID 72.0. EarthDial (EO-trained): UCM 92.4, AID 88.8 |
| Scene classification, 16-way (C6) | 12,405 | UCM 85.5, AID 85.2, WHU-RS19 93.4, NWPU 86.8 | GT + 15 random distractors; not comparable to N-way |
| **GEOBench-VLM** (C1) | 3,211 | **51.9** [50.3, 53.6] | Best in paper: LLaVA-OneVision 41.7; GPT-4o slightly lower |
| **LHRS-Bench** (C5), strict CircularEval | 690 | **77.5** (first order 79.6; answer flips 4.9%) | Strict: GPT-4-Turbo 44.6, LHRS-Bot 39.4. Lenient: GPT-4o 67.8, Qwen3-VL 64.1 |
| **RSVQA-LR** (C4) | 10,004 | comparison 78.4, presence 62.9, rural/urban 64.0, count (5 bins) 32.1 | GeoChat (trained on RSVQA-LR train): 90.3 / 91.1 / 94.0. InternVL2-8B 72.3 / 58.5 / 71. Text-only scores 73.8 overall |
| **BigEarthNet v2** (C7), 19-label | 1,000 × 19 | micro-F1 0.44, sample-F1 0.40, macro mAP 0.34 (0.38 on classes with ≥10 positives) | Gemini 2.5 (RGB) F1 0.41–0.45 |

## GEOBench-VLM by task

| Task | n | d3-flash | Task | n | d3-flash |
|---|---:|---:|---|---:|---:|
| Scene classification | 415 | **83.1** | General aircraft counting | 100 | 52.0 |
| Spatial relation | 466 | **81.3** | Ship type | 207 | 45.4 |
| Water bodies counting | 85 | 61.2 | General vehicle counting | 150 | 34.7 |
| Land use | 410 | 59.5 | Building counting | 170 | 33.5 |
| Disaster type | 227 | 58.6 | Trees counting | 85 | 31.8 |
| Aircraft type | 100 | 54.0 | Specific aircraft type counting | 140 | 30.7 |
| Crop type | 55 | 29.1 | Marine debris counting | 100 | 28.0 |
| Tree health | 157 | **23.6** | Fire risk | 120 | **17.5** |
| | | | Specific vehicle type counting | 224 | **15.2** |

Chance is 20% (5 options).

## Same-backbone control: d3-flash vs plain Qwen3.5-9B under the identical prompt and readout

Qwen3.5-9B is d3-flash's exact base. Here it is run through d3's own runtime: same prompt, same image handling, its own `lm_head` rows for the answer codes as the readout, causal attention ([scripts/make_base_package.py](../../scripts/make_base_package.py)). The difference isolates what d3's decision training contributes on EO.

| Benchmark | d3-flash | Qwen3.5-9B | Δ | ECE d3 / base |
|---|---:|---:|---:|---|
| UCM / AID / WHU-RS19 | 82.9 / 78.1 / 92.1 | 75.5 / 73.0 / 88.2 | +7.4 / +5.1 / +4.0 | 0.10 / 0.10; 0.07 / 0.15; 0.11 / 0.05 |
| SIRI-WHU / EuroSAT-RGB / NWPU-RESISC45 | 69.1 / 57.3 / 73.3 | 63.2 / 56.3 / 69.8 | +5.8 / +1.0 / +3.5 | 0.05 / 0.17; 0.04 / 0.18; 0.07 / 0.18 |
| GEOBench-VLM | 51.9 [50.3, 53.6] | 50.4 [48.6, 52.1] | +1.6 | 0.10 / 0.18 |
| RSVQA-LR comparison / presence / rural-urban / count | 78.4 / 62.9 / 64.0 / 32.1 | 74.3 / 65.2 / 65.0 / 32.1 | +4.1 / −2.3 / −1.0 / 0.0 | 0.06 / 0.08 (comparison) |
| LHRS-Bench, original option order | 79.6 | 77.5 | +2.1 | 0.02 / 0.06 |
| **LHRS-Bench, strict (all option rotations correct)** | **77.5** | **70.0** | **+7.5** | |
| **LHRS-Bench, answer changes across option rotations** | **4.9%** | **18.3%** | **3.7× fewer** | |

GEOBench-VLM per task (d3 − base):
- **Gains:** general vehicle counting +12.7, water-body counting +10.6, disaster type +7.5, spatial relation +5.2, ship type +3.4, land use +2.7.
- **Losses:** marine-debris counting −12.0, specific-vehicle counting −6.7, tree health −4.5, general aircraft counting −4.0, fire risk −3.3.
- **Unchanged:** building counting, crop type, trees counting.

| BigEarthNet v2 micro-F1 / macro mAP / AUROC | 0.444 / 0.338 / 0.795 | 0.418 / 0.331 / 0.768 | +0.026 / +0.007 / +0.027 | 0.033 / 0.075 |

**Reading.**
1. EO perception comes overwhelmingly from the Qwen3.5 backbone. Decision training adds a modest +1–7 points of accuracy.
2. Its main EO contribution is **reliability**: answers about 4× more stable under option reordering, and far better calibrated probabilities on almost every benchmark.
3. The weak skills (counting, crop type, tree health, fire risk, 10 m presence) are weak in *both* models. They are backbone / perception limits that the decision training did not fix, so fine-tuning has to bring EO-specific supervision rather than only a better readout.

## Batch 3: VRSBench, xBD, DVL, fMoW, VHM-HnstD, UrBench (d3-flash; Qwen3.5-9B where finished)

| Benchmark / slice | d3-flash | Qwen3.5-9B | Reference |
|---|---:|---:|---|
| **VRSBench-VQA-MC** (converted, 5k): category / color / shape / scene type | 83.4 / 80.2 / 81.7 / 80.5 | 76.8 / 77.4 / 77.0 / 71.8 | Paper numbers are GPT-judged open answers: not comparable |
| ↳ yes/no (AUROC) / rural-urban | 82.5 (0.86) / 93.3 | 83.2 (0.87) / 92.7 | |
| ↳ direction (3-way) / position (9-way) / quantity (8 levels; QWK) | 62.7 / 48.4 / 49.2 (0.62) | 62.7 / 46.8 / 50.1 (0.57) | |
| **xBD building damage** (4 levels, balanced 750 each; pre/post crops) | 41.1 (QWK 0.43, MAE 0.79) | 37.8 (QWK 0.31) | Chance 25. TEOChat (trained on xBD) F1 50.0; GPT-4o (3-shot) 38.3 |
| **xBD disaster type** (7-way, 933 scenes, pre/post 1024 px) | 55.4 | 48.2 | GEOBench-VLM disaster type (xBD-based): GPT-4o .63 |
| **DVL-Bench** basic change, single-choice / multi-select exact-set | 63.4 / 10.8 | 61.3 / 11.3 | o4-mini 34.1 overall, GPT-4o 29.7 (mixed protocol) |
| ↳ change speed (*when*), single-choice / multi-select per-option acc. (exact-set) | **29.2** / 68.3 (16.7) | 25.0 / **26.4 (0.2)** | the base model collapses on multi-select yes/no |
| ↳ *(55 DVL rows / 504 questions ran out of GPU memory on a shared GPU; retry pending, numbers above exclude them)* | | | |
| **fMoW-WILDS OOD** region (5-way) / country (16-way, same-region distractors) / land use (62-way) | 49.4 / **44.4** / 25.9 | 49.5 / 30.4 / 25.9 | Land use: GPT-4o 21.4, GPT-4V 19 |
| **UrBench geo**: image retrieval / city retrieval / camera localization / orientation | **76.9** / 68.5 / **50.0** / 30.5 | 53.8 / 60.8 / 37.2 / 26.6 | GPT-4o 85.9 / 79.2 / 35.3 / 30.7; human 92.6 / 30.0 / 82.9 / 85.7; random ≈24 |

**EarthMind-Bench MC** (1,810 questions; co-registered optical and SAR)

| | Optical | SAR | Fused (optical + SAR) | Paper: GPT-4o | Paper: EarthMind-4B (trained on it) |
|---|---:|---:|---:|---|---|
| d3-flash, all MC | **67.1** | **53.9** | **65.6** | 64.1 / 47.8 / 61.1 | 61.7 / 61.3 / 70.0 |
| hallucination detection | 92.9 | 66.4 | 92.5 | | |
| object counting | 42.3 | 33.3 | 39.9 | | |
| object existence | 83.4 | 79.4 | 82.9 | | |
| scene classification | 98.7 | 81.9 | 96.5 | | |
| spatial relationship | 80.4 | 63.8 | 80.4 | | |

d3 beats GPT-4o in every input mode, including SAR, which it was not trained on. But **fusion does not help**: optical + SAR (65.6) is slightly *below* optical alone (67.1), so the SAR channel is not used as complementary evidence. Counting is again the weakest skill.

**CDVQA test-2** (5,001 questions, bi-temporal 0.15 m pairs)

| Question type | n | d3-flash | Notes |
|---|---:|---:|---|
| did class X change? / increase? / decrease? | 826 / 291 / 283 | 71.5 / 74.6 / 67.5 | AUROC 0.82 / 0.79 / 0.67 |
| what did class X change to? (6-way) | 482 | 46.5 | |
| largest change (6-way) | 468 | 36.8 | Answers "buildings" for 401 of 468; gold is mostly "non-vegetated ground surface" (199) |
| smallest change (6-way) | 468 | **17.5** | Chance 16.7 |
| change percentage (11 bins) | 312 | **5.1** | Chance 9.1. Gold is bimodal (0–20% / 80–100%, partly "unchanged area" questions); d3 picks middle bins |
| change percentage of class X (11 bins) | 1,871 | 37.5 | Mostly the "0" / "0–10%" bins |

Detecting *whether* something changed works. **Quantifying change** (how much, which class changed most or least) does not. Two causes:
- **Salience bias:** "buildings" is picked as the largest change almost regardless of the image.
- **Ungrounded label:** SECOND's "non-vegetated ground surface" class doesn't seem to be grounded (label-definition ablation A7 will test this).

**VHM-HnstD (honesty / hallucination probes)**

| Probe | d3-flash | Qwen3.5-9B |
|---|---:|---:|
| absolute position, true / deceptive (adversarial / popular / random) | 83.0 / **37 / 43 / 59** | 85.7 / 37 / 34 / 58 |
| relative position, true / deceptive | 74.4 / **41 / 34 / 36** | 68.2 / 30 / 24 / 36 |
| color, true / object absent (adversarial / popular / random) / panchromatic image | 46.0 / **3 / 2 / 19** / 99.0 | 51.0 / **22 / 21 / 43** / 100 |
| presence (AUROC) | 79.9 (0.875) | 77.9 |
| image type / GSD bin (6-way) | 90.0 / **38.1** | 75.0 / **53.9** |

Decision training is **mixed on hallucination**:
- It improved abstention on deceptive *position* questions (up to +11).
- It sharply **reduced** abstention on deceptive *color* questions (−19 to −24).
- It cost 16 points on resolution (GSD) awareness.

Fine-tuning data needs explicit abstain examples, and these probes should be regression tests.

Deceptive items ask about objects that are *not* in the image; the right answer is the abstain option ("the object is invisible" / "not in the image").
- **Position questions:** d3 abstains correctly only 34–59% of the time.
- **Color questions:** it almost never abstains (2–19%). It confidently names a color for an absent object.
- **Contrast:** it correctly recognizes a panchromatic image as color-less 99% of the time.

This is the "silent hallucination" pattern reported for letter-readout classifiers (RSAJEV) and a priority for fine-tuning: abstain options in training data, plus calibration.

## No-image control (d3-flash, image removed, same questions)

| Benchmark | No image | With image | |
|---|---:|---:|---|
| Scene classification (6 sets) | 2.0–11.0 | 57.3–92.1 | At chance without the image, as expected |
| LHRS-Bench | 40.8 | 78.5 | Needs the image |
| GEOBench-VLM (chance 20) | 20.5 | 51.9 | |
| ↳ marine-debris counting | **60.0** | **28.0** | **Image makes it worse** |
| ↳ specific-vehicle counting | 20.1 | 15.2 | Image makes it worse |
| ↳ fire risk / crop type / tree health | 15.8 / 21.8 / 12.7 | 17.5 / 29.1 / 23.6 | Image barely helps |
| ↳ water-body counting / land use / disaster type | 35.3 / 28.3 / 25.1 | 61.2 / 59.5 / 58.6 | |
| RSVQA-LR comparison | 75.8 | 78.4 | Mostly answerable from language priors |
| **RSVQA-LR presence (Sentinel-2, 10 m)** | **69.5** | **62.9** | **Image makes it worse** |
| RSVQA-LR count / rural-urban | 27.5 / 56.0 | 32.1 / 64.0 | |

**Reading.**
1. RSVQA-LR is largely a language-prior benchmark (consistent with the literature's 73.8% text-only), so its headline numbers say little about perception.
2. For 10 m presence detection and for some counting tasks, d3's visual evidence is **actively misleading**: it does worse than its own priors without the image. That is the clearest perception bottleneck so far, and a concrete fine-tuning target.

## Emerging bottleneck hypotheses (to be tested with the controls, ablations and probes in docs/03 §8)

1. **Counting.**
   - GEOBench-VLM counting tasks average ≈33% (1,054 questions). That is *below* the published Qwen2-VL / GPT-4o (≈40%) despite d3 leading GEOBench overall.
   - Specific-vehicle counting (15%) is under chance.
   - RSVQA-LR count bins: 32%, QWK 0.42.
2. **Graded assessments and agriculture.** Fire risk (17.5%), tree health (23.6%) and crop type (29%) are at or below chance. BigEarthNet's agricultural and wetland classes have AP < 0.12.
3. **Many-option label sets.** NWPU goes from 73.3 (45 options) to 86.8 (16 options). Part of that is easier random distractors; ablation A5 separates the two.
4. **Small / low-resolution imagery.** EuroSAT-RGB (64 px, 10 m) is the weakest scene set (57%). RSVQA-LR presence (10 m) reaches only AUROC 0.72.
5. **Calibration is task-dependent.**
   - Under-confident on scene classification (e.g. AID: mean confidence 0.72 vs accuracy 0.78).
   - Over-confident on counting (marine debris: 0.60 vs 0.28).
   - Well calibrated on LHRS-Bench (ECE 0.016) and BigEarthNet (ECE 0.033).
6. **Strengths.** Scene and land-use recognition, spatial relations, and EO-style multiple-choice reasoning (LHRS-Bench, GEOBench-VLM overall) are clearly strong for an 8B zero-shot model.

## Pending

- Same-backbone control (Qwen3.5-9B) and no-image controls (batch 2, running).
- VRSBench-VQA-MC, xBD damage (ordinal, pre/post pairs) and xBD disaster type (batch 3, queued).
- DVL-Bench.
- Converters still to build:
  - Data downloaded or downloading: CHOICE, XLRS-Bench-lite, DisasterM3, UrBench, EarthMind-Bench, VHM honesty set, CDVQA test-2, RSVQA-HR test-2.
  - Blocked: fMoW-WILDS waits on the dataset's access gate (403).
