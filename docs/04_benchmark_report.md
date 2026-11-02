# 04 — Benchmarking `vllm-sr/d3-flash` on Earth observation: report

*2026-10-16. Zero-shot, public benchmarks only. Generated tables: [results/summary_tables.md](results/summary_tables.md). Per-benchmark details and running notes: [results/d3_flash_core.md](results/d3_flash_core.md). Setup validation: [results/phase0.md](results/phase0.md).*

## 1. Summary

d3-flash (8.4B, Qwen3.5-9B base) was run zero-shot on **21 EO benchmarks / ~145k questions**, against:
- the **same backbone without d3's decision training** (Qwen3.5-9B, identical prompt, image handling and letter-code readout);
- a **no-image** control;
- published numbers.

### Main findings

1. **Strong out of the box for an 8B model.** On the multiple-choice EO suites d3-flash matches or beats published GPT-4o numbers:
   - CHOICE perception / reasoning 85.1 / 79.1 (newly collected imagery);
   - GEOBench-VLM 51.9 (best published 41.7);
   - LHRS-Bench strict 77.5 (GPT-4-Turbo 44.6);
   - EarthMind optical / SAR / fused 67.5 / 54.2 / 66.0 (GPT-4o 64.1 / 47.8 / 61.1);
   - UrBench camera localization 50.0 (GPT-4o 35.3);
   - DisasterM3 relational reasoning 68.4 (GPT-4o 49.8).

   (Protocols differ, so these are context, not head-to-head comparisons.)
2. **The EO perception is inherited from the Qwen3.5 backbone.** Decision training adds only ~+1 to +4 points on most single-image benchmarks. Its real contributions are:
   - **reliability:** answers change under option reordering 2–4× less often; ECE roughly halves;
   - **geo-localization:** fMoW country +14.0, UrBench +11.0;
   - **consistency across option orders:** LHRS strict +7.5.
   - Multi-image sets are mixed: DVL +2.8 (single-choice) / +8.0 (multi-select exact-set), xBD type +2.5 (xBD labels), CDVQA −4.7. (Corrected 2026-10-21: an earlier version said DVL +16.6; see §5.)
3. **Clear bottlenecks, shared by both models (so they are backbone / perception limits, not readout limits):**
   - **counting:** 15–48% on every counting task, and *worse than no image* on two GEOBench counting tasks;
   - **quantitative change:** CDVQA "how much / which changed most" at or below chance;
   - **fine-grained agriculture and vegetation condition:** crop type, tree health and fire risk near chance; BigEarthNet agricultural / wetland classes AP < 0.12;
   - **low-resolution presence:** RSVQA-LR presence is worse *with* the image than without;
   - **SAR** (−13 to −20 vs optical, and fusion adds nothing);
   - **resolution:** ¼ resolution costs 9–15 points; extra tiles help counting and land use.
4. **Hallucination / abstention is a weakness that decision training made worse.**
   - On VHM's deceptive questions d3 abstains correctly only 2–19% of the time for color (base model: 21–43%) and 34–59% for position.
   - CHOICE hallucination detection: 77.6 vs the base model's 87.8.
5. **Calibration is good but task-dependent:**
   - under-confident on scene classification;
   - over-confident on counting, damage and color questions (ECE up to 0.58 on deceptive color).

6. **The information is often already there.** Linear probes on d3's own decision state reach 87–96% on AID / EuroSAT / NWPU where zero-shot gives 57–78% (vision features: 93–99%). For recognition, the bottleneck is the readout / label binding, not perception.

### What this means for fine-tuning (§7)

- **What to train on:** the weak skills above need **EO-specific supervision** (counting, change quantity, fine-grained LULC, SAR, abstention).
- **How:** a better readout or prompt alone won't fix them. Keep d3's decision-training benefits (consistency, calibration, geo-localization) as **regression tests**.

## 2. Setup

| | |
|---|---|
| Model | `vllm-sr/d3-flash` v3.1.0 (`521525effa`); transformers runtime shipped with the checkpoint |
| Hardware | A100-40GB. Parity with the official AMD MI325X run: 99.94% same top answer, median max \|Δp\| 0.0002 over 6,251 requests ([phase0.md](results/phase0.md)) |
| Prompt | P0: empty state (except where a benchmark needs it: image order / dates / sensor for multi-image items), the benchmark's question as instructions, its options as `choice` criteria |
| Question types | `choice` for MCQs; `noul` for yes/no and for each option of multi-select items (exact-set match also reported); `score` for ordinal (xBD damage, RSVQA-LR counts, VRSBench quantity) |
| Images | Native PNG/JPEG/WebP where possible. Non-native formats re-encoded to PNG. ≤1.6 MP enforced by the runtime; XLRS also tiled (overview + 4 quadrants) |
| Same-backbone control | Qwen3.5-9B with its own `lm_head` rows as the 255-code readout and causal attention, run through d3's runtime. Tokenizer, chat template and image processor are byte-identical to d3-flash's |
| Metrics | accuracy, macro-F1, AUROC (yes/no), QWK / MAE (ordinal), multi-label mAP / F1 / exact-set, ECE (15 bins), Brier, NLL; 95% bootstrap CIs over image groups |
| Code | `bench/` (converters, runner, metrics, probes, analysis), `slurm/run_d3.sbatch`, `scripts/` |

## 3. Results

Accuracy is question-weighted over each benchmark's slices. For yes/no questions it is accuracy at p = 0.5.

| Benchmark | Questions | d3-flash | Qwen3.5-9B | Δ | d3, no image | ECE d3 / Qwen | Published reference points |
|---|---:|---:|---:|---:|---:|---|---|
| Scene classification (UCM, AID, WHU-RS19, SIRI-WHU, EuroSAT-RGB, NWPU; full class list) | 20,205 | 71.2 | 67.4 | +3.8 | 5.8 | 0.06 / 0.16 | AID: GPT-4o 74.7, EarthDial 88.8; UCM: GPT-4o 88.8 |
| GEOBench-VLM (MCQ) | 3,211 | 51.9 | 50.4 | +1.6 | 20.5 | 0.10 / 0.18 | LLaVA-OV 41.7 (best in paper) |
| LHRS-Bench (4 option rotations pooled) | 2,392 | 78.5 | 76.0 | +2.4 | 40.8 | 0.02 / 0.06 | Strict (all rotations): d3 77.5 vs Qwen 70.0; GPT-4-Turbo 44.6 |
| CHOICE, perception / reasoning (newly collected imagery) | 7,407 / 2,000 | **85.1 / 79.1** | 84.1 / 74.8 | +1.1 / +4.3 | — | 0.03 / 0.04; 0.07 / 0.08 | Paper (mean of level-2 scores): Qwen2-VL-72B .77 / .71, GPT-4o .68 / .57 |
| VRSBench-VQA (closed-form conversion) | 5,000 | 71.8 | 70.1 | +1.7 | — | 0.07 / 0.11 | Not comparable (paper is GPT-judged) |
| RSVQA-LR | 10,004 | 60.1 | 59.1 | +0.9 | 59.5 | 0.12 / 0.16 | Language-only ≈ 74 (literature) |
| RSVQA-HR test-2 (presence + comparison) | 5,000 | 69.3 | 68.8 | +0.5 | — | 0.07 / 0.13 | Qwen3-VL-8B 66.0 / 75.4 |
| BigEarthNet v2 (19 labels; micro-F1 0.444 vs 0.418) | 19,000 | 75.7 | 75.5 | +0.2 | — | 0.03 / 0.08 | Gemini 2.5 F1 0.41–0.45 |
| EarthMind-Bench optical / SAR / fused | 1,810 each | 67.5 / 54.2 / 66.0 | 65.2 / 55.3 / 63.3 | +2.3 / −1.1 / +2.7 | — | | GPT-4o 64.1 / 47.8 / 61.1 |
| CDVQA test-2 | 5,001 | 43.9 | 48.6 | **−4.7** | — | 0.15 / 0.21 | TEOChat zero-shot 47.2 |
| xBD building damage (4 levels, balanced) | 3,000 | 41.1 (QWK 0.43) | 37.8 (QWK 0.31) | +3.3 | — | 0.17 / 0.13 | Chance 25 |
| xBD disaster type | 933 | 55.4 | 48.2 | +7.2 | — | 0.14 / 0.31 | |
| DisasterM3 (6 closed-form tasks) | 16,219 | 70.1 | 69.7 | +0.4 | — | 0.11 / 0.12 | GPT-4o: type 80.5, building-count 24.2, relations 49.8 |
| DVL-Bench (temporal, 4–10 images) | 8,415 | 63.0 | 46.4 | +16.6 pooled | — | 0.09 / 0.21 | Pooled over all questions, scoring each multi-select option as its own yes/no question, where the base collapses. On DVL's own metrics: single-choice 51.8 vs 49.0 (+2.8); multi-select exact-set 13.1 vs 5.1 (+8.0) |
| XLRS-Bench-lite, downscaled / overview + quadrants | 3,371 | 50.2 / 51.6 | 47.0 / — | +3.2 | — | 0.13 / 0.22 | Full XLRS (paper): Qwen2-VL 39.0, GPT-4o 32.2 |
| fMoW-WILDS OOD (region / country / land use) | 6,000 | 39.9 | 35.3 | +4.6 | — | 0.13 / 0.28 | Land use: GPT-4o 21.4 (d3 25.9) |
| UrBench geo-localization | 4,111 | 60.3 | 49.3 | **+11.0** | — | 0.06 / 0.20 | GPT-4o: image retrieval 85.9, camera localization 35.3 |
| VHM-HnstD honesty probes | 2,282 | 51.3 | 54.7 | **−3.4** | — | 0.18 / 0.20 | |

## 4. Capability map

| Capability | Evidence | Verdict |
|---|---|---|
| Scene / land-use recognition | Scene sets 57–92; GEOBench scene 83; EarthMind scene 98.7; XLRS land use 76–81 | **Strong** (weaker on 10 m Sentinel-2: EuroSAT 57) |
| Object category, color, shape (VHR) | VRSBench 80–83 | **Strong** |
| Spatial relations | GEOBench 81; EarthMind 80; DisasterM3 relations 68 (GPT-4o 50); XLRS 46–48 | Good on single objects; weaker in ultra-high-resolution scenes |
| Geo-localization / cross-view | UrBench image retrieval 77, camera localization 50; fMoW country 44 (16-way, same continent) | Good; clearly helped by decision training |
| **Counting** | GEOBench counting ≈33 (below Qwen2-VL / GPT-4o ≈40; no-image beats image on 2 tasks); XLRS 43–48; EarthMind 33–42; DisasterM3 building 26 / road 35; RSVQA-LR count bins 32 | **Weak: top bottleneck** |
| **Change quantity** | CDVQA "which changed most / least": 37 / 17.5; change % 5.1 (11 bins); DVL "when": 29 | **Weak** (detection of *whether* things changed is fine: 67–75) |
| **Damage grading** | xBD 41% / QWK 0.43; DisasterM3 "damaged objects" exact-set 3.9 | **Weak** |
| **Fine-grained vegetation / agriculture** | GEOBench crop type 29, tree health 24, fire risk 17.5; BigEarthNet ag / wetland AP < 0.12 | **Weak** |
| **SAR** | EarthMind SAR 54 vs optical 67.5; DisasterM3 SAR 52 vs optical 72; fused ≤ optical | **Weak**; SAR not used as complementary evidence |
| **Low-resolution / small objects** | RSVQA-LR presence: no image 69.5 > with image 62.9; EuroSAT 57; ¼ resolution −9 to −15 | **Weak** |
| **Abstention / hallucination** | VHM deceptive color 2–19% (base model 21–43%), deceptive position 34–59%; CHOICE hallucination detection 77.6 (base model 87.8) | **Weak, and regressed by decision training (two independent benchmarks)** |
| Resolution awareness | VHM GSD bins 38 (base model 54) | Weak, regressed |

## 5. What d3's decision training contributes (vs the same backbone)

- **Accuracy gains:**
  - **Per benchmark row (benchmark's own metric, results website):** d3 is ahead by more than 5 points on 8 of 31 rows, by 0.5–4.1 on 17, tied on 1 and behind on 5.
  - **Largest gains:** fMoW country +14.0 and UrBench +11.0 (geo-localization), DVL multi-select exact-set +8.0 (13.1 vs 5.1), LHRS strict +7.5, UCM +7.4, SIRI-WHU +5.9, DisasterM3 single-choice +5.4, AID +5.1.
  - **Multi-image sets are mixed, not uniformly larger:** DVL single-choice +2.8, xBD type +2.5 (xBD labels), xBD damage +3.3, EarthMind fused +2.7, CDVQA −4.7, UrBench +11.0.
  - **CHOICE sub-tasks:** spatial relationship +13.7, change detection +9.5, time property +8.0 (CHOICE overall +1.8).
  - Correction (2026-10-21): an earlier version listed DVL +16.6 as a multi-image gain. That figure pools DVL's 8,415 questions with each multi-select option scored as its own yes/no question, where the base model collapses; it is not DVL's own metric.
- **Accuracy losses:**
  - CDVQA −4.7, VHM −3.4, EarthMind-SAR −1.1.
  - **CHOICE hallucination detection −10.2.**
- **Consistency under option reordering** (3 permutations, 1,000-item subsets):

  | Subset | Items whose answer changes: d3 / Qwen | All orders correct: d3 / Qwen |
  |---|---|---|
  | Scene classification | 0–14% / 2–35% | |
  | GEOBench-VLM | 13.5% / 40.9% | 47.5 / 38.3 |
  | LHRS-Bench | 4.9% / 18.3% | |

- **Calibration:** ECE lower on 18 of 19 benchmarks above (the exception is xBD damage, 0.17 vs 0.13), often halved or better (e.g. scene 0.06 vs 0.16, UrBench 0.06 vs 0.20). Per website row: lower on 26 of 29 rows, halved or better on 13 (exceptions WHU-RS19, DVL single-choice, xBD damage).

Decision training is valuable for *reliability* (calibration, option-order consistency) and geo-localization, not for EO perception; its accuracy edge on most benchmarks is small.

## 6. Robustness and ablations (d3-flash)

| Test | Result |
|---|---|
| **Image rotation** (nadir imagery has no canonical up) | Single rotations ≈ unchanged (±1–2), but **correct under all four rotations**: scene 64.2 vs 71.5 single; GEOBench 40.4 vs 51.5. About a fifth of correct answers depend on orientation |
| **Resolution** (A1) | ½ resolution: −5.4 (scene) / −4.9 (GEOBench). ¼: −15.2 / −8.8 |
| **Tiles for ultra-high resolution** (XLRS V1 vs V0) | Counting +5.0, land use +4.9, multi-select land use exact-set 19.6 → 28.9; spatial relations −2.2 (cross-tile context lost) |
| **Metadata in the state** (A2: sensor, GSD; for BigEarthNet also country and date) | Scene: −0.3 (no effect). **BigEarthNet: micro-F1 +0.030, mAP +0.046.** Location and date help land-cover reasoning |
| **Option count** (A5; AID + NWPU, 460 items) | Full list (30 / 45 options) 74.6 → 16 options 86.1, 8 options 94.1, 4 options 97.0. Easier random distractors explain most of the gap, so long class lists are a real cost |
| **Permutation averaging** (A6, runtime flag) | +0.4 (scene and GEOBench). Not worth the 2× compute |

## 7. Bottlenecks and fine-tuning implications

| # | Bottleneck | Attribution (evidence) | Fine-tuning lever |
|---|---|---|---|
| 1 | Counting (objects, damaged buildings, roads) | Same in base and d3; extra resolution helps a little (XLRS tiles +5); image can be worse than prior | Counting supervision with ordinal targets (`score` + RPS loss); tile-based inputs; DOTA / DIOR / iSAID-derived count bins |
| 2 | Change quantity and timing | CDVQA quantity at chance, salience bias toward "buildings"; DVL "when" 29% | Bi-temporal training (SECOND / LEVIR-MCI / QFabric / DVL train) with area-fraction and timing targets |
| 3 | Fine-grained agriculture / vegetation and 10 m data | Same in base; metadata helps BigEarthNet (+0.046 mAP) | BigEarthNet / EuroSAT / crop datasets; metadata (location, date) in the state by default |
| 4 | SAR | −13 to −20 vs optical; fusion adds nothing | Optical–SAR paired data (EarthMind, BRIGHT, SARLANG train) with rendered SAR composites |
| 5 | Hallucination / abstention | Worse than the base on color; deceptive-position 34–59% | Explicit abstain options and absent-object questions in training (VHM-style); keep VHM-HnstD as a regression test |
| 6 | Orientation sensitivity | ~20% of correct answers flip under some rotation | Rotation / flip augmentation (cheap, label-preserving for nadir imagery) |
| 7 | Long label lists | 45-way −12 vs 16-way | Train with variable option counts and label definitions; or a shortlist at inference |

**Recipe direction** (from the literature review, §3):
- **Approach:** a JEV-style adapter (LoRA on the text tower, readout trained, vision frozen at first) with distribution supervision (CE / KL + Brier or RPS), option-permutation and rotation augmentation, replay of general decision data, and per-type temperature scaling.
- **The probes (§8) say that, at least for recognition, the vision tower can stay frozen.** A linear readout of d3's decision state already reaches 87–96% where zero-shot gives 57–78%, so start with an LLM-side adapter + readout. Probe counting / change / damage before deciding whether those need vision-side training.
- **Regression suite:** EarthMind-fused, UrBench, DVL, LHRS strict, the option-permutation tests and VHM-HnstD, to protect what decision training already provides.

## 8. Representation-vs-readout probes

Setup:
- **Features:** the **decision state** (the last-token hidden state that d3's readout scores, from the same prompt as zero-shot) and **pooled vision-tower features** (identical in both models: d3's vision tower is unchanged).
- **Probes:** a logistic-regression probe (C chosen on 10% of train) and a nearest-class-mean "prototype" readout, fit on train and scored on test.
- **Data:** AID 7k / 3k (test = the GeoChat 3k zero-shot set), EuroSAT 5k / 5.4k, NWPU 5k / 6.3k.

| Dataset | Zero-shot d3 / Qwen | Linear probe, decision state d3 / Qwen | Prototype, decision state d3 / Qwen | Linear probe, vision features | Prototype, vision |
|---|---|---|---|---:|---:|
| AID (30) | 78.1 / 73.0 | **96.4 / 96.4** | 69.8 / 86.8 | 98.6 | 94.9 |
| EuroSAT-RGB (10) | 56.8 / 55.7 | **94.3 / 94.0** | 69.6 / 76.7 | 97.4 | 89.1 |
| NWPU-RESISC45 (45) | 73.5 / 70.1 | **87.5 / 87.2** | 43.1 / 72.3 | 93.4 | 87.8 |

**Reading.**
1. **For recognition the bottleneck is the readout, not perception.** A linear readout of d3's own decision state beats its zero-shot answer by **+18 (AID), +37 (EuroSAT), +14 (NWPU)**. The vision features are near supervised state of the art (93–99). Fine-tuning the readout and an LLM-side adapter should recover most of this gap; the vision tower does not need unfreezing for these tasks.
2. **The language model loses a little** relative to raw vision features (−2 to −6, most on 45-class NWPU). Fine-grained many-class tasks may justify adapting the vision merger later.
3. **d3's decision state is linearly as informative as the base model's** (same probe accuracy), but **less clustered by class** (prototype 43–70 vs 72–87). Decision training reorganized the state toward answer codes, which is consistent with its sensitivity to label lists (A5) and to unseen vocabularies (RSAJEV).
4. **Not yet probed:** counting, change and damage, where the zero-shot gaps are largest. Extending the probes to DIOR count bins, CDVQA and xBD is the next diagnostic before deciding whether those skills need vision-side training.

## 9. Caveats

- **Protocol conversions:**
  - VRSBench (GPT-judged open answers), DVL / DisasterM3 / XLRS multi-select (per-option yes/no) and xBD (building crops) are our closed-form conversions; published numbers are context only.
  - Scene classification uses the full class list (GeoChat protocol for UCM / AID).
- **Contamination:** d3's training data is undisclosed. Classic sets (UCM, AID, NWPU, RSVQA) may be in Qwen3.5 pre-training. Conclusions lean on the newer sets (EarthMind, DVL, XLRS, CHOICE, UrBench) and on the no-image control.
- **Shared GPUs:** other users' processes shared some of our GPUs, so timings are noisy; correctness is unaffected (parity check).
- **Fixed issues:**
  - 55 DVL rows ran out of memory and were rerun.
  - XLRS / EarthMind row ids were non-unique: repaired by row-order mapping, verified chunk by chunk, no reruns needed.
  - One DisasterM3 item has an empty source image and was skipped.
- **Sampling:** several benchmarks are seeded, stratified subsamples (CDVQA 5k, DisasterM3 5k, VRSBench 5k, RSVQA-HR 5k, BigEarthNet 1k images, fMoW 2k).

## 10. Open EO specialists on the same questions

We ran the strongest open-weight Earth-observation models (docs/05_specialist_models.md) on exactly the rows d3-flash answered, in two ways: by option probabilities (as d3-flash is scored) and by writing their own answer. Each table shows the specialist's better of the two. "Trained on it" means the specialist was trained on that benchmark's training split. Metrics follow the rows of the benchmark guide, so some differ from §3's pooled numbers: LHRS-Bench is strict accuracy (right under all four option orders), and DVL and DisasterM3 are split into single-choice accuracy and multi-select exact-set match. Per-row details are in `runs/specialists/table.json` (`python -m bench.analysis.specialist_table`).

**Specialists ahead of d3-flash**

| Benchmark | Specialist | Specialist | d3-flash | Trained on it? |
|---|---|---:|---:|---|
| UCM | RSThinker | 94.2 | 82.9 | no |
| AID | RSThinker | 98.5 | 78.1 | yes |
| WHU-RS19 | RSThinker | 99.7 | 92.1 | no |
| NWPU | RSThinker | 97.0 | 73.3 | yes |
| EuroSAT | GeoZero | 77.7 | 57.3 | yes |
| RSVQA-HR | GeoZero | 87.9 | 69.3 | yes |
| RSVQA-LR | EarthDial | 76.9 | 60.1 | yes |
| VRSBench | RSThinker | 79.3 | 71.8 | yes |
| DVL single / multi | GeoChrono | 56.4 / 27.7 | 51.8 / 13.1 | yes |
| VHM deceptive | VHM-7B | 74.7 | 30.4 | yes |
| fMoW continent / country | Geo-R1 | 53.9 / 49.6 | 49.4 / 44.4 | no |
| xBD damage | TEOChat | 48.4 / QWK 0.56 | 41.1 / QWK 0.43 | yes |
| xBD disaster type (xBD's labels) | Qwen3-VL-8B xView2 LoRA | 87.8 | 70.0 | yes |

**d3-flash ahead**

| Benchmark | d3-flash | Best specialist |
|---|---:|---:|
| LHRS-Bench | 77.5 | 60.0 |
| CHOICE | 83.9 | 77.4 |
| DisasterM3 single / multi | 42.3 / 21.6 | 40.3 / 6.8 |
| UrBench | 60.3 | 55.6 |
| VHM honest | 71.3 | 67.8 |

Roughly tied: GEOBench-VLM, CDVQA, BigEarthNet and EarthMind optical.

**xBD disaster type uses xBD's own labels** since 2026-10-20 (fire, flooding, wind, earthquake, tsunami, volcano): this is the answer format of the dataset's metadata and of xBD-trained models.
- With these labels, d3-flash scores 70.0 and Qwen3.5-9B 67.5. Our earlier event labels (hurricane, wildfire, …, tornado) gave 55.4 and 48.2.
- Specialists in the common format: the xView2 LoRA 87.8, EarthDial 84.2 (written), TEOChat 74.9, TerraScope 39.9.
- Rows: `xbd/xbd_disaster_type_xbdlabels.jsonl`.

**Common question format.**
- Every model, d3-flash included, gets the same images, wording and lettered options; only each model's chat template differs.
- **Format check, not used for comparison:** the in-domain xBD models were also run in their own training format (`bench/specialists/native.py`) to make sure their scores are not an artefact of an unfamiliar prompt.
  - In that format they score higher: EarthDial damage 66.0, TEOChat type 96.8.
  - EarthDial's own-format test items reproduce its published disaster-type accuracy (97.5–98.8 vs 96.4).
  - The results stay in `runs/specialists/<model>/*-native/`.
