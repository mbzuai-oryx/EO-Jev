# 06 — Final benchmark suite: freeze proposal

*2026-10-19; decisions taken 2026-10-20 (§6). Status: **approved; row files being built**. This file and the row files it names become the frozen evaluation suite (eval-v1.0) used for every training run. Reference models per benchmark: §8.*

**Evidence base**
- *Adoption:* 41 RS-VLM papers whose results tables were read: 26 posted from Jul 2025, 18 peer-reviewed.
  - Full matrix: `scratchpad/bench_adoption.md`.
- *Recency and difficulty:* a sweep of RS benchmarks released mid-2025 to Oct 2026.
  - Every data link was checked live on 2026-10-19, and the annotation files were counted for the main candidates.
  - Full notes: `scratchpad/bench_hard_recent.md` (both scratch files are summarised here).
- *Catalog:* `02_benchmarks.md`. *Current results:* `04_benchmark_report.md`.

## 1. Summary

**Our current suite** (`02_benchmarks.md` §8.1, C1–C15) is broad and low-leakage, but it has three gaps.
1. **Visual grounding is missing.**
   - It is the #3 most-reported RS benchmark family: DIOR-RSVG appears in 18 of 41 papers, and grounding in 22 of 41.
   - Our plan had VRSBench-VG box-choice, but it was never run.
2. **Some benchmarks used for comparison are missing.**
   - *METER-ML:* 7 papers.
   - *Detection-dataset counting:* NWPU VHR-10 / DOTA-v2 in 9 papers.
   - *A second UHR set next to XLRS:* every UHR paper pairs XLRS with one.
3. **Too little hard headroom.**
   - Few of our sets are ones where strong 7–9B models sit near chance and humans near 90.
   - Such sets are released now: RSHR-Bench, Geo3DVQA, ChronoBench, VLRS-Bench, BigEarthNet.txt, TerraScope-Bench.

Several of our sets are **rarely reported by model papers** but are worth keeping for low leakage and their own large baseline tables:
| Benchmark | Papers reporting it (of 41) |
|---|---|
| CHOICE | 0 |
| GEOBench-VLM | 2 |
| CDVQA | 2 |
| EarthMind | 2 |
| DisasterM3 | 2 |
| DVL | 3 |
| HnstD | 3 |
| UrBench / RS geolocation | ≈1 |

They will not give head-to-head comparisons with specialist papers; the classic sets (AID, RSVQA-HR/LR, DIOR-RSVG, XLRS) are what gives those.

**Proposal**
- *Keep* all 15 current core sets. *Add* 10 to the core:
  - 4 for comparison: DIOR-RSVG + VRSBench-VG box-choice, METER-ML, detection counting, MME-RealWorld-RS.
  - 6 for hard headroom and new categories: RSHR-Bench, ChronoBench, Geo3DVQA, VLRS-Bench, BigEarthNet.txt, RS-Neg.
- Add **GeoMMBench** and **TerraScope-Bench** as core-secondary.
- Move geolocation to a "differentiator" role. Put UAV and agriculture behind a scope decision (§6).
- **Freeze rules (§5), including a training-exclusion list.**
  - Freezing the suite only protects us if no test image enters the fine-tuning data.
  - We have already found leaks: DisasterM3-Instruct contains 781 of our 933 xBD test scenes.

## 2. Selection principles

- **Every benchmark has a role.** A set stays in the core only if it serves at least one:
  - **A, Anchor:** the community reports it, so we can compare with published specialists. Its protocol is matched where possible.
  - **L, Low-leakage:** new or held-out imagery, so it gives an honest generalisation signal.
  - **H, Headroom:** strong 7–9B models are near chance or far below humans. Fine-tuning gains will show here, and ceiling effects won't hide them.
  - **C, Coverage:** it is the best released set for a capability category nothing else covers.
- **Format.** d3-flash only gives option probabilities, so every benchmark is closed-form (native or by a documented conversion). Generation-only tasks are declared out of scope (§4), so they are not silently missing.
- **Comparability.** Where a conversion changes the protocol, we report it as "not comparable to the published number". Examples:
  - box-choice vs Acc@0.5;
  - count bins vs exact counts;
  - closed-vocabulary VRSBench vs GPT-judged.
- **Independence.** Benchmarks sharing images do not count as independent evidence:
  - ChronoBench ↔ DVL;
  - RSHR ↔ XLRS;
  - TerraScope ↔ BigEarthNet / xBD;
  - Delta-QA ↔ CDVQA;
  - RS-Neg ↔ UCM / AID / VRSBench.

## 3. Category taxonomy (community-aligned)

The consensus top level, used by CHOICE, XLRS, RSHR, EarthMind-Bench, OmniEarth and the 2026 surveys, is **Perception / Reasoning / Robustness**. Temporal, multi-sensor and UHR are treated as **input conditions**. Our report will roll up to these three, with the categories below underneath.

Spell out category names, because abbreviations collide across benchmarks: OC means "object counting" in CHOICE but "object classification" in RSHR, and CD means "change detection" in one and "change description" in another.

Roles: **A** comparison anchor, **L** low-leakage, **H** headroom, **C** coverage. ✚ = new.

| Group | Category | Benchmarks in the frozen suite | Roles |
|---|---|---|---|
| **Perception: image** | Scene / land-use classification | UCM, AID, WHU-RS19, SIRI-WHU, NWPU-RESISC45 (leakage-flagged), EuroSAT, **METER-ML ✚**, fMoW facility type | A |
| | Multi-label land cover | BigEarthNet v2 (RGB) | A, C |
| | Image properties (modality, resolution, quality) | CHOICE image-level; VHM resolution / modality probes; *SenseBench (Tier 2)* | C |
| **Perception: instance** | Object presence and attributes (colour, shape, size, state) | RSVQA-HR / LR presence; VRSBench VQA (closed-vocabulary); MME-RW-RS colour | A |
| | Counting | GEOBench-VLM counting (1,054); RSVQA-LR count bins; **detection counting on NWPU VHR-10 / DOTA-v2 ✚**; MME-RW-RS count; DisasterM3 building counting | A, H |
| | Fine-grained object recognition | GEOBench-VLM aircraft / ship types; *FAIR1M crop choice (Tier 2, built by us)* | C |
| | Visual grounding (referring) | **DIOR-RSVG box-choice ✚**, **VRSBench-VG box-choice ✚** | A |
| | Spatial relations and position | VRSBench position / direction; XLRS spatial-relation subtasks; MME-RW-RS position; *EarthSpatialBench (Tier 2)* | A, H |
| **Input conditions** | Very-high / ultra-high resolution | XLRS-Bench-lite; **MME-RealWorld-RS ✚**; **RSHR-Bench ✚** | A, H |
| | Change and multi-temporal | CDVQA; DVL-Bench; **ChronoBench ✚**; *RSRCC (Tier 2)* | H, L |
| | Multi-sensor (SAR / multispectral) | EarthMind-Bench (optical / SAR / fused); DisasterM3 SAR; **BigEarthNet.txt ✚** (S1+S2 MCQ) | H, C |
| **Reasoning** | Disaster and damage assessment | xBD damage (ordinal) and type; DisasterM3 | H, C |
| | Quantitative and 3D (area, coverage, height) | **Geo3DVQA ✚**; **TerraScope-Bench ✚** (secondary) | H, C |
| | Complex reasoning and expert knowledge | CHOICE (reasoning branch); LHRS-Bench; GEOBench-VLM; **VLRS-Bench ✚**; **GeoMMBench ✚** (secondary) | L, H |
| | Geo-localisation and cross-view | UrBench geo-localization; fMoW continent / country (*differentiator: almost no RS paper reports geolocation*) | C |
| **Robustness** | Hallucination, honesty, negation | VHM HnstD; **RS-Neg ✚**; CHOICE hallucination detection | C |
| | Corruption robustness | *REOBench, SenseBench (Tier 2)* | C |

## 4. The frozen suite

### 4.1 Core: run on every checkpoint we report

**Kept from the current suite, all already run on d3-flash and the specialists:**

| ID | Benchmark | Rows (eval set) | Primary metric | Roles | d3-flash now |
|---|---|---|---|---|---|
| C1 | GEOBench-VLM (MCQ single) | 3,211 | accuracy | C, L | 51.9 |
| C2 | CHOICE | 9,407 | accuracy (perception / reasoning) | L | 83.9 |
| C3a | VRSBench VQA (closed-vocabulary) | 5k stratified | accuracy | A | 71.8 |
| C4 | RSVQA-HR test-2, RSVQA-LR | 5k + 10k | accuracy. Also report the GeoChat split (presence / comparison / rural-urban, no count / area) | A | 69.3 / 60.1 |
| C5 | LHRS-Bench | 690 × 4 orders | strict CircularEval. Also report a single order (ScoreRS and TinyRS use it) | A | 77.5 strict |
| C6 | Scene pack: UCM, AID, WHU-RS19, SIRI-WHU, EuroSAT, NWPU | ~20k | accuracy, full N-way | A | 57–92 |
| C7 | BigEarthNet v2 RGB | 1k images × 19 | micro-F1, mAP | A | 0.444 |
| C8 | CDVQA test-2 | 5k | accuracy | C | 43.9 |
| C9 | xBD damage (balanced) + disaster type | 3k + 933 | accuracy, QWK | H | 41.1 / 55.4 |
| C10 | DisasterM3 (closed-form) | 5k stratified | accuracy (single) / exact-set (multi) | H | 42.3 / 21.6 |
| C11 | EarthMind-Bench | 3 × 1,810 | accuracy | C | 67.5 / 54.2 / 66.0 |
| C12 | DVL-Bench | 5,854 | accuracy / exact-set | H | 51.8 / 13.1 |
| C13 | XLRS-Bench-lite (downscale + tiles) | lite set | accuracy (perception vs reasoning) | A, H | 48–49 |
| C14 | VHM HnstD | 1,642 | accuracy; deceptive and factual scored together | C | 71 / 30 |
| C15 | UrBench geo-localization + fMoW continent / country / facility | 4.1k + 2k | accuracy | C | 60.3 / 49 / 44 / 26 |

**Added: comparison anchors**

| ID | Benchmark | Rows | Conversion | Metric | Why |
|---|---|---|---|---|---|
| C3b ✚ | **VRSBench-VG** box-choice | 2k (unique / non-unique flag) | K lettered candidate boxes | accuracy (chance 1/K) | Grounding: 9 papers. Already planned |
| C16 ✚ | **DIOR-RSVG** box-choice | 2k from 7.5k test | same | accuracy (chance 1/K) | Grounding: #3 overall, 18 papers. **Not comparable to Acc@0.5** |
| C17 ✚ | **METER-ML** scene classification | VHM eval split | native N-way `choice` | accuracy | 7 papers (the VHM-split bundle) |
| C18 ✚ | **Detection counting** (NWPU VHR-10, DOTA-v2) | ~2–3k | exact count as `choice` over 0..K (or ±20 / 40% bins) | accuracy + expected-value MAE | 9 papers (RSThinker protocol). Counting is d3's #1 bottleneck |
| C19 ✚ | **MME-RealWorld-RS** | 3,738 | native 5-option MCQ; downscale + tiles | accuracy, with and without the reject option E (E is correct in only 14 items) | Second UHR anchor. **Qwen3.5-9B base is published at 62.1** (FaithRS, 2610.18369), so it directly measures what d3 training did to the backbone |

**Added: headroom and missing categories**

| ID | Benchmark | Rows | Fit | Hardness (7–9B open / best / human) | Category |
|---|---|---|---|---|---|
| C20 ✚ | **RSHR-Bench** (arXiv 2512.17319; ~70 MP images) | 1,932 MCQ | native; tile / downscale; rescale grounding options | ≈25 (chance) / 48.8 / **92.9** | UHR. Hardest released set. **Blind control mandatory**: text-only beats open VLMs on its reasoning split |
| C21 ✚ | **ChronoBench** (GeoChrono, ACM MM'26 per its README, unverified) | 17,689 (5k stratified for checkpoints) | multi-image (4–19 frames, 1024² each); multi-select as exact-set; 880 ordered-list items dropped or converted | Qwen3-VL-8B 39.2 / 57.5 / **92.3** | Long-term temporal. **Shares DVL's images** |
| C22 ✚ | **Geo3DVQA** (WACV'26) | 10,232 (subsample) | MCQ native; multi-label as per-label `noul`; numeric binned | 24–26 (≈chance) / 33.8 / — | 3D / height: the only released EO set. Needs GeoNRW imagery |
| C23 ✚ | **VLRS-Bench** (2026) | 1,500 closed | native (5-way; true/false; 8-option multi-select) | Qwen3-VL-8B 35.9 / 43.9 / — | Complex reasoning, multi-temporal (≤8 images) |
| C24 ✚ | **BigEarthNet.txt** (S1+S2) | 6,927 yes/no + 5,550 MCQ | rendered S2 + S1 | Qwen3-VL-8B 62.0 / 37.6 (best zero-shot) | Multi-sensor at 10 m; the only multispectral MCQ |
| C25 ✚ | **RS-Neg** (ECCV'26) | 9,205 yes/no + 5,675 MCQ (5k stratified) | native | Qwen3-VL-8B 71.5 / 57.3 | Negation / hallucination. Leaky image sources, so read it as a behaviour test |

**Added: core-secondary**, run on the final candidates rather than every checkpoint:

| ID | Benchmark | Rows | Notes |
|---|---|---|---|
| C26 ✚ | **GeoMMBench** (CVPR'26 Highlight) | 1,053 | Expert multi-sensor knowledge. **Shuffle options**: always answering "B" scores 40.3%. Many images are charts |
| C27 ✚ | **TerraScope-Bench** (CVPR'26) | 3,549 | Quantitative area / coverage / distance. 7–9B models score 36–43; a fine-tuned model reaches 68.9. Imagery overlaps BigEarthNet and xBD |

**Added: UAV / drone imagery (D2, core)**. These are low-altitude, often oblique images and video. They test the drone's own position, navigation and 3D layout as well as object perception:

| ID | Benchmark | Rows (closed-form) | Fit | 7–9B open / best / human |
|---|---|---|---|---|
| C28 ✚ | **MM-UAVBench** ([2512.23219](https://arxiv.org/abs/2512.23219)) | 5,702 MCQ | native; slight downscale (~1.7 MP) | Qwen3-VL-8B 51.0 / Qwen3-VL-32B 55.4 / 80.4 |
| C29 ✚ | **SpatialUAV** ([2606.27876](https://arxiv.org/abs/2606.27876)) | ~2,000 of 4,331 (option-letter tasks, plus clock direction, region and distance converted to `choice`) | adapted; downscale (up to 15 MP) | Qwen3.5-9B 42.5 / GPT-5.4 56.7 / 89.0 |
| C30 ✚ | **SIS-Bench** ("Self in Space", ACM MM'26) | 4,856 MCQ on 1,646 drone videos | adapted: video sampled to frames | Qwen3-VL-8B 64.6 / Gemini-3-Flash 71.6 / 91.7 |
| C31 ✚ | **SpatialSky-Bench** ([2511.13269](https://arxiv.org/abs/2511.13269), CVPR'26) | ~600 of 1,300 (colour, spatial relation, distance, height, counting, reverse pointing) | adapted; downscale from 5 MP | 7–8B models 12–47 per task / Sky-VLM (fine-tuned) 53 avg / — |

**Cost.** One full core run is about 190k question-forwards with the UAV sets, against about 115k today. A per-checkpoint "dev-core" subsample (§5) is about 27k.

### 4.2 Tier 2: extended, run at milestones or on demand

- **Resolution:** LRS-VQA (closed part), LRS-GRO (ZoomEarth), UHR-Micro.
- **Change:** RSRCC (region-level change, Google), Delta-QA MCQ, TEOChatlas change QA (S2Looking, QFabric, ABCD).
- **Spatial:** EarthSpatialBench yes/no and direction subsets (needs SatlasPretrain imagery), FIT-RSRC.
- **SAR:** SARLANG-1M SARVQA1 (closed types), SAREval (if the images can be obtained).
- **Fine-grained:** FAIR1M / MAR20 / ShipRSImageNet crop → type `choice`, built by us. This is the best available option for a category no released benchmark covers well.
- **Robustness:** REOBench, SenseBench, RSPOPE / DDFAV.
- **Broad MCQ:** OmniEarth (KDD'26), OmniEarth-Bench (report with and without "unable"; many inputs are plots).
- **Disaster:** MONITRS (Sentinel-2 series; blind control), BRIGHT (optical→SAR damage).
- **Cross-view:** CVSBench (CC-BY; shuffle options, because the letter distribution is skewed).

### 4.3 Out of scope for d3-flash, declared rather than silently absent

These are widely reported but need generated text, boxes or masks:
- **Captioning** (13 of 41 papers): RSICD, NWPU-Captions, RSITMD, UCM-Captions, Sydney.
- **Referring / reasoning segmentation** (12 of 41): RRSIS-D, EarthReason.
- **Change captioning:** LEVIR-MCI, MUDS, Dubai-CC.
- **Detection mAP.**

Optional proxies, each labelled "not comparable":
- caption-choice for captioning;
- mask-candidate choice for segmentation.

### 4.4 Watchlist: add when data appears

| Benchmark | Why it is wanted |
|---|---|
| LongEarth-Bench | 15–30 frame sequences |
| VertiCue-Bench / GeoHeight-Bench | height |
| RRS-10K | rare targets, low leakage |
| UAVQA-Bench | UAV |
| RSVideo-Bench | satellite video |
| XLRS full VQA | — |
| GROC | counting under clouds or at night |

## 5. Freeze rules

1. **Frozen row files.** Each benchmark's row file is fixed by:
   - row IDs;
   - option-order seeds;
   - count bins;
   - image transform (downscale method, tile layout, SAR/MS rendering);
   - prompt template.

   The suite is versioned `eval-v1.0`, with an md5 manifest. Any change makes a new version, and results across versions are not compared.
2. **Protocol variants fixed in advance** to match the literature:
   - LHRS strict + single order;
   - RSVQA full + GeoChat split;
   - MME-RW with and without E;
   - GeoMMBench and CVSBench with shuffled options;
   - VHM deceptive and factual combined;
   - XLRS and RSHR perception vs reasoning;
   - box-choice reported with chance level.
3. **Controls on every run:**
   - **blind** (no image) and **shuffled-image** controls on the core;
   - **option-permutation** consistency on the MCQ sets.

   A gain that also appears in the blind control is a prior, not perception.
4. **Training-exclusion list (most important).** Before any fine-tuning data is used, remove every test image of the frozen suite from it, by row ID and by perceptual hash. Known overlaps to purge:
   - DisasterM3-Instruct ⊃ 781 of our 933 xBD test scenes;
   - NWPU-RESISC45 and RSVQA-LR images, which sit in most RS instruction sets (GeoChat, VHM, LHRS, Falcon);
   - UCM / AID inside caption datasets;
   - ChronoInstruct / DVL train vs ChronoBench / DVL test: check the split is disjoint;
   - TEOChatlas xBD vs our xBD rows;
   - TerraScope-Bench's BigEarthNet / xBD sources.

   Training on a benchmark's *train* split is allowed, but the benchmark is then reported as **in-domain**, as we already do for the specialists.
5. **Dev vs test.** All checkpoint selection and hyperparameter tuning uses a **dev-core**: a fixed, stratified ~25k-question subsample taken from benchmark *validation* splits where they exist, and otherwise from rows held out of the reported test rows. Frozen test rows are scored only for reporting, never for selection.
6. **The regression set stays fixed.** It guards what decision training already provides: EarthMind-fused, UrBench, DVL, LHRS strict, option-permutation tests, and VHM HnstD (`04_benchmark_report.md` §7).

## 6. Decisions (taken by the user on 2026-10-20)

**Decided:** D1 yes · D2 **core** · D3 **left out** · D4 approved · D5 out of scope · D6 keep · D7 yes.

| # | Decision | Recommendation at the time |
|---|---|---|
| D1 | Include **visual grounding** as box-choice, even though it is not comparable to published Acc@0.5? | **Yes.** It is the #3 most-reported family, and leaving it out leaves a visible gap. We report it with chance level and a "not comparable" note |
| D2 | **Scope: UAV / drone imagery** (MM-UAVBench, SpatialUAV, SIS-Bench, SpatialSky-Bench)? | **Tier 2 only**, unless the product targets drones. These are oblique, low-altitude images, a different domain from satellite and aerial |
| D3 | **Scope: agriculture** (AgroMind, NeurIPS'25; filter to its overhead images)? | **Tier 2.** It does cover a known d3 weakness (crops and vegetation) |
| D4 | **Large downloads:** RSHR-Bench (210 GB), GeoNRW (Geo3DVQA), BEN v2 S1/S2 (BigEarthNet.txt), DOTA-v2 val (counting). DVL images are already local and serve ChronoBench too | **Approve.** All go to /nfs-stor |
| D5 | Declare **captioning / segmentation / detection mAP** out of scope, rather than building proxies? | **Out of scope**, with optional caption-choice later |
| D6 | Keep **geolocation** (UrBench, fMoW continent / country) in the core although the community barely reports it? | **Keep** as a differentiator. d3 is strong there and it is cheap |
| D7 | Accept **~175k question-forwards per full core run**, with a ~25k dev-core for checkpoints? | **Yes** |

**D8 (2026-10-22): report only the downscaled variants** of the ultra-high-resolution sets (XLRS-lite, MME-RealWorld-RS, RSHR-Bench): the whole image shrunk to 1.6 MP. The tiled variants (overview + four quarters) were run but are not reported; they added little (RSHR single-turn: d3 +0.6, Qwen3.5-9B −0.2, UHR-BAT2 −1.5).

## 7. Hardest benchmarks in the frozen suite (headroom ranking)

| Rank | Benchmark | 7–9B open models | Best | Human | Chance | d3-flash |
|---|---|---|---|---|---|---|
| 1 | RSHR-Bench (perception) | ≈25 | 48.8 | 92.9 | ≈25 | — |
| 2 | Geo3DVQA | 24–26 | 33.8 | — | ≈25 | — |
| 3 | DVL-Bench, multi-select exact-set | 5–14 | — | — | — | 13.1 |
| 4 | DisasterM3, multi-select damage (BBR / DSR) | 2–12, every model | — | — | — | 21.6 overall multi |
| 5 | ChronoBench | 36–39 | 57.5 | 92.3 | — | — |
| 6 | VLRS-Bench | 36 | 43.9 | — | — | — |
| 7 | BigEarthNet.txt MCQ | best zero-shot 37.6 | — | — | 25 | — |
| 8 | TerraScope-Bench | 36–43 | 68.9 (fine-tuned) | — | — | — |
| 9 | XLRS-lite, perception subtasks (counting, spatial relations, land use) | 20–40 | — | — | — | — |
| 10 | MME-RW-RS counting | 10–30 for most models | — | — | — | — |

DVL multi-select ranges over the BCA / CSE columns; ChronoBench's long-term-memory task is below 5% for open models.

d3-flash's own weakest current scores are DVL multi-select 13.1, DisasterM3 multi 21.6, fMoW facility type 26, VHM deceptive 30 and xBD damage 41.1. These line up with the capability bottlenecks in `04_benchmark_report.md` §7 (counting, change quantity, damage, SAR, abstention), which are exactly what fine-tuning targets.

## 8. Reference models per benchmark (user rule, 2026-10-20)

- **Used as published** (same items, format and metric): the best published result is enough for comparison. Cite it and flag it exact.
- **Adapted in any way:** we run the best model ourselves on our version, in the common question format (same images, wording and lettered options for every model). Adaptations include:
  - box-choice;
  - count bins;
  - downscaling or tiling;
  - a closed-form subset;
  - shuffled options;
  - relabelling;
  - video → frames.
- **Which "best model":** the best **open-weight** model, preferring RS specialists. Use a general open model when it is the best open option. Closed APIs cannot be run and are shown as "published, not reproducible".

Per benchmark (research 2026-10-20; details and sources in `scratchpad/best_models_new.md`; all model repos are public on HF):

**Can be cited as published**, if their row files keep the published protocol:

| ID | Benchmark | Condition | Best published (cite) |
|---|---|---|---|
| C17 | METER-ML (565 items, 6 classes) | VHM's prompt, class names in fixed order, scored over the names (not our lettered, shuffled converter) | RSCoVLM-7B 75.93 (open; cheap to run as well) |
| C25 | RS-Neg | full sets only (~15k); a 5k subsample would make it adapted | Qwen3-VL-32B 77.7 (yes/no) / 64.5 (MCQ), zero-shot |
| C27 | TerraScope-Bench | HF release has 5 of the paper's 6 tasks (no Building Change): cite per task and the 5-task mean | TerraScope 72.2 (5-task); best zero-shot EarthMind-4B 46.4 |

**Adapted: we run the reference model ourselves**:

| ID | Benchmark | Why adapted | Model to run (published score) | Runner-up / ≤12B |
|---|---|---|---|---|
| C3b, C16 | VRSBench-VG, DIOR-RSVG | box-choice | **RSThinker** (90.4 / 93.1 Acc@0.5), already set up | RSCoVLM-7B (84.55 DIOR-RSVG) |
| C18 | Detection counting (NWPU VHR-10, DOTA-v2) | count as choice; RSThinker's test items were not released, so we build ours the same way | **RSThinker** | GeoSolver |
| C19 | MME-RealWorld-RS | downscale / tiles | **RS-OPSD** (61.5; Qwen3-VL-8B base) | Qwen3.5-9B base (our control). The published "62.1" for it is doubtful, so we re-measure |
| C20 | RSHR-Bench | downscale / tiles; multi-turn history is teacher-forced (gold answers of earlier turns), which inflates the 342 multi-turn questions (d3 83% vs 44% single-turn), so **compare on the 1,590 single-turn questions** | **UHR-BAT2** (50.7 all / 45.66 single-turn, self-reported on its HF card; resized input) | GeoLens (48.8) |
| C21 | ChronoBench | 880 history-memory items need ordered letter lists; we drop or convert them | **GeoChrono** (78.3), already set up (cross-sequence frame grouping needs a fix) | Qwen3-VL-32B (46.7) |
| C22 | Geo3DVQA | numeric → bins, multi-label → per-label yes/no | **Qwen3.5-9B** (no RS specialist has results; best published open: Qwen2.5-VL-7B 26.4; the best fine-tune has no weights) | — |
| C23 | VLRS-Bench | closed-form subset | **ScoreRS** (≈52 on closed formats), already set up | Qwen3-VL-32B |
| C24 | BigEarthNet.txt | our S2 + S1 rendering (published models saw S2 RGB only): run both conditions | **EarthMind-4B multi** (57.8 / 35.3; best RS specialist), already set up | general best: Qwen3-VL-8B 62.0 / 37.6 (cite only) |
| C26 | GeoMMBench | shuffled options | **VHM-7B** (39.1; best RS specialist), already set up | general best: Qwen3-VL-30B-A3B 66.7 (cite only). The 88.4 "best" is a closed tool agent |
| C28 | MM-UAVBench | ~30% of images over 1.6 MP (up to 20 MP), downscaled | **Qwen3.5-9B** (no RS / UAV specialist has results) | general best: Qwen3-VL-32B 55.4 (cite only) |
| C29 | SpatialUAV | closed subset + downscale | **Qwen3.5-9B** (published 42.5; no RS / UAV specialist has results) | general best: Qwen3.6-27B 49.5 (cite only) |
| C30 | SIS-Bench | video → frames | **SIS-Motion** (76.9; custom code) | Qwen3-VL-8B (64.6) |
| C31 | SpatialSky-Bench | closed subset (official inference already caps at ~1.6 MP) | **Sky-VLM** (53.3; stock Qwen2.5-VL) | InternVL3.5-8B |

**Published, not reproducible** (no weights or closed): Earth-OneVision, SkyMoE, the Geo3DVQA fine-tune, RS-InternVL, EgoSIS, GeoMMAgent, and all closed APIs.

**User decision (2026-10-20):**
- Reference models are RS / UAV specialists only.
- Where no specialist has results, the reference is Qwen3.5-9B (our control): Geo3DVQA, MM-UAVBench, SpatialUAV.
- Better general models are cited only.
- DOTA-v2 counting uses the 458 locally available validation images (the other 135 are on Baidu only).

**Corrections from this check:**
- ChronoBench has 4–19 frames per item, all 1024², so none need resizing.
- MM-UAVBench's downscale is not slight.
- Geo3DVQA's GeoNRW RGB tiles still need downloading.

## 9. Row files built (eval-v1.0 additions, 2026-10-21)

**Notes and reports:** converter reports are in `$EOJEV_DATASETS/notes/converters_{uhr,temporal_sensor,perception,uav}.md`; the shared spec is `notes/converter_spec.md`.

**Status:** all files pass `validate_row`, keep every image within d3's 65,536–1,638,400 px, and are deterministic: a re-run gives byte-identical files. Paths are under `$EOJEV_DATASETS/converted/`.

| ID | File(s) | Rows | Status |
|---|---|---|---|
| C3b | `grounding/vrsbench_vg_choice` | 2,000 (1,000 unique / 1,000 non-unique) | adapted: box-choice, K=4 |
| C16 | `grounding/dior_rsvg_choice` | 2,000 (official test split) | adapted: box-choice, K=4 |
| C17 | `meter_ml/meter_ml_vhm` | 565 | as-published (VHM prompt) |
| C18 | `det_count/det_count_{nwpu_vhr10,dota_v2_val}` (+ `det_count_vhr10_absent`, reported separately) | 888 + 1,500 questions (+ 300) | adapted: RSThinker protocol, count as `choice` 0..K |
| C19 | `mme_rw_rs/mme_rw_rs_{v0_downscale,v1_tiles}` | 3,738 each | adapted: resolution |
| C20 | `rshr/rshr_{v0_downscale,v1_tiles}` | 1,932 each | adapted: resolution, box coordinates rescaled |
| C21 | `chronobench/chronobench` (+ `chronobench_5k`) | 16,809 (5,000) | adapted: 880 ordered-list items dropped |
| C22 | `geo3dvqa/geo3dvqa` | 10,232 | adapted: bins, per-label yes/no, drawn overlays |
| C23 | `vlrs/vlrs_closed` | 1,500 | adapted: fill-in dropped |
| C24 | `bigearthnet_txt/bigearthnet_txt_{s2rgb,s2s1}` | 12,477 each | adapted: rendering, two conditions |
| C25 | `rs_neg/rs_neg_{vqa,mcq,ucm,aid}_negated` (+ `rs_neg_5k`) | 9,205 / 5,675 / 2,100 / 3,000 | full sets as-published; 5k adapted |
| C26 | `geommbench/geommbench_test` (+ `_val`) | 1,016 (37) | adapted: seeded option shuffle |
| C27 | `terrascope_bench/terrascope_bench` | 3,549 | as-published (5 released tasks) |
| C28 | `mm_uavbench/mm_uavbench` | 5,702 | adapted: downscale, boxes redrawn, video at 3 fps |
| C29 | `spatialuav/spatialuav_closed` | 2,356 | adapted: closed subset, downscale |
| C30 | `sis_bench/sis_bench` | 4,852 (32 frames each) | adapted: video → frames (released-code rule) |
| C31 | `spatialsky/spatialsky_closed` | 600 | adapted: closed subset |

**Defaults taken in the main session on the converters' open points:**
- Option text repeated within an item: the letter is appended and the item flagged; it still counts toward the score.
- RSHR box coordinates are rescaled in the text, not drawn.
- RSHR multi-turn: earlier turns are given with their correct answers.
- SpatialSky's 19-way superlative items are reported separately from the 2-way pairs.
- MM-UAVBench: boxes are redrawn after the downscale.
- METER-ML and RS-Neg scene classification: the duplicated class list is accepted, because d3's runtime renders its own options.
- Box-choice is always reported with its chance level.
- Overlapping evidence: C13 / C19 / C20 (shared DOTA and XLRS images) and C12 / C21 (the same NAIP frames).
