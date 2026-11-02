# Open-weight EO specialist models for the SOTA re-measurement

Checked 2026-10-17 by two literature passes. Every HF repo below was confirmed through `huggingface.co/api/models/<id>`.

**Goal:** replace the uncertain SOTA cells on the website with numbers measured on **our exact row files**:
- same images, options and conversions;
- both answer modes: option probabilities and generated answers;
- full sets.

User decision (2026-10-17): **EO-specialist models only** (no general open models, no closed APIs).

## Per-benchmark shortlist (best first; open weights only)

| Our benchmark | Specialists | Train-split overlap with the benchmark |
|---|---|---|
| Scene: UCM, AID, WHU-RS19, SIRI-WHU, EuroSAT, NWPU | RSThinker `minglanga/RSThinker`; GeoZero `hjvsl/GeoZero` (subfolder `GeoZero-8B-without-RFT/`); RSCoVLM `Qingyun/RSCoVLM-7B-2512` | RSThinker: AID, NWPU. GeoZero: AID, NWPU, WHU, EuroSAT. RSCoVLM: NWPU, UCM |
| BigEarthNet v2 | EarthMind `sy1998/EarthMind-4B`; EarthDial `akshaydudhane/EarthDial_4B_RGB`; TerraScope `sy1998/TerraScope` | all three |
| GEOBench-VLM | MLRS `INSAIT-Institute/MLRS` (50.3); EarthDial; GeoZero | — |
| CHOICE | GeoZero (no-RFT checkpoint; its paper reports no CHOICE score, the 71.7 cited here earlier was unsourced); VHM `FitzPC/vhm_7B` | — |
| LHRS-Bench | ScoreRS `PumpkinCat/Qwen2VL-7B-RS-R1` (69.2; same group as the benchmark); ZoomEarth `HappyBug/ZoomEarth-3B` | — |
| VRSBench | GeoSolver `minglanga/GeoSolver` / RSThinker; EarthMind | all |
| RSVQA-LR | EarthDial (93.1); RSCoVLM; TEOChat `jirvin16/TEOChat` | all |
| RSVQA-HR | LHRS-Bot-Nova `LHRS/LHRS-Bot-Nova` (custom code); GeoZero; RSCoVLM | all |
| CDVQA | GeoChrono `Davidup1/GeoChrono` (59.2); EarthDial (52.1); TEOChat (47–50); TerraScope (unreported) | — (GeoChrono and TEOChat are zero-shot here) |
| DVL-Bench | GeoChrono (far ahead); EarthDial (30.3) | GeoChrono trains on DVL-train-derived data |
| xBD damage / disaster type | TEOChat; TerraScope; EarthDial | all three trained on the xBD **train** split; our questions use the **test** split (same events, different images) |
| DisasterM3 | TerraScope (damage counting 54.1, vs TEOChat 22.5); EarthDial; TEOChat | TerraScope trains on xBD train |
| EarthMind-Bench | EarthMind `sy1998/EarthMind4B_multi` (fused) / `EarthMind-4B`; TerraScope; EarthDial | EarthMind: same source datasets, train splits |
| XLRS-Bench | Text-Before-Vision `initiacms/Text-Before-Vision` (60.4); GeoEyes `initiacms/GeoEyes` (54.2); GeoLens `NaOHjiang/GeoLens` (54.2); GeoLLaVA-8K `initiacms/GeoLLaVA-8K` (51.5) | same author group as the benchmark |
| fMoW-WILDS | Geo-R1 `miniHui/Geo-R1` (geolocation); TEOChat / EarthDial / GeoZero (facility type) | TEOChat, EarthDial, GeoZero, VHM, LHRS-Bot-Nova all trained on fMoW; the OOD-year images may be in their training data |
| UrBench | MLRS (51.9); Geo-R1 | MLRS: cross-view CVG-Text |
| VHM honesty | VHM-7B; GeoZero | VHM trained on the HnstD train split |

**Strong models with no released weights** (quote their numbers only; they cannot be run):
- Earth-OneVision (2606.10819): trains on many of our test sources.
- ScaleEarth (2605.07562): XLRS 59.4.
- GeoVLM-R1, DVLChat, the DisasterM3 fine-tunes, SkyNative, EarthGPT-X, SkyMoE.
- GeoZero-with-RFT: RFT trains on each benchmark's train split.

**General models beat the specialists** on DisasterM3 (base Qwen2.5-VL-7B 31.2 vs TEOChat / EarthDial ~23), UrBench, LHRS-Bench and zero-shot scene classification (2026 survey 2607.20284). On those rows, the specialist column is a specialist reference, not the absolute ceiling.

## Proposed run set

### Phase 1: covers every benchmark, 10 checkpoints in 6 environments

| Environment | Models |
|---|---|
| Stock transformers + vLLM | GeoZero (Qwen3-VL-8B), RSThinker (GLM-4.1V-9B), RSCoVLM (Qwen2.5-VL-7B), ScoreRS (Qwen2-VL-7B), Geo-R1 (Qwen2.5-VL-7B) |
| Sa2VA (`transformers==4.42.3`) | TerraScope, EarthMind-4B, EarthMind4B_multi |
| EarthDial (`transformers==4.37.2`) | EarthDial_4B_RGB |
| TEOChat (`transformers==4.31.0`) | TEOChat |
| GeoChrono (py3.12, torch 2.8) | GeoChrono LoRA on Qwen3-VL-4B |
| VHM (opendatalab/VHM) | VHM-7B |

### Phase 2: harder engineering, narrower coverage

| Model | Benchmark it serves | Why it is harder |
|---|---|---|
| Text-Before-Vision, GeoEyes, GeoLens | XLRS | Need vLLM zoom-tool loops; generated mode only; native resolution. Text-Before-Vision ships as FSDP shards, so it must be converted to HF format first |
| GeoLLaVA-8K | XLRS | Needs the LongVA codebase |
| MLRS | GEOBench, UrBench | InternVL3.5 LoRA plus the gated `facebook/sam3` |
| LHRS-Bot-Nova | RSVQA-HR only | Custom codebase |

## Validation

Before trusting a model's numbers, reproduce one published number where our benchmark matches the paper's protocol:

| Model | Published number to reproduce |
|---|---|
| EarthMind | EarthMind-Bench 61.7 / 61.3 / 70.0 |
| GeoZero | UCM 93.8 (no-RFT checkpoint, arXiv 2511.22645); not CHOICE, which the paper does not report |
| RSThinker | UCM 92.1 |
| GeoChrono | CDVQA 59.2 (test 1) |
| TEOChat | CDVQA ~47 |

A large gap means the prompt adapter is wrong, not the model.

Sources: specialist papers listed above; survey arXiv 2607.20284; the per-model HF pages.
