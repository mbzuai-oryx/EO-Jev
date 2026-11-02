# 02 — EO benchmark catalog for `vllm-sr/d3-flash`

*Compiled 2026-10-16. "Cites" are Semantic Scholar counts on that date (Google Scholar is typically 1.3–2× higher). [M] marks details taken from a secondary source or memory that were not re-verified. HF IDs in the core suite were checked as live on 2026-10-16.*

## How to read the "fit" column

d3-flash only outputs a probability distribution over lettered options. It has no text generation, boxes or masks. Each benchmark is rated:

| Fit | Meaning |
|---|---|
| **Native** | Already MCQ / yes-no / classification / ordinal. Run as-is. |
| **Easy** | A mechanical conversion, e.g. closed-vocabulary VQA → `choice`, multi-label → one `noul` per label, counts → bins → `score`/`choice`. |
| **Hard** | Needs resolution surgery (>1.6 MP), non-RGB rendering, or box/mask → candidate-choice conversion. |
| **S2** | Needs generation (captions, open answers, coordinates). Out of scope for d3-flash, or only via a "pick the right caption" proxy. |

### Conversion recipes used below

- **N-way:** d3 accepts up to 255 options. For datasets with more than 16 classes, the primary metric is the full N-way `choice`. Also report a 16-way variant (GT + 15 seeded distractors) and label it "16-way"; it is not comparable to N-way accuracy.
  - If full N-way turns out to be out-of-distribution for d3 (ablation A5), switch to a **shortlist**. Ask one `noul` per class in a single request, rank the classes by P(yes), then ask a `choice` over the top 8–16. A GT missing from the shortlist counts as wrong. This keeps the result a true N-way top-1.
- **Per-label `noul`:** for multi-label sets. Gives mAP / F1 from P(yes).
- **Bins:** count or area → ordinal bins as `score`, or 5-option MCQ with ±20% / ±40% distractors (the GEOBench-VLM / DisasterM3 convention).
- **Box-choice:** draw K lettered candidate boxes on the image: the GT plus same-class / other-referent distractors with IoU < 0.3. Ask a `choice`. Always report chance (1/K). This is **not comparable to Acc@0.5**.
- **Caption-choice:** pick the true caption among K, with negatives from *other* scene classes, since in-class captions are near-duplicates.
- **Render:** multispectral → true colour (B4/B3/B2) plus false colour (B8/B4/B3), and NDVI/NDWI as extra grayscale images. SAR → VV/VH/ratio in dB, stretched to the 2–98th percentile. Keep the rendering fixed and documented.

---

## 1. Multi-task EO VLM benchmarks (VQA / reasoning suites)

| Benchmark | Venue | Test set | Imagery | Native format · metric | d3 fit | Reference scores | Access · license · cites |
|---|---|---|---|---|---|---|---|
| **GEOBench-VLM** ([2411.19325](https://arxiv.org/abs/2411.19325)) | ICCV'25 | >10k instr. / 31 tasks; **3,211 MCQ in the HF "single" split** (1,054 of them counting) | ~40 sources (xBD, DOTA, FAIR1M, fMoW, So2Sat, PASTIS, FireRisk, …), 49 px–1.5k px | 5-option MCQ, plus bbox / mask / caption subtasks | **Native** (MCQ); box tasks → box-choice | Best overall: LLaVA-OV 41.7, GPT-4o slightly lower. Scene: EarthDial .771, GPT-4o .711, Qwen2-VL .676. Counting: LLaVA-OV .438, Qwen2-VL .402, GPT-4o .397, EarthDial .363, GeoChat .292 | `aialliance/GEOBench-VLM` · Apache code; xBD images separate · 72 |
| **CHOICE** ([2411.18145](https://arxiv.org/abs/2411.18145)) | NeurIPS'25 D&B | 10,507: **9,407 MCQ** over 21 tasks; 600 grounding; 500 RES | **Newly collected** (GEE, Landsat-8 FC, S1 SAR, S2, SDGSAT night-light), 0.1–30 m, 512–800 px; change pairs | 2–4-option MCQ, hierarchical (perception / reasoning) | **Native** | Perception / reasoning: Qwen2-VL-72B .77/.71, InternVL2-40B .73/.72, GPT-4o .68/.57, GeoChat .53/.48 | `An-Xiao/CHOICE` · MIT repo, imagery academic · 26 |
| **LHRS-Bench** ([2402.02544](https://arxiv.org/abs/2402.02544)) | ECCV'24 | 690 Qs / 108 images | Google Earth | Single-choice, 2–4 options, each asked 4× with shuffled options (strict) | **Native** | Strict: GPT-4-Turbo 44.6, LHRS-Bot 39.4. Lenient ([2607.20284](https://arxiv.org/abs/2607.20284)): GPT-4o 67.8, Qwen3-VL 64.1, GeoChat 37.2 | `PumpkinCat/LHRS_Data` · MIT · 221 |
| **VRSBench** ([2406.12384](https://arxiv.org/abs/2406.12384)) | NeurIPS'24 D&B | Test: 9,350 imgs; **37,408 VQA**; 16,159 referring exprs; 9,350 captions | DOTA-v2 + DIOR, 512² | VQA: short open answers, GPT-4-judged (10 types). VG: Acc@0.5 | **Easy**: presence → `noul`; category / colour / shape / direction → `choice`; quantity → bins. VG → box-choice (has an `is_unique` flag) | VQA: GPT-4V 65.6, GeoChat zero-shot 40.8 (fine-tuned 76.0). VG @0.5: Qwen3-VL 53.3, GeoChat-FT 49.8, GPT-4V 5.1 | `xiang709/VRSBench` · CC-BY-4.0 tag vs NC text · 175 |
| **XLRS-Bench** ([2503.23771](https://arxiv.org/abs/2503.23771)) | CVPR'25 | 1,400 imgs; 32,389 VQA / 16 subtasks (lite subset on HF) | **~8,500² px avg (~72 MP)** | 4-option MCQ | **Hard** (resolution only; format native) | Qwen2-VL 39.0, LLaVA-OV 38.9, GPT-4o 32.2, GeoChat 22.0; humans >90. Counting: GPT-4o 24.3 | `initiacms/XLRS-Bench-lite` · CC-BY-NC-SA · 67. Also in VLMEvalKit / lmms-eval |
| **MME-RealWorld (RS subset)** ([2408.13257](https://arxiv.org/abs/2408.13257)) | ICLR'25 | 1,298 imgs / 3,738 QA (count / colour / spatial) | FAIR1M, Potsdam, …; up to >5000² | 5-option MCQ (E = "reject") | **Hard** (resolution) | Qwen2-VL 44.8, InternVL2 39.4, GPT-4o 28.9 | `yifanzhang114/MME-RealWorld` · academic · 322 |
| LRS-VQA ([2503.07588](https://arxiv.org/abs/2503.07588)) | ICCV'25 | 1,657 imgs / 7,333 QA | avg 7,099×6,329, up to 27k px | open ≤3 words | Hard | Qwen2-VL 34.9, GPT-4o 27.6 | `ll-13/LRS-VQA` · NC · 44 |
| RSHR-Bench ([2512.17319](https://arxiv.org/abs/2512.17319)) | arXiv'25 | 5,329 imgs / ~3.9k VQA | avg 8,700×8,065 | MCQ A–D + open | Hard | Gemini-2.5-Pro 34.9/45.5 (perception / reasoning) | GH Yunkaidang/RSHR · 14 |
| **VLEO-Bench** ([2401.17600](https://arxiv.org/abs/2401.17600)) | 2024 | Landmarks 602, PatternNet 1k, fMoW-WILDS 2k, BigEarthNet 1k, counting (COWC 1k / NEON 194 / animals 112), xBD 933, RSVG, RSICD 1,009 | mixed | MC, multi-label F1, MAE, IoU | Mixed: landmarks (geo-recognition, 5-way) and PatternNet **Native**; counts → bins | GPT-4V: landmarks .671, PatternNet .73, fMoW .19, BEN F1 .38, RSVG .076 | `mit-ei/vleo-benchmark-datasets` · 61 |
| **UrBench** ([2408.17267](https://arxiv.org/abs/2408.17267)) | AAAI'25 | 10.5k Qs / 14 tasks | satellite + street view, multi-view | MC + open | **Native** (MC part, multi-image) | GPT-4o 61.2, InternVL2-8B 48.8; human 69.9 | opendatalab · 54 |
| **OmniEarth** ([2603.09471](https://arxiv.org/abs/2603.09471)) | KDD'26 | 9,275 imgs / 44,210 instr. / 28 tasks | JL-1, S1/S2, Capella; 0.05–30 m | 2–6-option MCQ + box / mask / open; text-only blind test; 5-paraphrase consistency | **Native** (MCQ) | Scene: Gemini-2.0-Flash 71.3, GPT-4o 65.8, EarthDial 58.6, GeoChat 43.9 | `sjeeudd/OmniEarth` · Apache-2.0 · 4 |
| OmniEarth-Bench ([2505.23522](https://arxiv.org/abs/2505.23522)) | 2025 | 29,855 (27,082 MCQ), 109 tasks, 6 Earth spheres | ERA5, MODIS, S1, CryoSat… **many inputs are rendered plots** | 5-option MCQ (E = "unable") | Native | InternVL3-72B 33.3, GPT-4o 11.2 (collapse onto reject) | `initiacms/OmniEarth-Bench` · 15 |
| **EarthMind-Bench** ([2506.01667](https://arxiv.org/abs/2506.01667)) | 2025 | 2,841 optical–SAR pairs, 10 tasks | BEN-MM, OpenEarthMap-SAR, DFC23, WHU-OPT-SAR, MSAW | MC + 0–5-scored open + masks | **Native** (2 images) | MC opt / SAR / fused: GPT-4o 64.1 / 47.8 / 61.1; EarthMind-4B 61.7 / 61.3 / 70.0; GeoChat 35.5 / 34.3 | GH shuyansy/EarthMind · 17 |
| **VLRS-Bench** ([2602.07045](https://arxiv.org/abs/2602.07045)) | 2026 | 2,000 QA / 14 tasks; ≤8 temporal images | LoveDA, Potsdam, DOTA, FAIR1M, xView2, SECOND, SpaceNet7 | single-choice (5), multi-choice (8), T/F, fill-in | **Native** (single / TF); multi → per-option `noul` | GPT-5.4 .439, Gemini-3.1-Pro .436, Qwen3-VL-32B .395, **Qwen3-VL-8B .359**, GPT-4o .361 | GH MiliLab/VLRS-Bench · 5 |
| **VHM HnstD** ([2403.20213](https://arxiv.org/abs/2403.20213)) | AAAI'25 | presence 242 (Y/N), abs / rel position 400 each (5-way), colour 600 | — | honesty / hallucination probes | **Native** | Deceptive absolute position: VHM 90.7 vs LLaVA-1.5 12.0 | GH opendatalab/VHM · NC · 122 |
| FIT-RSRC ([2406.10100](https://arxiv.org/abs/2406.10100)) | 2024 | size [M] | DOTA-derived | relation single-choice, CircularEval | **Native** | Qwen3-VL 64.3, GPT-4o 55.8, GeoChat 21.9 | `ll-13/FIT-RS` · CC-BY-NC · 132 |
| **GeoChat-Bench** ([2311.15826](https://arxiv.org/abs/2311.15826)) | CVPR'24 | UCM 2,100; AID 3,000; LRBEN; HRBEN; referring 7,653; grounding 758 | — | JSONL | Classification N-way; RSVQA native; grounding → box-choice | GeoChat UCM 84.4, AID 72.0, LR 90.7, HR 72.3 | `MBZUAI/GeoChat-Bench` · Apache · 586 |
| EarthDial eval suite ([2412.15190](https://arxiv.org/abs/2412.15190)) | CVPR'25 | 44 downstream sets (RGB / SAR / MS / temporal), not packaged | — | — | Native for ≤16-class sets | EarthDial vs GPT-4o: AID 88.8/74.7, BEN-RGB 68.8/49, So2Sat-MS 60.7/15.5 | GH hiyamdebary/EarthDial · 117 |
| TEOChatlas eval ([2410.06234](https://arxiv.org/abs/2410.06234)) | ICLR'25 | fMoW, xBD, S2Looking, QFabric (2–5 imgs), ABCD, CDVQA | 224² | change QA, classification, detection F1 | **Native** `noul`/`choice` | xBD damage F1: TEOChat 50.0, GPT-4o 38.3, Qwen2-VL 17.3 | `jirvin16/TEOChatlas` · Apache tag, xBD NC · 88 |
| MSEarth ([2505.20740](https://arxiv.org/abs/2505.20740)) | ACL'26 | 2,784 four-option MCQ | **figures from Earth-science papers, not EO imagery** | MCQ | Native (report separately) | GPT-4o 58.0, Qwen2.5-VL-7B 44.8 | HF MSEarth · CC-BY · 5 |
| AVI-Math ([2509.10059](https://arxiv.org/abs/2509.10059)) | ISPRS-J'25 | 3,773 Qs / 360 UAV imgs (4000×2250) | UAV | 36% MC, 6% T/F, 58% numeric | MC / TF Native after downscale; numeric S2 | GPT-4o 33.5, Qwen2.5-VL-7B 27.9 | `erenzhou/AVI-Math` · Apache · 4 |
| Landsat30-AU ([2508.03127](https://arxiv.org/abs/2508.03127)) | AAAI'26 | 17,725 VQA | Landsat 30 m | format [M] | probably Easy | Qwen2.5-VL-7B .74 | GH · 5 |
| EarthSpatialBench ([2602.15918](https://arxiv.org/abs/2602.15918)) | 2026 | 325k QA / 21k tiles (**not yet released**) | Satlas ~1 m | Y/N, 8-way direction, numeric, bbox | Native when released | GPT-5 .86 (binary distance) | — |
| RSIEval ([2307.15266](https://arxiv.org/abs/2307.15266)) | 2023 | 936 open QA, 100 captions | DOTA | human-scored | Hard; small, gated | — | form · 375 |
| RSVLM-QA ([2508.07918](https://arxiv.org/abs/2508.07918)) | ACM MM'25 | 162k QA / 13.8k imgs | WHU, LoveDA, INRIA, iSAID | GPT-judged sentences | presence / quantity Easy; rest S2 | Ovis2 67.0 | GH · 23 |

## 2. Classic RS VQA, change, disaster and temporal

| Benchmark | Test set | Imagery | Format | d3 fit + recipe | Reference scores / notes | Access · license |
|---|---|---|---|---|---|---|
| **RSVQA-LR** ([2003.07333](https://arxiv.org/abs/2003.07333), TGRS'20, 433 cites) | ~7k (GeoChat protocol, no count / area); full test 10,004 (count 2,947 with raw integers; bin yourself) | S2 RGB 10 m, 256² | presence / comparison Y/N; rural / urban; count 5 bins | **Native** (`noul`, `choice`) | Presence / comparison / R-U: EarthDial 92.6/92.8/94.0, GeoChat 91.1/90.3/94.0, InternVL2-8B 58.5/72.3/71. **Leaky:** train split is in every RS-VLM; text-only scores 73.8% | Zenodo 6344334 · CC-BY-4.0 |
| **RSVQA-HR** | test-2 (Philadelphia, unseen sensor): ~47k presence + comparison | USGS 15 cm, 512² | Y/N; area bins | **Native** | GeoChat 58.5/83.2; EarthDial 72.5 avg; TEOChat 67.5/81.1; **Qwen3-VL-8B 66.0/75.4**; GPT-4o 67.9 avg | Zenodo 6344367 · CC-BY-4.0 |
| RSVQAxBEN | ~3.4M test QA (subsample) | BEN S2 120² | 81% Y/N, rest LC lists | Native / per-label `noul` | classical ~75% | Zenodo 5084904 |
| **CDVQA** ([2112.06343](https://arxiv.org/abs/2112.06343), TGRS'22, 76) | test1 39,686 / test2 31,036 QA (968 pairs) | SECOND bi-temporal 512² | **19-answer closed set** | **Native**: 2 images; `noul`, `choice`(6 classes), ratio bins (11) | baseline ~69 OA; TEOChat zero-shot 47.2 | GH YZHJessica/CDVQA · SECOND academic |
| EarthVQA ([2312.12222](https://arxiv.org/abs/2312.12222), AAAI'24) | 1,809 imgs / 63k QA; **test labels hidden (Codabench)** | LoveDA 0.3 m, 1024² | 166 answers | Easy (per-type answer sets) | SOBA 78.1 | `Kingdrone-Junjue/EarthVLSet` · NC-SA |
| HRVQA ([2301.09460](https://arxiv.org/abs/2301.09460)) | ~19.7k imgs | 8 cm aerial, 1024² | 10 types | Easy | 81.7 OA | hrvqa.nl · NC |
| FloodNet VQA ([2012.02951](https://arxiv.org/abs/2012.02951), 498) | community split 891 Qs | UAV **12 MP → downscale** | Y/N, condition, counting | Easy | counting 0.2–0.35 for all methods; in GeoChat-Instruct | GH BinaLab |
| **DisasterM3** ([2505.21089](https://arxiv.org/abs/2505.21089), NeurIPS'25 D&B) | 6k imgs (5,024 opt + 976 SAR) / 30,042 instr.; 6 MC tasks | WorldView 0.8 m pre/post; Capella / Umbra SAR post | MC (some multi-select; counts ±20/40%) + RES + reports | **Native** (multi-select → per-option `noul`) | GPT-4.1 42.3, Qwen2.5-VL-72B 40.5, GPT-4o 39.3, **Qwen2.5-VL-7B 31.2**, InternVL3-8B 31.3, TEOChat 23.0, EarthDial 22.9, GeoChat 10.7; fine-tuned Qwen2.5-VL-7B 40.4 | `Kingdrone-Junjue/DisasterM3` (HF live) / GH form · CC-BY-NC-SA |
| **DVL-Bench** ([2505.21076](https://arxiv.org/abs/2505.21076), NeurIPS'25 D&B) | 5,854 QA | NAIP 1 m, 1024², 5–10 frames | single / multi-choice | **Easy** (multi-image) | o4-mini 34.1, GPT-4o 29.7, Qwen2.5-VL-7B 23.3, TEOChat 17.2 | `weihao1115/dvl_suite` · Apache |
| **xBD / xView2** ([1911.09296](https://arxiv.org/abs/1911.09296), 423) | 933 test + 933 holdout pairs; 109,724 test polygons | Maxar RGB 1024², ≤0.8 m | 4-level damage | **Easy**: building crop (≈2× context) pre + post → `score` 0–3; tile → disaster type `choice`, "any destroyed?" `noul` | Damage F1: TEOChat 50.0, GPT-4o (3-shot) 38.3, Qwen2-VL 17.3, GeoChat 11.8. EarthDial disaster-type 96.4 vs GPT-4o 68.0 | xview2.org (registration) · CC-BY-NC-SA |
| BRIGHT ([2501.06019](https://arxiv.org/abs/2501.06019), ESSD'25, 120) | 4,246 pairs, 7:1:2 split | optical pre + **SAR post**, 0.3–1 m, 1024² | masks: intact / damaged / destroyed | **Easy**: building crops → `score` 0–2 (cross-modal) | no VLM numbers | `Kullervo/BRIGHT`, Zenodo · mixed BY / BY-NC |
| ABCD (MVA'17) | 8,506 pairs 160² | tsunami aerial 0.4 m | washed-away Y/N | **Native** `noul` | TEOChat zero-shot 85.6 vs specialist 95.2 | GH gistairc |
| S2Looking ([2107.09244](https://arxiv.org/abs/2107.09244)) | 1,000 test pairs 1024² | off-nadir sat 0.5–0.8 m | new / demolished building masks | Easy: `noul` "new building?", "demolished?" | TEOChat det F1 34.5, GPT-4o 21.5 | GH |
| LEVIR-CD / CD+ / CC / MCI | 128 / 348 / 1,929 test | Google Earth 0.5 m | change masks / captions | Easy `noul`; captions S2 | ChangeChat binary change 93.2 vs GPT-4 84.8 | academic |
| QFabric | 504 locations × 5 dates | WV-2 8192² (tile) | 6 change types, 9 status classes | Easy / Hard (`choice`, ordinal status → `score`) | TEOChat RQA F1 66.7–74.3 | via TEOChatlas |
| fMoW temporal ([1711.07846](https://arxiv.org/abs/1711.07846)) | val / test seqs | 0.3–1.6 m, variable px | 62 classes | Easy (N-way, multi-date) | TEOChat 75.1; EarthDial 70.0 vs GPT-4o 21.4 | custom NC |
| SpaceNet 7, DynamicEarthNet | 41 / 10 test AOIs | Planet 4 m / 3 m, 1024² | monthly change | Easy (`noul`, "which class expanded?") | — | CC-BY-SA / [M] |
| RescueNet (+VQA) | 450 test | UAV 12 MP | damage seg; VQA 103k | Easy after downscale | VQA damage-level ~0.64 | **CC-BY-NC-ND** |
| Sen1Floods11, OSCD, QuakeSet, CrisisLandMark | small | S1 / S2 10 m | masks / labels | Hard (render) → `noul` | QuakeSet: EarthDial 57.5 vs GPT-4o 55.9 | various NC |
| Geo3DVQA ([2512.07276](https://arxiv.org/abs/2512.07276)) | 842 (HF test) | GeoNRW + LiDAR GT | 3–4-way MC, numeric height | Easy (tiers 1–2) | Gemini-2.5-Flash 33.8, GPT-4o 31.0 | `mt129/Geo3DVQA` |
| RSCC ([2509.01907](https://arxiv.org/abs/2509.01907)) | 988 test pairs | xBD / EBD | change captions | S2 / caption-choice | — | `BiliSakura/RSCC` |

## 3. SAR, multispectral, hyperspectral, UAV and video

| Benchmark | Test set | Format | d3 fit | Reference scores | Access |
|---|---|---|---|---|---|
| **SARLANG-1M** ([2504.03254](https://arxiv.org/abs/2504.03254)) | 14,497 imgs / 125k QA | Y/N, class, count, 5-way position, open | **Easy** (subsample closed types) | zero-shot: InternVL2.5-4B 44.0, Qwen2.5-VL-7B 38.6; fine-tuned Qwen2.5-VL-7B 73.3 | `YiminJimmy/SARLANG-1M` |
| SARChat-Bench-2M ([2502.08168](https://arxiv.org/abs/2502.08168)) | 226k (10k classification, 11.7k counting) | 6-class, counts, boxes | classification Native `choice`(6); counts → bins | fine-tuned only (~91–93) | GH JimmyMa99/SARChat · NC |
| **BigEarthNet.txt** ([2603.29630](https://arxiv.org/abs/2603.29630)) | 1,082 S1+S2 pairs; 6,927 binary + 5,550 four-way MCQ | Y/N + MCQ | **Native** (rendered S2 + S1) | Qwen3-VL: binary 62.0, MCQ 37.6; EarthDial (RGB) binary 58.4 | `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt` |
| HM-Bench (HSI) ([2604.08884](https://arxiv.org/abs/2604.08884)) | 19,337 QA / 2,178 HSI | judgment + MC | Easy (pseudo-RGB + PCA renders + spectral text) | Qwen2.5-VL-7B 38.2 | no data link yet |
| UrbanVideo-Bench ([2503.06157](https://arxiv.org/abs/2503.06157)) | 5,355 MCQ / 1,547 drone videos | MC A–E | Native video (subsample >16 s) | GPT-4o 43.6, Qwen2-VL-7B 36.2 | `EmbodiedCity/UrbanVideo-Bench` |
| MM-UAVBench ([2512.23219](https://arxiv.org/abs/2512.23219)) | 5,702 MCQ | MC | Native (avg 1.68 MP, so downscale slightly) | **Qwen3-VL-8B 51.0**, GPT-4o 44.9; human 80.4 | `daisq/MM-UAVBench` |
| AirCopBench ([2511.11025](https://arxiv.org/abs/2511.11025)) | 14,610 multi-drone MCQ | MC | Native (multi-image) | GPT-4o 51.8 | — |
| RSVideo-Bench ([2608.02039](https://arxiv.org/abs/2608.02039)) | 2,731 five-choice | MC | Native video (code pending) | Qwen3-VL-8B 29.7 | — |

## 4. Scene classification (zero-shot as `choice`; multi-label as per-label `noul`)

| Dataset | Classes | Eval set used by VLM papers | Imagery | Reference zero-shot scores | Leakage risk | HF |
|---|---|---|---|---|---|---|
| **UC Merced** | 21 | all 2,100 (GeoChat protocol) | USGS 0.3 m, 256² | EarthDial 92.4, GPT-4o 88.8, TEOChat 86.3, GeoChat 84.4, Qwen3-VL-8B 75.7; **RSJEV (fine-tuned) 97.7** | **High** | `torchgeo/ucmerced`, `blanchon/UC_Merced`, GeoChat-Bench |
| **AID** | 30 | 3,000 (GeoChat list; the paper says 20%, but the file has 3,000) / 2,000 (SkyScript) / 10k | GE 600², 0.5–8 m | EarthDial 88.8, GPT-4o 74.7, InternVL3.5 72.5, GeoChat 72.0, Qwen3-VL 69.5; GeoZero 92.6; RSJEV 96.6 | Medium | `isaaccorley/aid`, GeoChat-Bench `aid.jsonl` |
| **NWPU-RESISC45** | 45 | test 6,300 | GE 256² | GPT-5 82.2, Qwen2.5-VL 68.9; RSCoVLM 98.3 (fine-tuned); RSJEV 94.6 | **Severe** (all images are in GeoChat / LHRS / VHM / Falcon training) | `timm/resisc45` |
| **WHU-RS19** | 19 | all 1,005 | GE 600–958 px | GPT-4o ~90–91, Qwen3-VL 85.7, EarthDial 96.2 | low–medium | `jonathan-roberts1/WHU-RS19` |
| **SIRI-WHU** | 12 | 2,400 | 200², 2 m | GPT-5 75.0, Qwen2.5-VL 67.3 | low | `jonathan-roberts1/SIRI-WHU` |
| RSSCN7 | 7 | 2,800 | 400² | — | low | `jonathan-roberts1/RSSCN7` |
| **EuroSAT-RGB** | 10 | test 5,400 | S2 64² (**upsampled to ≥256²**) | LHRS-Bot-Nova 63.5, GPT-4o 56.8 (72.4 with 1-shot), Qwen3-VL 50.9, GeoChat 35.4 | medium | `timm/eurosat-rgb` |
| EuroSAT-MS / -SAR | 10 | — | 13 bands / S1 | Gemini multi-render 66.3 → 69.1 | — | `blanchon/EuroSAT_MSI`, `wangyi111/EuroSAT-SAR` |
| PatternNet | 38 | 1k (VLEO) | 256² | GPT-4V .73 | medium | `jonathan-roberts1/PatternNet` |
| MLRSNet | 46 / 60 labels | subsample | 256² | RemoteCLIP 66.3 | — | `jonathan-roberts1/MLRSNet` |
| Million-AID | 8 / 28 / 51 (hierarchical) | 10k labelled (test hidden) | variable | SkyCLIP 67.5 | in RS5M / VHM | `jonathan-roberts1/Million-AID` |
| **BigEarthNet v2 (reBEN)** ([2407.03653](https://arxiv.org/abs/2407.03653)) | 19 multi-label | test 119,825 (subsample; geographically held-out) | S2 120² | v1-RGB: EarthDial 68.8, GPT-4o 49, GeoChat 20.4; Gemini F1 0.414 → 0.453 with multi-render | v1 in EarthDial train+test | `timm/bigearthnet-v2-rgb` |
| fMoW-WILDS | 62 | OOD test 22,108 | 224² crops | GPT-4V .19, GPT-4o 21.4 | train split widely used | `danielz01/fMoW` |
| So2Sat LCZ42 | 17 | test 24,188 | 32², S1 / S2 | EarthDial 60.7 vs GPT-4o 15.5 | EarthDial | `zhu-xlab/So2Sat-LCZ42` |
| SAT-4 / SAT-6 | 4 / 6 | 100k / 81k | NAIP 28² | OpenCLIP-G .54 / .45 | — | `jonathan-roberts1/SAT-4` |
| CLRS / NaSC-TG2 / RS_C11 / RSI-CB256 | 25 / 10 / 11 / 35 | full | — | EarthGPT CLRS 77.4, NaSC-TG2 74.7 | **lowest** | `jonathan-roberts1/*` |
| ~~Optimal-31~~ | 31 | — | — | — | **Skip: all 1,860 images are md5-identical to NWPU-RESISC45 images** (verified by download) | — |
| **SkyScript fine-grained** ([2312.12856](https://arxiv.org/abs/2312.12856)) | roof shape 6 / road smoothness 5 / surface 5 | 60–100 per class | NAIP / GEE | SkyCLIP-L 46.8 / 35.8 / 67.5 | low | S3 zips |

One-stop loader for 27 classification sets: **SATIN** ([2304.11619](https://arxiv.org/abs/2304.11619), `jonathan-roberts1/SATIN`).

## 5. Detection-derived presence / counting / fine-grained (converted)

| Dataset | Key facts | Recipe | Published VLM numbers |
|---|---|---|---|
| **GEOBench-VLM counting** | 1,054 five-way MCQ (vehicles, aircraft, buildings, trees, water, marine debris) | **Native** | LLaVA-OV .438, Qwen2-VL .402, GPT-4o .397, EarthDial .363, GeoChat .292 |
| **DIOR** ([1909.00133](https://arxiv.org/abs/1909.00133), 2,359) | 23,463 imgs 800²; 20 classes; **test 11,738, labels public**; CC-BY-NC | per-class presence `noul`; count bins `score`; A–D box-choice | Falcon (own split) counting: Falcon .770, LHRS-Bot .455, GeoChat .453 |
| **NWPU VHR-10** (976) | 800 imgs incl. **150 negatives**; 10 classes; CC-BY-NC | clean "no" presence cases; count bins; 10-way crop classification | EarthDial ROD 25.4 vs GeoChat 13.2 |
| HRSC2016 (727) | 1,061 imgs ≤1.35 MP; ship type hierarchy | ship-type crop `choice`; count bins | Falcon counting .838 |
| FAIR1M ([2103.05569](https://arxiv.org/abs/2103.05569), 614) | 15k imgs 1k–10k px; 37 fine-grained types | **crop → fine-grained type `choice`** (e.g. 11 aircraft models) | Falcon counting .816, GeoChat .377 |
| DOTA v1 / v2 (3,147 / 719) | 800–20,000 px; test labels withheld → use val | tile 1024; presence / count bins | Falcon DOTA2 counting .619 |
| xView (457) | WV-3 0.3 m, 60 classes; only train labels public; CC-BY-NC-SA | tile; vehicle-subtype crop `choice` | Falcon counting .278 |
| iSAID / RSOD / CARPK / NWPU-MOC / RSOC | various | count bins; NWPU-MOC gives RGB + NIR as 2 images | — |
| VLEO counting (COWC / NEON / animals / xBD) | regression | bins or ±20/40% MCQ | GPT-4V COWC MAE 2.85 (specialist 0.25) |
| **RSVQA-LR count** | 2,947 test Qs, 5 bins | **Native** | baseline 67.0 (no RS-VLM reports count) |

## 6. Grounding, retrieval and cross-view (converted)

| Dataset | Test | Recipe | Caveat |
|---|---|---|---|
| **DIOR-RSVG** ([2210.12634](https://arxiv.org/abs/2210.12634), 271) | 7,500, 800² | **box-choice** K=4–8 (other DIOR boxes in the same image, same class first, + proposals IoU<0.3) | Published Acc@0.5 is very protocol-dependent (LHRS-Bot 88.1 self-reported vs 17.6 re-run by GeoGround). Don't mix with box-choice |
| VRSBench-VG | 16,159 (unique / non-unique flag) | box-choice; report unique vs non-unique | best-structured: all objects annotated |
| GeoChat referring / grounding | 7,653 / 758 | rotated box-choice; region caption → reverse `choice` | OBB |
| OPT-RSVG | 24,477 (subsample) | box-choice | — |
| RRSIS-D / RefSegRS / EarthReason / GeoPixelD | 3,481 / 1,817 / 1,928 / 5,427 | mask-candidate overlay `choice` | Hard |
| **RSICD / RSITMD / UCM-Captions / NWPU-Captions** | ~1,093 / 452 / 210 / ~3,150 imgs | **caption-choice** (I→T) and multi-image `choice` (T→I). For R@K comparability: re-rank GeoRSCLIP top-20 by `noul` P(match) | captions are templated; use cross-class negatives |
| SkyScript retrieval | 30k pairs | caption-choice | — |
| University-1652 / CVUSA / VIGOR | see §6b | drone or ground image + K satellite tiles → `choice` | cross-view geo-localisation |

## 6b. Geolocalization and geo-reasoning on overhead imagery

GEOBench-VLM has **no** geolocation task (it uses fMoW only for land use), and no VLM paper reports fMoW country / continent or cross-view benchmarks in MC form, so the converted sets below would be new results.

| Benchmark | Test set | Imagery | Format | d3 fit + recipe | Reference scores | Access · license · cites |
|---|---|---|---|---|---|---|
| **UrBench Geo-Localization** ([2408.17267](https://arxiv.org/abs/2408.17267), AAAI'25) | city retrieval 2.06k, image retrieval 800, camera localization 1.07k, orientation 628 | Google Earth satellite + Street View; ≤6 images; panoramas ~2048×1024 (downscale) | 4-option MCQ | **Native** (multi-image `choice`) | CR / IR / CL / OR: GPT-4o 79.2 / 85.9 / 35.3 / 30.7; InternVL2-26B 61.3 / 23.0 / 32.3 / 24.7; human 30.0 / 92.6 / 82.9 / 85.7; random ≈24 | `bczhou/UrBench` · Google ToS · 54 |
| **VLEO-Bench landmarks** ([2401.17600](https://arxiv.org/abs/2401.17600)) | 602 US landmarks | NAIP ≤1 m, 421–2.05k px | 5-way name MC (Wikidata distractors) | **Native**; harder variant: US state among 8–16 | GPT-4V .671, LLaVA-1.5 .296, Qwen-VL-Chat .292 | `danielz01/landmarks` · CC-BY-SA · 61 |
| CHOICE GD + LR ([2411.18145](https://arxiv.org/abs/2411.18145)) | 200 + 100 | GEE 0.3 m, 512² | 4-option MC | Native (**saturated**: 95–99% for Qwen2-VL-7B / GPT-4o), so a sanity check only | Qwen2-VL-7B .970 / .980; GeoChat .810 / .790; LHRS-Bot .435 / .500 | in `An-Xiao/CHOICE` |
| **fMoW-WILDS region / country** ([2012.07421](https://arxiv.org/abs/2012.07421)) | OOD test 22,108 (subsample) | 224² WILDS crops; region and year labels; country in metadata (recompute from lat/lon) | — | **Easy**: continent 5-way `choice`; country 8–16-way (easy: random distractors; hard: same-region) | none for geolocation (VLM papers report land use only) | `danielz01/fMoW`, `wilds` · fMoW NC-SA-like · 544 |
| GeoVista satellite subset ([2511.15705](https://arxiv.org/abs/2511.15705)) | 108 | S2 L2A city mosaics 2000² (downscale) | city accuracy | Easy: city among 8–16 (same-continent distractors) | Gemini-2.5-pro 88.1, GPT-5 53.4, GeoVista-7B 44.9, Qwen2.5-VL-7B 16.1 | `LibraTree/GeoVistaBench` · research only · 9 |
| S2-100K ([SatCLIP](https://huggingface.co/datasets/torchgeo/s2-100k)) | 100k (subsample) | S2 L2A 256², 12 bands, global, with coords | — | Easy: continent / country `choice` (hard: 2.56 km footprint) | — | `torchgeo/s2-100k` · MIT |
| MMLandmarks ([2512.17492](https://arxiv.org/abs/2512.17492), CVPR'26) | 1,000 aerial queries | NAIP 800² | GPS-distance | Easy: NAIP → US state (8–16) or landmark among K | MMCLIP 36.9% within 1 km | link [M] · 0 |
| University-1652 ([2002.12186](https://arxiv.org/abs/2002.12186)) / SUES-200 ([2204.10704](https://arxiv.org/abs/2204.10704)) | 701 sat / 37,855 drone queries; 200 scenes × 4 altitudes | Google sat + rendered drone | R@K | Easy: drone + K=4–8 sat tiles (same-university hard negatives) → `choice`; or "same place?" `noul` | Sample4Geo D→S R@1 92.7 (specialist) | **by email request**, research only · 512 / 195 |
| CVUSA / CVACT / VIGOR | 8,884 / 92,802 / ~53k | panorama ↔ aerial tile (VIGOR panos 2.1 MP → downscale) | R@K | Easy: panorama + K aerial tiles → `choice` (exclude VIGOR semi-positives) | Sample4Geo R@1 98.7 / 71.5 / 61.7 (cross-area) | request forms, no redistribution · — / 409 / 346 |
| EarthLoc / AstroLoc ([2403.06758](https://arxiv.org/abs/2403.06758)) | 6 test regions, 682–6,142 queries | astronaut photos vs S2 tiles 1024² | R@N | Easy–Hard: photo + K tiles; or region `choice` | EarthLoc R@1 57.1 | vandaldata.polito.it · MIT code |
| OmniEarth-Bench geolocation tasks | 502 + 500 | [M] | MC / open | check format first | erratic (GPT-4o 2.6) | NC-SA |

Ground-level geolocation sets (OSV-5M, GAEA-Bench, EarthWhere, GeoReasoner, GeoChain) are out of scope for EO, but GAEA-Bench / EarthWhere MCQs are native if a ground-view control is ever wanted.

## 7. Geospatial foundation-model benchmarks (adaptable)

| Benchmark | Usable tasks | Fit |
|---|---|---|
| **GEO-Bench** ([2306.03831](https://arxiv.org/abs/2306.03831), `recursix/geo-bench-1.0`) | m-eurosat (10), m-brick-kiln (2), m-pv4ger (2), m-forestnet (12), m-so2sat (17), m-bigearthnet (43 multi-label) | **Easy** (render + upsample); segmentation → dominant-class / coverage questions |
| GEO-Bench-2 ([2511.15658](https://arxiv.org/abs/2511.15658)) | classification: BEN v2, TreeSatAI, So2Sat, ForestNet | Easy |
| **Copernicus-Bench** ([2503.11849](https://arxiv.org/abs/2503.11849), `wangyi111/Copernicus-Bench`) | EuroSAT-S1/S2, BigEarthNet-S1/S2, LC100Cls-S3, LCZ-S2 | Easy (SAR / S3 rendering); regression Hard |
| PANGAEA ([2412.04204](https://arxiv.org/abs/2412.04204)) | burn / flood / xView2 (segmentation) | Hard → `noul` / `choice` |
| **REOBench** ([2505.16793](https://arxiv.org/abs/2505.16793), `xiang709/REOBench`) | AID classification + VRSBench VQA under 12 corruptions × 5 severities | **Easy; the ready-made robustness add-on** |
| Agentic (ThinkGeo, Earth-Bench, GeoLLM-Engine, GeoAnalystBench, CityBench) | tool use | S2 only |

---

## 8. Recommended suites

### 8.1 Core suite (Tier 1): run everything here first

Chosen to cover every capability axis while staying native/easy for d3-flash, and to mix "everyone reports it" sets with low-leakage 2025–26 sets.

| # | Benchmark (variant) | Capability axis | d3 types | Eval set we use | Approx. questions |
|---|---|---|---|---|---|
| C1 | **GEOBench-VLM** (MCQ "single") | broad: scene, counting, disaster, fine-grained, temporal, SAR / MS | `choice`(5) | 3,211 | 3.2k |
| C2 | **CHOICE** | hierarchical perception + reasoning, **new imagery (low leakage)**, multi-sensor | `choice`(2–4) | 9,407 MCQ | 9.4k |
| C3 | **VRSBench** VQA (closed-vocabulary conversion) + VG (box-choice) | the most-cited 2024 RS VLM benchmark; grounding | `noul` / `choice` / bins | stratified 5k VQA + 2k VG | 7k |
| C4 | **RSVQA-HR test-2** + **RSVQA-LR** (GeoChat protocol + count bins) | canonical RS VQA; most baselines | `noul` / `choice` | 5k HR + full LR | ~15k |
| C5 | **LHRS-Bench** (strict, 4 circular orders) | sanity + order robustness | `choice` | 690 × 4 | 2.8k |
| C6 | **Scene pack:** UCM, AID (GeoChat 3k), WHU-RS19, SIRI-WHU, EuroSAT-RGB, + NWPU-RESISC45 (flagged) | zero-shot classification; RSJEV comparability | `choice` (full N-way; 16-way secondary) | standard sets | ~20k |
| C7 | **BigEarthNet v2 RGB** | multi-label LULC, Sentinel-2 10 m | 19 × `noul` | 1k images | 19k |
| C8 | **CDVQA** test2 | bi-temporal change | `noul` / `choice` | 5k subsample | 5k |
| C9 | **xBD** building damage + disaster type | **ordinal `score`**; pre / post pair | `score`(0–3) / `choice` | 3k buildings (class-stratified) + 933 tiles | ~4k |
| C10 | **DisasterM3** (MC tasks, optical + SAR) | disaster response; optical → SAR | `choice` / per-option `noul` | ~5k stratified | 5k |
| C11 | **EarthMind-Bench** (MC) | SAR and optical–SAR fusion | `choice` | 2,841 | 2.8k |
| C12 | **DVL-Bench** (or VLRS-Bench) | long-horizon multi-temporal reasoning (5–10 images) | `choice` / `noul` | 5,854 | 5.9k |
| C13 | **XLRS-Bench-lite** (two variants: full-image downscale vs. tiles as multi-image) | ultra-high-resolution stress test | `choice`(4) | lite set | ~4k × 2 |
| C14 | **VHM HnstD** | hallucination / deceptive questions (open-set behaviour) | `noul` / `choice` | 1,642 | 1.6k |
| C15 | **UrBench Geo-Localization** + **fMoW-WILDS continent / country** | geo-localisation and cross-view (satellite ↔ street) | multi-image `choice` | 4.5k UrBench + 2k fMoW-OOD | 6.5k |

**Total ≈ 115k question-forwards** for d3-flash. Capability coverage:

| Axis | Covered by |
|---|---|
| Scene / LULC classification | C1, C6, C7 |
| Object presence / counting | C1 (1,054 counting), C3, C4 (LR count), C10 |
| Fine-grained objects | C1 (FAIR1M / RarePlanes tasks), C2 |
| Spatial relations / grounding | C2, C3-VG, C13 |
| Change / temporal | C1, C8, C9, C12 |
| Disaster | C9, C10 |
| SAR / multi-sensor | C2, C10, C11 |
| Ultra-high resolution | C13 |
| Hallucination / honesty | C14 |
| Geo-localisation / cross-view | C15 (+ CHOICE GD / LR as a saturated sanity check) |
| Ordinal decisions (`score`) | C9 |
| Low-leakage signal | C2, C11, C12, C13 (newly collected 2025–26 imagery) |

### 8.2 Extended suite (Tier 2): once the core runs are stable

- **Resolution:** MME-RealWorld-RS, RSHR-Bench, LRS-VQA (full downscale vs. crops).
- **Broad MCQ:** OmniEarth (2026), VLRS-Bench, UrBench, OmniEarth-Bench (report "reject" handling separately).
- **Relations:** FIT-RSRC.
- **SAR / MS / HSI:** SARLANG-1M (closed types), SARChat classification, BigEarthNet.txt, EuroSAT-MS / -SAR, So2Sat, the GEO-Bench and Copernicus-Bench classification tasks, HM-Bench (once released).
- **Change / disaster:** TEOChatlas change QA (S2Looking, QFabric, ABCD), BRIGHT (optical → SAR damage), FloodNet VQA, Geo3DVQA tiers 1–2, EarthVQA (Codabench), HRVQA.
- **Fine-grained / harder classification:** FAIR1M aircraft-type crops, SkyScript fine-grained, PatternNet, fMoW-WILDS OOD (N=62), Million-AID hierarchical.
- **Converted spatial:** DIOR-RSVG box-choice, DIOR / NWPU-VHR-10 presence + count bins, RSICD / RSITMD caption-choice.
- **Geolocation:** VLEO landmarks, GeoVista satellite, S2-100K continent, University-1652 / SUES-200 / CVUSA / VIGOR cross-view `choice` (request-only data), EarthLoc.
- **Robustness:** REOBench.
- **UAV / video:** MM-UAVBench, UrbanVideo-Bench, AirCopBench, AVI-Math (MC / TF).
- **Not EO imagery (report separately):** MSEarth.

### 8.3 Pitfalls (summary)

1. **Leakage.**
   - The worst sets are NWPU-RESISC45 (all images in many RS instruction sets; **Optimal-31 is a byte-identical subset of it**), RSVQA-LR (a question-only model trained on it reaches 73.8%, Chappuis et al.), UCM (UCM-Captions), fMoW and BigEarthNet v1 (EarthDial even used BEN test), and xBD imagery (under DisasterM3, RSCC and GEOBench-VLM).
   - d3's own training data is undisclosed.
   - Mitigation: always run **blind** (no image) and **shuffled-image** controls; weight conclusions toward CHOICE, EarthMind, DVL / VLRS, XLRS, the BEN v2 held-out test and fMoW-WILDS OOD.
2. **Resolution.**
   - Over the 1.6 MP cap: XLRS (~72 MP), RSHR, LRS-VQA, MME-RW-RS, FloodNet / RescueNet (12 MP), DOTA / xView / FAIR1M, AVI-Math, MM-UAVBench (~1.68 MP).
   - Tiny chips are upscaled to ≥256²: SAT (28²), So2Sat (32²), EuroSAT (64²), BEN (120²). Log the resize factor.
3. **More than 16 options:** AID 30, NWPU 45, PatternNet 38, fMoW 62, MLRSNet 46. Report full N-way and 16-way separately.
4. **Reject / "unable" options and multi-select:** decide per benchmark and document it. Use per-option `noul` with a threshold fixed on a dev split.
5. **Protocol drift:**
   - AID: 3k / 2k / 10k.
   - LHRS-Bench: strict vs. lenient (≈45 vs ≈68 for GPT-4-class).
   - RSVQA-LR: count / area dropped.
   - VRSBench / RSVLM-QA: GPT-judged.
   - Counting depends on binning; grounding Acc@0.5 ≠ box-choice accuracy.
   - When citing published numbers, match the protocol or say it differs.
6. **Hidden test labels:** EarthVQA, Million-AID, the FloodNet challenge, DOTA / FAIR1M / xView test.
7. **Licenses:**
   - Most sets are NC or academic-only (DOTA, Google Earth imagery, xBD, fMoW, DisasterM3, XLRS, DIOR-RSVG); RescueNet is **ND**.
   - Research use is fine; publish converters, not converted images.
   - Relatively permissive: GEOBench-VLM (Apache code), OmniEarth (Apache), EuroSAT (MIT), BigEarthNet v2 (CDLA-P), So2Sat (CC-BY), REOBench (CC-BY), RSVQA (CC-BY).

### 8.4 Searched for but not found

- **No such EO benchmark:** SpatialEarth, GeoReasoning, SAR-Bench / SARBench, MS-VQA, SITS-VQA, GeoBench-Video, RS-MMBench, GeoCrossBench, MMEarth-Bench, xBD-QA, MDCD.
- **Exist but not usable yet:** GeoMMBench (CVPR'26, 1,053 MCQ) exists but its link could not be verified; EarthSpatialBench, RSVideo-Bench and LongEarth-Bench are announced but have no data yet.
