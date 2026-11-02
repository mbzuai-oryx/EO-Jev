# Re-checked SOTA specialist models (answer, 2026-10-17)

You were right: my earlier list was outdated. Two research agents rechecked the field, and newer, stronger specialists now lead most benchmarks; TEOChat, VHM and EarthDial lead almost nowhere anymore. The full findings are in [05_specialist_models.md](05_specialist_models.md). Only models with downloadable weights are listed:

| Our benchmark | Best open specialists (best first) |
|---|---|
| Scene classification (6 sets) | **RSThinker** (2025), **GeoZero** (2025), **RSCoVLM** (2025) |
| BigEarthNet | EarthMind, EarthDial, TerraScope |
| GEOBench-VLM | **MLRS** (2026; 50.3), EarthDial, GeoZero |
| CHOICE | **GeoZero** (no published CHOICE score; the 71.7 cited earlier was unsourced), VHM |
| LHRS-Bench | **ScoreRS** (69.2), ZoomEarth |
| VRSBench | GeoSolver / RSThinker, EarthMind |
| RSVQA-LR / HR | EarthDial, RSCoVLM, TEOChat / LHRS-Bot-Nova, GeoZero |
| CDVQA, DVL-Bench | **GeoChrono** (2026; best on both), EarthDial, TEOChat |
| xBD, DisasterM3 | **TerraScope** (beats TEOChat on DisasterM3 damage counting, 54.1 vs 22.5), TEOChat, EarthDial |
| EarthMind-Bench | EarthMind, TerraScope, EarthDial |
| XLRS-Bench | **Text-Before-Vision** (60.4), GeoEyes, GeoLens: all zoom-tool agents |
| fMoW | Geo-R1 (geolocation), TEOChat / GeoZero (facility type) |
| UrBench | MLRS, Geo-R1 |
| VHM honesty | VHM, GeoZero |

## What to keep in mind when reading these numbers

- **Some top published scores can't be reproduced.** Earth-OneVision, ScaleEarth (XLRS 59.4), DVLChat and the DisasterM3 fine-tunes never released weights. Their numbers can only be quoted.
- **Many specialists trained on the benchmark's own training split**, including scene classification, RSVQA, VRSBench, BigEarthNet, xBD and fMoW. Their scores will be marked "in-domain" in the table. For xBD, our questions come from the **test** split (checked), so it's the same events but different images.
- **On some benchmarks general models beat every specialist**: DisasterM3, UrBench and LHRS. There, the specialist column is not the true ceiling.

## Proposed execution

- **Phase 1:** 10 models in 6 software environments, covering every benchmark: GeoZero, RSThinker, RSCoVLM, ScoreRS, Geo-R1, TerraScope, EarthMind (×2), EarthDial, TEOChat, GeoChrono, VHM.
  - Setup: about 2–3 days. Runs: about 2–3 days on our 4 GPUs.
- **Phase 2:** the harder ones.
  - XLRS zoom agents: they need a tool loop at full resolution, generated answers only.
  - MLRS: needs you to accept Meta's gated SAM 3 license on Hugging Face.
  - LHRS-Bot-Nova and GeoLLaVA-8K.
  - Adds about 2 days.
- **Validation:** before trusting any model, reproduce one of its published numbers on a benchmark where our setup matches theirs. Examples: GeoZero on UCM (93.8), EarthMind on EarthMind-Bench, GeoChrono on CDVQA.

Storage is no issue: about 300 GB of weights, against 1.7 PB free on `/l`.

**Open question:** start Phase 1 now and Phase 2 after it, or do both?

---

# Which model runs on which benchmark (follow-up)

Not every model runs on every benchmark. Each model runs only on the benchmarks where it is among the best open specialists, plus the one benchmark used to validate it. Lean plan, Phase 1:

| Model | Runs on | Questions (approx.) |
|---|---|---|
| RSThinker | 6 scene sets, VRSBench | 25k |
| GeoZero | 6 scene sets (best on EuroSAT), CHOICE (validation), GEOBench-VLM, RSVQA-HR, VHM honesty, fMoW facility type | 42k |
| ScoreRS | LHRS-Bench | 2.4k |
| Geo-R1 | fMoW continent + country, UrBench | 8k |
| TerraScope | xBD damage, xBD disaster type, DisasterM3 | 9k |
| EarthMind | EarthMind-Bench (validation), BigEarthNet | 24k |
| EarthDial | RSVQA-LR | 10k |
| GeoChrono | CDVQA (validation), DVL-Bench | 9k |
| VHM-7B | VHM honesty | 2.3k |

- **Two models on the scene sets:** the leader differs by dataset (RSThinker on AID / WHU / SIRI, GeoZero on EuroSAT, and the others are within ~2 points), and the scene sets are cheap.
- **Dropped:**
  - TEOChat: GeoZero covers its only remaining slot, fMoW facility type.
  - RSCoVLM: it duplicates RSThinker and EarthDial.
  - Dropping both saves a software environment.
- **Total:** about 130k questions per answer mode (5 environments, 9 models), instead of more than a million if every model ran every benchmark.
- **Time:** runs take about 1 day on 4 GPUs; setup (5 environments plus validation) about 2 days.

Phase 2, unchanged:

| Model | Benchmark |
|---|---|
| Text-Before-Vision, GeoEyes | XLRS |
| MLRS | GEOBench-VLM, UrBench |
| LHRS-Bot-Nova | RSVQA-HR |
