"""Open-weight EO specialist models for the SOTA re-measurement.

Each entry names the adapter that loads it and the model's own prompt conventions, taken from the authors'
inference code (third_party/<repo>):
- GeoZero: optional "Type1" system prompt, reasoning in prose then <answer>...</answer>; greedy for classification
  (GeoZero/single_infer_eval_geozero_think.py).
- RSThinker: no system prompt; <think>...</think><answer>...</answer>; top_k=1, repetition_penalty 1.1
  (RSThinker/RSThinker_infer_dataset.py).
- ScoreRS (Qwen2VL-7B-RS-R1): the R1 system prompt with <think>/<answer> tags (ScoreRS/README.md).
- Geo-R1: R1-style <think>/<answer> tags.
- TerraScope / EarthMind: bench/specialists/sa2va.py; EarthDial: bench/specialists/earthdial.py;
  GeoChrono: bench/specialists/geochrono.py; VHM: bench/specialists/vhm.py.
Adapters are imported lazily because some models need their own Python environment.
"""

from __future__ import annotations

import os
from importlib import import_module

R1_SYSTEM = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The "
             "assistant first thinks about the reasoning process in the mind and then provides the user with the "
             "answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> "
             "tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>")
GEOZERO_SYSTEM = ("When answering, first describe your reasoning or visual observations in natural language, and then "
                  "provide the final answer enclosed in <answer></answer>.")

SPECIALISTS: dict[str, dict] = {
    "geozero": dict(adapter="bench.specialists.hf_vlm:HFVLM", repo="hjvsl/GeoZero", subfolder="GeoZero-8B-without-RFT",
                    kwargs=dict(system=GEOZERO_SYSTEM, answer_prefix="<answer>", answer_end="</answer>")),
    "rsthinker": dict(adapter="bench.specialists.hf_vlm:HFVLM", repo="minglanga/RSThinker",
                      kwargs=dict(answer_prefix="<think></think><answer>", answer_end="</answer>",
                                  gen_kwargs={"repetition_penalty": 1.1}, max_new_tokens=2048)),
    "scorers-r1": dict(adapter="bench.specialists.hf_vlm:HFVLM", repo="PumpkinCat/Qwen2VL-7B-RS-R1",
                       kwargs=dict(system=R1_SYSTEM, answer_prefix="<think> </think><answer>", answer_end="</answer>")),
    "geo-r1": dict(adapter="bench.specialists.hf_vlm:HFVLM", repo="miniHui/Geo-R1",
                   kwargs=dict(system=R1_SYSTEM, answer_prefix="<think> </think><answer>", answer_end="</answer>")),
    # Sa2VA family (eo-sa2va env)
    "terrascope": dict(adapter="bench.specialists.sa2va:Sa2VA", repo="sy1998/TerraScope",
                       # TerraScope writes "</think>\n<answer>\nD. flood\n</answer>": the newline after <answer> is
                       # part of the prefix, else the next token is "\n" and the letter probabilities are noise
                       kwargs=dict(kind="terrascope", answer_prefix="<think>\n\n</think>\n<answer>\n",
                                   answer_end="\n</answer>")),
    "earthmind": dict(adapter="bench.specialists.sa2va:Sa2VA", repo="sy1998/EarthMind-4B",
                      kwargs=dict(kind="earthmind")),
    "earthmind-multi": dict(adapter="bench.specialists.sa2va:Sa2VA", repo="sy1998/EarthMind4B_multi",
                            kwargs=dict(kind="earthmind_multi")),
    # InternVL2 code (eo-sa2va env)
    "earthdial": dict(adapter="bench.specialists.earthdial:EarthDial", repo="akshaydudhane/EarthDial_4B_RGB"),
    # GeoChrono (eo-geochrono env): LoRA + TempEnc on Qwen3-VL-4B-Instruct
    "geochrono": dict(adapter="bench.specialists.geochrono:GeoChrono", repo="Davidup1/GeoChrono",
                      kwargs=dict(base_repo="Qwen/Qwen3-VL-4B-Instruct")),
    # VHM (eo-vhm env)
    "vhm": dict(adapter="bench.specialists.vhm:VHM", repo="FitzPC/vhm_7B"),
    # TEOChat (eo-teochat env): Video-LLaVA-7B + LoRA on TEOChatlas (temporal pairs / sequences; trained on xBD train)
    "teochat": dict(adapter="bench.specialists.teochat:TEOChat", repo="jirvin16/TEOChat"),
    # eval-v1.0 reference models, stock Qwen-VL fine-tunes with their base chat templates:
    # UHR-BAT2 (Qwen3-VL-8B; its own eval scores option logits at <=1.31 MP) for RSHR-Bench (+ MME-RW-RS anchor),
    # RS-OPSD (Qwen3-VL-8B; weights in F32, loaded in bf16) for MME-RealWorld-RS,
    # Sky-VLM (Qwen2.5-VL-7B; official inference caps images at ~1.6 MP) for SpatialSky-Bench
    "uhr-bat2": dict(adapter="bench.specialists.hf_vlm:HFVLM", repo="RL-MIND/UHR-BAT2"),
    "rs-opsd": dict(adapter="bench.specialists.hf_vlm:HFVLM", repo="ronniejiangC/RS-OPSD"),
    "sky-vlm": dict(adapter="bench.specialists.hf_vlm:HFVLM", repo="llxs/Sky-VLM"),
    # SIS-Motion (eo-sismotion env): Qwen2.5-VL-7B + merged LoRA + optical-flow motion branch, for SIS-Bench
    "sis-motion": dict(adapter="bench.specialists.sis_motion:SISMotion", repo="choucsan/SIS-Motion",
                       kwargs=dict(base_repo="Qwen/Qwen2.5-VL-7B-Instruct")),
    # Qwen3-VL-8B-Instruct + a QLoRA adapter for xView2 disaster type (in-domain on xBD; trained 4-bit, run 4-bit)
    "waybob-xview2": dict(adapter="bench.specialists.hf_vlm:HFVLM", repo="Qwen/Qwen3-VL-8B-Instruct",
                          kwargs=dict(adapter_repo="WayBob/Qwen3VL-8B-QLora-4bit-xView2-Disaster-Recognition",
                                      load_in_4bit=True)),
}


def local_path(spec: dict) -> str:
    """The downloaded snapshot (scripts/download_specialists.sh). Read from the cache layout directly: a snapshot
    fetched with allow_patterns counts as incomplete for snapshot_download(local_files_only=True)."""
    cache = os.environ["HF_HUB_CACHE"]
    repo_dir = os.path.join(cache, "models--" + spec["repo"].replace("/", "--"))
    with open(os.path.join(repo_dir, "refs", "main")) as f:
        root = os.path.join(repo_dir, "snapshots", f.read().strip())
    return os.path.join(root, spec["subfolder"]) if spec.get("subfolder") else root


def load(name: str, **overrides):
    spec = SPECIALISTS[name]
    module, cls = spec["adapter"].split(":")
    kwargs = {**spec.get("kwargs", {}), **overrides}
    for key in [k for k in kwargs if k.endswith("_repo")]:  # e.g. base_repo -> base=<local snapshot of that repo>
        kwargs[key[: -len("_repo")]] = local_path({"repo": kwargs.pop(key)})
    return getattr(import_module(module), cls)(local_path(spec), **kwargs)
