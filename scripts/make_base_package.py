"""Build a d3-format package from plain Qwen/Qwen3.5-9B: the same-backbone control.

The package links the base checkpoint's files (weights, tokenizer, chat template and image processor, all
byte-identical to d3-flash's except the weights) and adds:
  - decision_config.json: d3-flash's, with attention_mode "causal" (the base model was trained causal);
  - readout.safetensors: the base lm_head rows of the 255 answer-code tokens, in FP32.
Run through d3's own runtime, this is the "read the letter logits of an instruction model" zero-shot baseline
(Releasing the Inner Jev, arXiv 2610.17205) under the exact d3 prompt, image placement and readout.

    python scripts/make_base_package.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HUB = Path(os.environ["HF_HUB_CACHE"])
D3 = HUB / "models--vllm-sr--d3-flash/snapshots/521525effa79469d899a570726547dd7f00da3af"
BASE = HUB / "models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
OUT = Path(os.environ["EOJEV_MODELS_ROOT"]) / "packages" / "qwen3.5-9b-d3readout"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in BASE.iterdir():
        link = OUT / f.name
        if not link.exists():
            link.symlink_to(f.resolve())

    config = json.loads((D3 / "decision_config.json").read_text())
    if config["base_model"] != "Qwen/Qwen3.5-9B" or config["revision"] != BASE.name:
        raise ValueError("d3-flash's base revision does not match the downloaded Qwen3.5-9B snapshot")
    config["attention_mode"] = "causal"
    (OUT / "decision_config.json").write_text(json.dumps(config, indent=2))

    index = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(BASE / index["lm_head.weight"]), framework="pt") as handle:
        lm_head = handle.get_tensor("lm_head.weight")
    ids = torch.tensor(config["token_ids"])
    readout = lm_head[ids].float().contiguous()
    save_file({"weight": readout}, str(OUT / "readout.safetensors"))

    # Sanity: d3-flash's readout rows A..Z equal the base lm_head rows (checked during the literature review).
    with safe_open(str(D3 / "readout.safetensors"), framework="pt") as handle:
        d3_readout = handle.get_tensor("weight")
    cos = torch.nn.functional.cosine_similarity(d3_readout[:26].float(), readout[:26], dim=1)
    print(f"package: {OUT}\nreadout {tuple(readout.shape)}; cosine to d3-flash readout rows A-Z: "
          f"min {cos.min():.4f}, mean {cos.mean():.4f}")


if __name__ == "__main__":
    main()
