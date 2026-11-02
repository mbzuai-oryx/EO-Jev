"""Weighted average of LoRA adapters trained from the same base, written as one adapter (a "model soup").

Adapter i adds w_i * s_i * B_i A_i to each layer (s_i = alpha_i / r_i). The sum is exactly one LoRA of rank sum(r_i):
the A matrices stacked row-wise, the B matrices side by side, block i scaled by w_i * s_i / s, where s is the first
adapter's scale. train.export merges the result like any adapter (and runs its export check on it).

    MODULE=train.merge_adapters ARGS="--out <merged.safetensors> <adapter A>:0.5 <adapter B>:0.5" \
        sbatch slurm/convert_cpu.sbatch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("adapters", nargs="+", help="<adapter.safetensors>:<weight>")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    parts = []
    for spec in args.adapters:
        path, _, w = spec.rpartition(":")
        with safe_open(path, framework="pt") as f:
            info = f.metadata()
        parts.append((path, float(w), int(info["r"]), float(info["alpha"]) / int(info["r"]), load_file(path)))
    scale = parts[0][3]
    names = sorted({k.rsplit(".", 1)[0] for k in parts[0][4]})
    for path, *_, state in parts[1:]:
        if sorted({k.rsplit(".", 1)[0] for k in state}) != names:
            raise ValueError(f"{path} covers other layers than {parts[0][0]}")
    merged = {}
    for name in names:
        merged[f"{name}.lora_a"] = torch.cat([st[f"{name}.lora_a"] for *_, st in parts], 0).contiguous()
        merged[f"{name}.lora_b"] = torch.cat([st[f"{name}.lora_b"] * (w * s / scale) for _, w, _, s, st in parts],
                                             1).contiguous()
    r = sum(r for _, _, r, _, _ in parts)
    meta = {"merged_from": [{"adapter": str(Path(path).resolve()), "weight": w} for path, w, *_ in parts]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(merged, args.out, metadata={"format": "eojev-d3-lora/1", "r": str(r), "alpha": str(scale * r),
                                          "meta": json.dumps(meta)})
    print(f"{len(names)} layers, rank {r}, scale {scale}: {json.dumps(meta)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
