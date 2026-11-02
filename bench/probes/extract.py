"""Extract probe features while answering rows. For each row (one question, any number of images) it saves:
- the decision state, i.e. the last-token hidden state that the readout scores;
- the vision-tower output, mean-pooled per image and concatenated in image order.

Chunks are claimed through lock files like bench/runners/d3.py, so a task on a dead GPU just exits. Image paths
are relative to $EOJEV_DATASETS/converted unless absolute. Choice, score and noul questions are all supported.
Labels and predictions are option keys: the level index for score questions, "true" / "false" for noul.

    python -m bench.probes.extract --model d3-flash --rows probe_AID.jsonl --out-dir <dir> [--chunks 32]

Writes <dir>/chunk<c>.npz with ids, labels, split, pred, p_gold, hidden [N, 4096] and vision [N, images x D], all
as float16.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bench.common import CONVERTED, read_jsonl
from bench.runners.d3 import MODELS, claim, load_runtime


def tensor_of(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return tensor_of(output[0])
    for name in ("pooler_output", "last_hidden_state"):
        value = getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError(f"cannot find a tensor in {type(output)}")


def per_image_mean(features: torch.Tensor, grid, merge: int) -> torch.Tensor:
    """Mean of each image's tokens, concatenated. The vision tower returns the tokens of all of a prompt's images in
    one tensor; grid_thw gives each image's patch grid (merged tokens = t*h*w / merge^2)."""
    if grid is None or len(grid) == 1:
        return features.mean(0)
    patches = grid.prod(-1).tolist()
    for sizes in ([n // merge**2 for n in patches], patches):
        if sum(sizes) == features.shape[0]:
            return torch.cat([part.mean(0) for part in torch.split(features, sizes)])
    raise ValueError(f"vision output has {features.shape[0]} tokens; grid_thw {grid.tolist()} does not split it")


def probabilities(question: dict, answer: dict) -> dict[str, float]:
    if question["type"] == "noul":
        return {"false": 1.0 - answer["noul"], "true": answer["noul"]}
    return answer["probabilities"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="d3-flash", help=f"{', '.join(sorted(MODELS))} or a package directory")
    p.add_argument("--rows", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--chunks", type=int, default=32)
    args = p.parse_args()
    rows = list(read_jsonl(args.rows))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    host = f"{socket.gethostname()} GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}"
    try:
        runtime = load_runtime(args.model, batch_size=1, permutation_average=False)
    except Exception as exc:  # noqa: BLE001
        print(f"{host}: model load failed ({type(exc).__name__}: {exc})", flush=True)
        sys.exit(3)

    visual = runtime.backbone.visual
    merge = getattr(visual, "spatial_merge_size", None) or runtime.backbone.config.vision_config.spatial_merge_size
    captured: dict[str, torch.Tensor] = {}

    def vision_hook(module, hook_args, hook_kwargs, output):
        grid = hook_kwargs.get("grid_thw", hook_args[1] if len(hook_args) > 1 else None)
        captured["vision"] = per_image_mean(tensor_of(output).float(), grid, merge).cpu()

    runtime.backbone.register_forward_hook(
        lambda m, i, o: captured.__setitem__("hidden", tensor_of(o)[:, -1].float().cpu()))
    visual.register_forward_hook(vision_hook, with_kwargs=True)

    for c in range(args.chunks):
        if (out_dir / f"chunk{c}.done").exists() or not claim(out_dir, c):
            continue
        rec = {k: [] for k in ("ids", "labels", "split", "pred", "p_gold", "hidden", "vision")}
        for i, row in enumerate(rows):
            if i % args.chunks != c:
                continue
            captured.clear()
            images = []
            for rel in row["images"]:
                with Image.open(CONVERTED / rel) as im:  # an absolute path stays absolute
                    images.append(im.convert("RGB"))
            result = runtime.system_one(state=row["state"], questions=row["questions"], images=images)
            qid = next(iter(row["questions"]))  # probe rows carry exactly one question
            q = row["questions"][qid]
            probs = probabilities(q, result["answers"][qid])
            gold = row["answer"][qid]
            gold = ("true" if gold else "false") if q["type"] == "noul" else str(gold)
            rec["ids"].append(row["id"])
            rec["labels"].append(gold)
            rec["split"].append(row["meta"].get("probe_split", ""))
            rec["pred"].append(max(probs, key=probs.get))
            rec["p_gold"].append(probs[gold])
            rec["hidden"].append(captured["hidden"][0].numpy().astype(np.float16))
            rec["vision"].append(captured["vision"].numpy().astype(np.float16))
        np.savez(out_dir / f"chunk{c}.npz", ids=np.array(rec["ids"]), labels=np.array(rec["labels"]),
                 split=np.array(rec["split"]), pred=np.array(rec["pred"]), p_gold=np.array(rec["p_gold"]),
                 hidden=np.stack(rec["hidden"]), vision=np.stack(rec["vision"]))
        (out_dir / f"chunk{c}.done").touch()
        print(f"{host}: chunk {c} done ({len(rec['ids'])} rows)", flush=True)


if __name__ == "__main__":
    main()
