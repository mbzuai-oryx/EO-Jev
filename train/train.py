"""LoRA fine-tuning of d3-flash on the EO training rows. The defaults are the settings of the EO-Jev model in the
paper (4 GPUs, 20,000 steps of 16 questions).

**Loss per question:** KL(target || p) over the question's own codes, plus rps_weight × RPS for ordinal questions
(score levels, and counting questions with increasing options). p is the readout softmax at the checkpoint
temperature. Targets are built in train/data.py.

**Optimizer:** AdamW on the LoRA weights only, linear warm-up then cosine decay to 10%, gradient clipping at 1.0.
A micro-batch that runs out of memory is retried one question at a time.

**Adapters:** rank 32 on the text tower; --merger-rank adds adapters on the vision merger (connector) at the text
learning rate; --vision-rank adds them on all vision blocks at --vision-lr-scale times it (0.2: LLaVA-OneVision tunes
its vision encoder at a fifth of the language model's rate; Prismatic VLMs found full fine-tuning of the encoder
hurts). Set either rank to 0 to keep that part frozen.

**Several GPUs:** started with torchrun, each GPU draws its own micro-batches (own sampler seed) and does
--accum / world-size of them, so the step's batch stays --micro x --accum questions. Gradients are averaged over
GPUs before clipping, so every GPU takes the same step from the same adapters (broadcast from GPU 0 at the start).
GPU 0 logs, evaluates and saves; every GPU keeps its own sampler state (state_rank<r>.pt) for resuming.

    torchrun --standalone --nproc_per_node=4 -m train.train --out-dir $EOJEV_RUNS/train/eo-jev

    STEPS="train export" TRAIN_GPUS=4 NAME=eo-jev sbatch -J eojev-train --ntasks=1 --nodes=1 slurm/train_eval.sbatch

**Outputs in --out-dir:**
- args.json;
- log.jsonl: training metrics every --log-every steps, dev metrics every --eval-every steps;
- adapter_step<N>.safetensors (train/export.py turns one into a runtime package);
- state.pt: optimizer, step and sampler state. A rerun with the same --out-dir resumes from it.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from bench.common import MAX_PIXELS, MIN_PIXELS
from train import data
from train.d3model import D3Trainable, Example


def loss_terms(logits: torch.Tensor, examples: list[Example], temperature: float, rps_weight: float):
    """Per-question KL, RPS and total loss ([B] each)."""
    valid = torch.isfinite(logits)
    target = torch.zeros_like(logits)
    for i, ex in enumerate(examples):
        target[i, : len(ex.target)] = torch.as_tensor(ex.target, dtype=torch.float32)
    logp = torch.log_softmax(logits / temperature, -1).masked_fill(~valid, 0.0)
    kl = (target * (target.clamp_min(1e-12).log() - logp)).sum(-1)
    p = logp.exp() * valid
    k = valid.sum(-1).clamp_min(2)
    rps = ((p.cumsum(-1) - target.cumsum(-1)) ** 2 * valid).sum(-1) / (k - 1)
    ordinal = torch.as_tensor([ex.ordinal for ex in examples], device=logits.device, dtype=torch.float32)
    return kl, rps, kl + rps_weight * ordinal * rps


def batch_stats(logits: torch.Tensor, examples: list[Example]) -> list[dict]:
    """Per question: top-1 agreement with the target's top option, probability of that option, and NLL of it."""
    probs = logits.detach().float().softmax(-1)
    out = []
    for i, ex in enumerate(examples):
        best = max(range(len(ex.target)), key=ex.target.__getitem__)
        p = float(probs[i, best])
        s = {"family": ex.group, "correct": int(int(probs[i].argmax()) == best), "p_top": p,
             "nll": -math.log(max(p, 1e-12))}
        if ex.question["type"] == "noul":  # index 1 is "yes": tracks the model's yes / no balance
            s["yes_pred"], s["yes_gold"] = int(int(probs[i, :2].argmax()) == 1), int(best == 1)
        out.append(s)
    return out


TOKEN_BUDGET = 6000  # padded tokens per forward pass (questions x the longest one) when a micro-batch is split


def est_tokens(ex: Example) -> int:
    """Rough input length: d3's image tokens (one per 32 x 32 px after its resize) plus ~160 text tokens."""
    n = 0
    for im in ex.images:
        h, w = smart_resize(im.height, im.width, factor=32, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        n += (h // 32) * (w // 32)
    return n + 160


def length_groups(examples: list[Example]) -> list[list[Example]]:
    """A micro-batch split into forward passes of similar length, each at most TOKEN_BUDGET padded tokens. Padding
    every question to the longest one made a batch with one DVL question (~3.7k tokens) cost four of them (9.9 s per
    step on 2 GPUs). The loss is unchanged: each pass is normalised by the micro-batch size."""
    groups, cur, cur_max = [], [], 0
    for ex in sorted(examples, key=est_tokens, reverse=True):
        t = est_tokens(ex)
        if cur and (len(cur) + 1) * max(cur_max, t) > TOKEN_BUDGET:
            groups.append(cur)
            cur, cur_max = [], 0
        cur.append(ex)
        cur_max = max(cur_max, t)
    return groups + ([cur] if cur else [])


def run_micro(model: D3Trainable, items: list, rng, args, backward: bool = True):
    """Forward (and backward) one micro-batch, in passes of similar length; on OOM, again one question at a time.
    Returns (loss sum, stats)."""
    examples = [data.to_example(it, rng if args.augment else None) for it in items]
    try:
        return _run(model, length_groups(examples), args, backward)
    except torch.cuda.OutOfMemoryError:
        pass  # retried outside the handler, so the failed pass's tensors are released first
    torch.cuda.empty_cache()
    print(f"OOM on a micro-batch of {len(examples)}; retrying one question at a time", flush=True)
    return _run(model, [[ex] for ex in examples], args, backward)


def _run(model, groups, args, backward):
    total, stats = 0.0, []
    for exs in groups:
        logits, _ = model.logits(exs)
        _, _, loss = loss_terms(logits, exs, model.temperature, args.rps_weight)
        if backward:
            (loss.sum() / (args.micro * args.accum_local)).backward()
        total += float(loss.detach().sum())
        stats += batch_stats(logits, exs)
    return total, stats


@torch.no_grad()
def evaluate(model: D3Trainable, dev: dict[str, list], args) -> dict:
    model.eval()
    per = collections.defaultdict(list)
    for fam, items in dev.items():
        for s in range(0, len(items), args.micro):
            exs = [data.to_example(it) for it in items[s: s + args.micro]]
            logits, _ = model.logits(exs)
            kl, _, _ = loss_terms(logits, exs, model.temperature, args.rps_weight)
            for st, k in zip(batch_stats(logits, exs), kl.tolist()):
                per[fam].append({**st, "kl": k})
    model.train()
    out = {}
    for fam, v in per.items():
        if not v:
            continue
        out[fam] = {"n": len(v), "acc": sum(x["correct"] for x in v) / len(v), "kl": sum(x["kl"] for x in v) / len(v),
                    "nll_top": sum(x["nll"] for x in v) / len(v)}
        yn = [x for x in v if "yes_pred" in x]
        if yn:  # predicted vs gold yes share over the family's yes / no questions
            out[fam]["yes_pred"] = round(sum(x["yes_pred"] for x in yn) / len(yn), 3)
            out[fam]["yes_gold"] = round(sum(x["yes_gold"] for x in yn) / len(yn), 3)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--micro", type=int, default=4, help="questions per forward pass")
    p.add_argument("--accum", type=int, default=4, help="micro-batches per optimizer step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=float, default=64.0)
    p.add_argument("--merger-rank", type=int, default=64, help="LoRA rank on the vision merger (0: frozen)")
    p.add_argument("--vision-rank", type=int, default=16, help="LoRA rank on the vision blocks (0: frozen)")
    p.add_argument("--vision-lr-scale", type=float, default=0.2, help="vision-block learning rate / text learning rate")
    p.add_argument("--rps-weight", type=float, default=0.25)
    p.add_argument("--sord-count", type=float, default=10.0)
    p.add_argument("--sord-level", type=float, default=2.0)
    p.add_argument("--replay-share", type=float, default=0.2)
    p.add_argument("--mix", default=data.DEFAULT_MIX,
                   help="family weights, e.g. fam1_labels=0.3,fam2_counting=0.2,... ('' for the default_weights rule)")
    p.add_argument("--limit-per-family", type=int, help="train on a fixed random subset of rows per family")
    p.add_argument("--dev-per-family", type=int, default=120)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:  # the others wait at a barrier while GPU 0 runs the dev evaluation: allow an hour
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"), timeout=timedelta(hours=1))
    if args.accum % world:
        raise ValueError(f"--accum {args.accum} must be a multiple of the {world} GPUs")
    args.accum_local = args.accum // world
    main_rank = rank == 0
    say = print if main_rank else (lambda *a, **k: None)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed + 1000 * rank)
    torch.manual_seed(args.seed)

    train_items = data.load_items("train", limit_per_family=args.limit_per_family, seed=args.seed,
                                  sord_count=args.sord_count, sord_level=args.sord_level)
    dev_items = {k: v[: args.dev_per_family] for k, v in data.load_items(
        "dev", seed=args.seed, sord_count=args.sord_count, sord_level=args.sord_level).items()}
    weights = (dict((kv.split("=")[0], float(kv.split("=")[1])) for kv in args.mix.split(","))
               if args.mix else data.default_weights(train_items, args.replay_share))
    mixer = data.Mixer(train_items, weights, seed=args.seed + 1000 * rank)
    say("train questions per family:", {k: len(v) for k, v in train_items.items()}, flush=True)
    say("weights:", {k: round(v, 3) for k, v in weights.items()}, flush=True)
    say(f"GPUs {world}: {args.accum_local} micro-batches of {args.micro} per GPU per step", flush=True)
    missing = sorted(set(weights) - set(mixer.names))
    if missing:
        raise ValueError(f"families with a weight but no training rows: {missing}")
    replay_has_d3 = any(it.row.get("meta", {}).get("d3_probs") for it in train_items.get("replay", [])[:100])
    say("replay targets:", "d3-flash's own probabilities" if replay_has_d3 else "gold (replay_d3.jsonl missing)",
        flush=True)

    model = D3Trainable(device=f"cuda:{local_rank}")
    n_params = model.add_lora(args.rank, args.alpha)
    n_params += model.add_vision_lora(args.merger_rank, args.vision_rank, scale=args.alpha / args.rank)
    model.enable_checkpointing()
    model.train()
    params = model.lora_parameters()
    if world > 1:  # every GPU starts from GPU 0's adapters
        for prm in params:
            dist.broadcast(prm.data, 0)
    groups = model.lora_groups()
    opt = torch.optim.AdamW([g for g in ({"params": groups["text"] + groups["merger"], "lr_scale": 1.0},
                                         {"params": groups["vision"], "lr_scale": args.vision_lr_scale}) if g["params"]],
                            lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    say("adapter parameters by part:", {k: sum(p.numel() for p in v) for k, v in groups.items()}, flush=True)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))

    start = 0
    if (out / "state.pt").exists():
        state = torch.load(out / "state.pt", weights_only=False)
        model.load_lora(out / f"adapter_step{state['step']}.safetensors")
        opt.load_state_dict(state["opt"])
        start = state["step"]
        own_file = out / f"state_rank{rank}.pt"
        if rank == 0 or own_file.exists():
            own = torch.load(own_file, weights_only=False) if rank else state
            mixer.load_state(own["mixer"])
            random.setstate(own["random"])
        else:  # a GPU added on resume (e.g. 2 -> 4 GPUs) starts its own sampler stream
            mixer = data.Mixer(train_items, weights, seed=args.seed + 1000 * rank + start)
        say(f"resumed at step {start} on {world} GPUs", flush=True)
    elif main_rank:
        (out / "args.json").write_text(json.dumps({**vars(args), "world": world, "lora_params": n_params,
                                                   "weights": weights}, indent=1))
        say(f"LoRA r={args.rank} alpha={args.alpha}: {n_params / 1e6:.1f}M trainable parameters on "
            f"{len(model.lora)} layers", flush=True)
    rng = random.Random(f"aug:{args.seed}:{rank}:{start}")
    log = open(out / "log.jsonl", "a") if main_rank else None

    def record(rec: dict) -> None:
        if main_rank:
            log.write(json.dumps(rec) + "\n"), log.flush()
            print(json.dumps(rec), flush=True)

    def save(step: int) -> None:
        if rank:  # the other GPUs keep only their own sampler state; GPU 0 writes it last, after the adapter
            torch.save({"mixer": mixer.state(), "random": random.getstate()}, out / f"state_rank{rank}.pt")
        if world > 1:  # the adapters must be identical on every GPU: compare a checksum
            s = torch.stack([prm.detach().float().abs().sum() for prm in params]).sum().reshape(1)
            all_s = [torch.zeros_like(s) for _ in range(world)]
            dist.all_gather(all_s, s)
            vals = [float(x) for x in all_s]
            if max(vals) - min(vals) > 1e-4 * abs(vals[0]):
                say(f"WARNING step {step}: adapter checksums differ across GPUs {vals}", flush=True)
            dist.barrier()
        if main_rank:
            model.save_lora(out / f"adapter_step{step}.safetensors", {"step": step, "args": vars(args)})
            torch.save({"step": step, "opt": opt.state_dict(), "mixer": mixer.state(), "random": random.getstate()},
                       out / "state.pt.tmp")
            (out / "state.pt.tmp").replace(out / "state.pt")
        if world > 1:
            dist.barrier()

    def dev_eval(step: int) -> None:
        if main_rank:
            record({"step": step, "dev": evaluate(model, dev_items, args)})
        if world > 1:
            dist.barrier()

    if start == 0:
        dev_eval(0)

    window, t0 = [], time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(start, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step) * g.get("lr_scale", 1.0)
        loss_sum = 0.0
        for _ in range(args.accum_local):
            loss, stats = run_micro(model, mixer.sample(args.micro), rng, args)
            loss_sum += loss
            window += stats
        if world > 1:  # average the gradients over GPUs (one flat buffer), and the logged loss
            for prm in params:  # a GPU whose questions were all text has no vision-adapter gradient: count it as 0
                if prm.grad is None:
                    prm.grad = torch.zeros_like(prm)
            grads = [prm.grad for prm in params]
            flat = torch.cat([g.reshape(-1) for g in grads])
            dist.all_reduce(flat)
            flat /= world
            k = 0
            for g in grads:
                g.copy_(flat[k: k + g.numel()].view_as(g))
                k += g.numel()
            loss_t = torch.tensor(loss_sum, device=flat.device)
            dist.all_reduce(loss_t)
            loss_sum = float(loss_t)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        done = step + 1
        if done % args.log_every == 0:
            by = collections.defaultdict(list)
            for s in window:
                by[s["family"]].append(s["correct"])
            record({"step": done, "lr": lr_at(step), "loss": loss_sum / (args.micro * args.accum),
                    "grad_norm": grad_norm, "acc": {k: round(sum(v) / len(v), 3) for k, v in by.items()},
                    "sec_per_step": (time.perf_counter() - t0) / args.log_every,
                    "peak_gb": torch.cuda.max_memory_allocated() / 2**30})
            window, t0 = [], time.perf_counter()
        if done % args.eval_every == 0 or done == args.steps:
            dev_eval(done)
        if done % args.save_every == 0 or done == args.steps:
            save(done)
    say("done", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
