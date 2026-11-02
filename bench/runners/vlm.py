"""Run an EO specialist model (bench/specialists) over a rows file; same chunking, resumability and result format as
bench/runners/d3.py, so bench/metrics/score.py scores both alike.

    python -m bench.runners.vlm --model geozero --mode logprob|generate --rows <rows.jsonl> --out-dir <run dir>
        [--chunks 64] [--limit N] [--group 8]

logprob: one forward pass per question; option probabilities from the model's answer tokens (bench/specialists/
prompting.py). generate: greedy decoding in the model's own reasoning format; the answer is parsed from the text, the
raw text is kept in the record ("raw"), and parse failures are flagged ("unparsed": true in the answer).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import socket
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from bench.common import CONVERTED, read_jsonl
from bench.runners.d3 import claim
from bench.specialists import load
from bench.specialists.native import parse_native, render_native, to_ours_probs
from bench.specialists.prompting import parse_answer, render, to_answer, to_answer_generated


def _fit(im: Image.Image, max_pixels: int | None) -> Image.Image:
    if max_pixels and im.width * im.height > max_pixels:  # as LLaMA-Factory's image_max_pixels
        scale = (max_pixels / (im.width * im.height)) ** 0.5
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.Resampling.LANCZOS)
    return im


def run_group_native(model, model_name: str, rows: list[dict], mode: str) -> list[dict]:
    """--native: each question in the model's own format (bench/specialists/native.py), its labels mapped to ours.
    A question with several views (e.g. the tiles of a scene) is asked once per view and the answers are combined:
    mean of the mapped probabilities, or a majority vote of the parsed written answers."""
    nats, per_row = [], []
    for row in rows:
        row_nats = []
        for qid in row["questions"]:
            nat = render_native(model_name, row, qid)
            if nat is None:
                raise ValueError(f"{model_name} has no native format for {row['benchmark']}")
            row_nats.append(nat)
        nats += row_nats
        per_row.append(row_nats)
    items, owner = [], []
    for k, nat in enumerate(nats):
        full = []
        for rel in nat.image_paths:
            with Image.open(CONVERTED / rel) as im:
                full.append(im.convert("RGB"))
        for crop in nat.crops or [None]:
            views = [_fit(im.crop(crop) if crop else im, nat.max_pixels) for im in full]
            items.append(dataclasses.replace(nat.item, images=views))
            owner.append(k)
    answers, raws = [], [None] * len(nats)
    if mode == "logprob":
        sums = [None] * len(nats)
        for k, p in zip(owner, model.score(items)):
            q = to_ours_probs(nats[k], p)
            sums[k] = q if sums[k] is None else [a + b for a, b in zip(sums[k], q)]
        answers = [to_answer(nat.ours, s) for nat, s in zip(nats, sums)]
    else:
        texts = model.generate(items)
        votes = [[] for _ in nats]
        outs = [[] for _ in nats]
        for k, t in zip(owner, texts):
            outs[k].append(t)
            v = parse_native(nats[k], t)
            if v is not None:
                votes[k].append(v)
        for k, nat in enumerate(nats):
            best = Counter(votes[k]).most_common(1)[0][0] if votes[k] else None
            answers.append(to_answer_generated(nat.ours, best))
            raws[k] = outs[k][0] if len(outs[k]) == 1 else " | ".join(o.strip() for o in outs[k])
    records, k = [], 0
    for row, row_nats in zip(rows, per_row):
        rec = {"answers": {}}
        for nat in row_nats:
            rec["answers"][nat.item.qid] = answers[k]
            if mode == "generate":
                rec.setdefault("raw", {})[nat.item.qid] = (raws[k] or "")[-3000:]
            k += 1
        records.append(rec)
    return records


def run_group(model, rows: list[dict], mode: str) -> list[dict]:
    items, per_row = [], []
    for row in rows:
        images = []
        for rel in row["images"]:
            with Image.open(CONVERTED / rel) as im:
                images.append(im.convert("RGB"))
        row_items = []
        for qid in row["questions"]:
            it = render(row, qid)
            it.images = images
            row_items.append(it)
        items += row_items
        per_row.append(row_items)
    if mode == "logprob":
        outs = model.score(items)
        answers = [to_answer(it, p) for it, p in zip(items, outs)]
        raws = [None] * len(items)
    else:
        raws = model.generate(items)
        answers = [to_answer_generated(it, parse_answer(it, t)) for it, t in zip(items, raws)]
    by_key = {(it.row_id, it.qid): (a, r) for it, a, r in zip(items, answers, raws)}
    records = []
    for row, row_items in zip(rows, per_row):
        rec = {"answers": {it.qid: by_key[(row["id"], it.qid)][0] for it in row_items}}
        if mode == "generate":
            rec["raw"] = {it.qid: (by_key[(row["id"], it.qid)][1] or "")[-3000:] for it in row_items}
        records.append(rec)
    return records


def _run(model, rows: list[dict], args) -> list[dict]:
    if args.native:
        return run_group_native(model, args.model, rows, args.mode)
    return run_group(model, rows, args.mode)


def run_chunk(model, rows: list[dict], path: Path, args, variant: dict) -> int:
    done = {json.loads(line)["id"] for line in path.open()} if path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    n = 0
    with path.open("a") as f:
        for s in range(0, len(todo), args.group):
            group = todo[s: s + args.group]
            t = time.perf_counter()
            try:
                try:
                    recs = _run(model, group, args)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()  # retry the group one row at a time
                    recs = [_run(model, [r], args)[0] for r in group]
                ms = round((time.perf_counter() - t) * 1000 / len(group), 2)
                for row, rec in zip(group, recs):
                    f.write(json.dumps({"id": row["id"], "benchmark": row["benchmark"], "variant": variant,
                                        "status": "ok", **rec, "wall_ms": ms}) + "\n")
            except Exception as exc:  # noqa: BLE001 - retry rows one by one so one bad row does not sink the group
                torch.cuda.empty_cache()
                for row in group:
                    rec = {"id": row["id"], "benchmark": row["benchmark"], "variant": variant}
                    try:
                        rec.update(status="ok", **_run(model, [row], args)[0])
                    except Exception as exc1:  # noqa: BLE001
                        rec.update(status="error", error=f"{type(exc1).__name__}: {exc1}",
                                   trace=traceback.format_exc(limit=3), group_error=f"{type(exc).__name__}")
                    f.write(json.dumps(rec) + "\n")
            f.flush()
            n += len(group)
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=["logprob", "generate"], required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--chunks", type=int, default=64)
    p.add_argument("--limit", type=int)
    p.add_argument("--group", type=int, help="rows per batch (default 8 for logprob, 16 for generate)")
    p.add_argument("--batch-size", type=int, help="sequences per forward / generate call, for adapters that batch "
                   "(default: the adapter's, raised to 16 for generate)")
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--native", action="store_true", help="ask in the model's own format (bench/specialists/native.py)")
    args = p.parse_args()

    rows = list(read_jsonl(args.rows))[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variant = {"model": args.model, "mode": args.mode, **({"native": True} if args.native else {})}
    host = f"{socket.gethostname()} GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}"
    try:
        model = load(args.model, **({"max_new_tokens": args.max_new_tokens} if args.max_new_tokens else {}))
    except Exception as exc:  # noqa: BLE001 - a dead GPU: leave the chunks to the other processes
        print(f"{host}: model load failed, claiming nothing ({type(exc).__name__}: {exc})", flush=True)
        traceback.print_exc()
        sys.exit(3)
    if args.group is None:
        args.group = 16 if args.mode == "generate" else 8
    if hasattr(model, "batch_size") and getattr(model, "batch_size", 1) > 1:  # adapters that batch (bench/specialists/hf_vlm.py)
        model.batch_size = args.batch_size or (16 if args.mode == "generate" else model.batch_size)
    print(f"{host}: {args.model} loaded; {len(rows)} rows in {args.chunks} chunks ({args.mode})", flush=True)

    started, total = time.perf_counter(), 0
    for c in range(args.chunks):
        if (out_dir / f"chunk{c}.done").exists() or not claim(out_dir, c):
            continue
        todo = [r for i, r in enumerate(rows) if i % args.chunks == c]
        total += run_chunk(model, todo, out_dir / f"chunk{c}.jsonl", args, variant)
        (out_dir / f"chunk{c}.done").touch()
        rate = total / max(1e-9, time.perf_counter() - started)
        print(f"{host}: chunk {c} done ({total} rows so far, {rate:.2f} rows/s)", flush=True)
    print(f"{host}: finished, {total} rows in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
