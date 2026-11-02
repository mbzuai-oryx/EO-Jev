"""Run d3-flash over a rows file and write one result line per row (resumable).

Work is split into --chunks pieces (row i belongs to chunk i % chunks). Each process loads the model first, then
claims chunks one at a time by creating <out dir>/chunk<c>.lock exclusively (O_EXCL on the shared filesystem),
writes <out dir>/chunk<c>.jsonl and marks it done with chunk<c>.done. A process whose GPU fails to initialize
exits before claiming anything, so the healthy processes finish the whole file; some cluster nodes advertise
GPUs that no longer initialize (docs: gpu-55, one GPU each on gpu-05 / gpu-52 on 2026-10-16).

    python -m bench.runners.d3 --rows <rows.jsonl> --out-dir <run dir> [--chunks 64]
        [--image-transform rot90|hflip|down2|...] [--blind] [--shuffle-images] [--permutation-average]

--blind sends no image (language-prior control); --shuffle-images pairs every row with another row's images
(fixed seed), the shuffled-image control of docs/03 §9. A crashed run is resumed by clearing locks that have
no .done marker (slurm/run_d3.sbatch does this before starting) and running again.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image

from bench.common import CONVERTED, D3_FLASH, RUNS, read_jsonl, transform_image


# Models run through d3's runtime: d3-flash itself, and Qwen3.5-9B packaged with its own lm_head rows as the
# readout and causal attention (scripts/make_base_package.py) as the same-backbone control.
MODELS = {
    "d3-flash": (D3_FLASH, "fast"),
    "qwen3.5-9b": (RUNS.parent / "packages" / "qwen3.5-9b-d3readout", "none"),
}


def load_runtime(model: str, batch_size: int, permutation_average: bool):
    """A model of MODELS, or the directory of a package exported by train/export.py (loaded unverified)."""
    sys.path.insert(0, str(D3_FLASH))
    from d3_runtime import D3  # noqa: PLC0415 - shipped with the checkpoint

    root, verify = MODELS.get(model) or (Path(model), "none")
    name = model if model in MODELS else Path(model).name
    return D3.from_pretrained(
        str(root), device="cuda:0", batch_size=batch_size, verify=verify,
        permutation_average=permutation_average, model_name=name,
    )


def claim(out_dir: Path, chunk: int) -> bool:
    try:
        fd = os.open(out_dir / f"chunk{chunk}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{socket.gethostname()} pid={os.getpid()} {time.ctime()}\n".encode())
    os.close(fd)
    return True


def run_chunk(runtime, todo: list[tuple[dict, list[str]]], path: Path, args, variant: dict) -> int:
    done = {json.loads(line)["id"] for line in path.open()} if path.exists() else set()
    n = 0
    with path.open("a") as f:
        for row, imgs in todo:
            if row["id"] in done:
                continue
            record = {"id": row["id"], "benchmark": row["benchmark"], "variant": variant}
            try:
                images = []
                if not args.blind:
                    for rel in imgs:
                        with Image.open(CONVERTED / rel) as im:
                            images.append(transform_image(im.convert("RGB"), args.image_transform))
                torch.cuda.synchronize()
                t = time.perf_counter()
                try:
                    result = runtime.system_one(state=row["state"], questions=row["questions"], images=images or None)
                except torch.cuda.OutOfMemoryError:
                    # Many questions x many large images (DVL, XLRS tiles) can exceed 40 GB in one batch, all the
                    # more on GPUs shared with other jobs: retry this row one question per forward pass.
                    torch.cuda.empty_cache()
                    saved, runtime.batch_size = runtime.batch_size, 1
                    try:
                        result = runtime.system_one(state=row["state"], questions=row["questions"], images=images or None)
                    finally:
                        runtime.batch_size = saved
                    record["oom_retry"] = True
                torch.cuda.synchronize()
                record.update(status="ok", answers=result["answers"], usage=result["usage"],
                              wall_ms=round((time.perf_counter() - t) * 1000, 2))
            except Exception as exc:  # noqa: BLE001 - one bad row must not stop the run
                record.update(status="error", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc(limit=3))
            f.write(json.dumps(record) + "\n")
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default="d3-flash", help=f"{', '.join(sorted(MODELS))} or a package directory")
    p.add_argument("--chunks", type=int, default=64)
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-transform")
    p.add_argument("--blind", action="store_true")
    p.add_argument("--shuffle-images", action="store_true")
    p.add_argument("--permutation-average", action="store_true")
    args = p.parse_args()

    rows = list(read_jsonl(args.rows))[: args.limit]
    image_lists = [r["images"] for r in rows]
    if args.shuffle_images:
        random.Random(0).shuffle(image_lists)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variant = {"model": args.model, "image_transform": args.image_transform, "blind": args.blind,
               "shuffle_images": args.shuffle_images, "permutation_average": args.permutation_average}

    host = f"{socket.gethostname()} GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}"
    try:
        runtime = load_runtime(args.model, args.batch_size, args.permutation_average)
    except Exception as exc:  # noqa: BLE001 - a dead GPU: leave the chunks to the other processes
        print(f"{host}: model load failed, claiming nothing ({type(exc).__name__}: {exc})", flush=True)
        sys.exit(3)
    print(f"{host}: model loaded; {len(rows)} rows in {args.chunks} chunks", flush=True)

    started, total = time.perf_counter(), 0
    for c in range(args.chunks):
        if (out_dir / f"chunk{c}.done").exists() or not claim(out_dir, c):
            continue
        todo = [(r, im) for i, (r, im) in enumerate(zip(rows, image_lists)) if i % args.chunks == c]
        total += run_chunk(runtime, todo, out_dir / f"chunk{c}.jsonl", args, variant)
        (out_dir / f"chunk{c}.done").touch()
        rate = total / max(1e-9, time.perf_counter() - started)
        print(f"{host}: chunk {c} done ({total} rows so far, {rate:.2f} rows/s)", flush=True)
    print(f"{host}: finished, {total} rows in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
