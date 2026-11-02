"""Leak blocklist, eval side: fingerprint every image the eval row files reference, and collect the source ids their
rows record, so training sources can be checked against both.

Row files are the *.jsonl one or two levels under $EOJEV_DATASETS/converted (image folders are not walked). Variant
files of the same benchmark (e.g. tiled versions) are included on purpose: their tiles help catch crops.

Outputs in $EOJEV_DATASETS/trainset/blocklist/:
- eval_images.jsonl      one line per unique image: path, sha256, size, benchmarks, rows
- eval_hashes.npz        hash (uint64), image index, grid, i, j for every whole-image and tile pHash
- eval_source_ids.jsonl  per row: file, benchmark, id, group_id and the meta values that name a source item
                         (keys containing id / name / file / scene / image / source / tile / event / patch / uid)
- eval_index_errors.jsonl images that could not be read

Run as a CPU job (never on the login node):
    OPENBLAS_NUM_THREADS=1 MODULE=trainset.eval_index sbatch slurm/convert_cpu.sbatch
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from bench.common import CONVERTED, DATASETS, read_jsonl
from trainset.hashing import fingerprint

OUT = DATASETS / "trainset" / "blocklist"
SKIP_DIRS = {"images", "frames", "media", "tiles", "raw"}
ID_KEY = re.compile(r"(id|name|file|scene|image|source|tile|event|patch|uid)", re.I)


def row_files() -> list[str]:
    files = []
    for top in sorted(os.scandir(CONVERTED), key=lambda e: e.name):
        if top.is_file() and top.name.endswith(".jsonl"):
            files.append(top.path)
        elif top.is_dir() and top.name not in SKIP_DIRS:
            for e in sorted(os.scandir(top.path), key=lambda e: e.name):
                if e.is_file() and e.name.endswith(".jsonl"):
                    files.append(e.path)
                elif e.is_dir() and e.name not in SKIP_DIRS:
                    files += sorted(f.path for f in os.scandir(e.path) if f.is_file() and f.name.endswith(".jsonl"))
    return files


def source_ids(meta, prefix="", depth=0, out=None) -> dict:
    """Meta values whose key looks like a source identifier (strings / ints, or short lists of them)."""
    out = {} if out is None else out
    if depth > 3 or not isinstance(meta, dict):
        return out
    for k, v in meta.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            source_ids(v, key + ".", depth + 1, out)
        elif ID_KEY.search(k):
            if isinstance(v, (str, int)) and not isinstance(v, bool):
                out[key] = v
            elif isinstance(v, list) and 0 < len(v) <= 32 and all(isinstance(x, (str, int)) for x in v):
                out[key] = v
    return out


def _fp(job):
    idx, rel = job
    try:
        return idx, fingerprint(CONVERTED / rel), None
    except Exception as exc:  # noqa: BLE001 - record and continue; a missing image must not stop the index
        return idx, None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    images: dict[str, int] = {}
    refs: dict[int, dict] = defaultdict(lambda: {"benchmarks": set(), "rows": 0})
    n_rows = 0
    with open(OUT / "eval_source_ids.jsonl", "w") as sid:
        for f in row_files():
            rel_file = os.path.relpath(f, CONVERTED)
            for row in read_jsonl(f):
                if not isinstance(row, dict) or "images" not in row:
                    continue
                n_rows += 1
                for p in row["images"]:
                    k = images.setdefault(p, len(images))
                    refs[k]["benchmarks"].add(row.get("benchmark", "?"))
                    refs[k]["rows"] += 1
                sid.write(json.dumps({"file": rel_file, "benchmark": row.get("benchmark"), "id": row.get("id"),
                                      "group_id": row.get("group_id"),
                                      "ids": source_ids(row.get("meta", {}))}, ensure_ascii=False) + "\n")
    print(f"{n_rows} rows, {len(images)} unique images ({time.time() - t0:.0f}s)", flush=True)

    paths = [None] * len(images)
    for p, k in images.items():
        paths[k] = p
    fps: list = [None] * len(paths)
    n_proc = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    with Pool(n_proc) as pool, open(OUT / "eval_index_errors.jsonl", "w") as err:
        for done, (k, fp, e) in enumerate(pool.imap_unordered(_fp, enumerate(paths), chunksize=32), 1):
            if e:
                err.write(json.dumps({"path": paths[k], "error": e}) + "\n")
            fps[k] = fp
            if done % 5000 == 0:
                print(f"  {done}/{len(paths)} images ({time.time() - t0:.0f}s)", flush=True)

    hashes, owner, grid, ti, tj = [], [], [], [], []
    with open(OUT / "eval_images.jsonl", "w") as f:
        for k, p in enumerate(paths):
            fp = fps[k]
            f.write(json.dumps({"path": p, "sha256": fp and fp["sha256"], "size": fp and fp["size"],
                                "benchmarks": sorted(refs[k]["benchmarks"]), "rows": refs[k]["rows"]}) + "\n")
            for g, i, j, h in (fp["hashes"] if fp else []):
                hashes.append(h), owner.append(k), grid.append(g), ti.append(i), tj.append(j)
    np.savez_compressed(OUT / "eval_hashes.npz", hash=np.array(hashes, dtype=np.uint64),
                        image=np.array(owner, dtype=np.int32), grid=np.array(grid, dtype=np.int8),
                        i=np.array(ti, dtype=np.int8), j=np.array(tj, dtype=np.int8))
    n_err = sum(fp is None for fp in fps)
    print(f"done: {len(paths)} images, {len(hashes)} hashes, {n_err} unreadable, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
