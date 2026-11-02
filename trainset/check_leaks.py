"""Leak check of the training images against the eval image index.

Every image used by the training row files is fingerprinted like the eval images (trainset/hashing.py: sha256, pHash of
the whole image and of its 2x2 / 3x3 tiles) and matched against $EOJEV_DATASETS/trainset/blocklist/eval_hashes.npz.

Matching: candidate pairs share one of four 16-bit blocks of the 64-bit hash, then the Hamming distance is checked.
Featureless content (open sea, bare desert) gives similar hashes everywhere, so:
- a training hash that matches more than 20 different eval images is ignored;
- bucket keys with more than 5,000 eval hashes are skipped.

Verdict per (training image, eval image) pair:
- likely leak: identical bytes; or whole images at distance <= 6; or >= 3 matching hash pairs (tile-level evidence
  that one is a crop of the other); or 2 pairs with one at distance <= 3.
- possible: any other match, which is reported but not dropped.

Outputs in $EOJEV_DATASETS/trainset/leaks/: train_fingerprints.jsonl, matches.jsonl and a summary printout.
With --apply it also writes rows/final/<file>.jsonl without the rows that use a likely-leaked image, plus
rows/final/manifest.json.

    OPENBLAS_NUM_THREADS=1 MODULE=trainset.check_leaks [ARGS="--apply"] sbatch -J eojev-leaks slurm/convert_cpu.sbatch
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from bench.common import read_jsonl, write_jsonl
from trainset.common import ROWS, TRAIN
from trainset.hashing import fingerprint, hamming

FILES = ["fam1_labels", "fam2_counting", "fam9_abstention", "fam4_damage", "replay",
         "fam10_multiselect", "fam11_geo", "fam12_presence_lr",
         # then the benchmarks' own training splits (trainset/bench_train.py)
         "fam13_buildings", "fam14_count_distill", "fam15_rsvqa", "fam16_vrsbench", "fam17_cdvqa", "fam18_disasterm3",
         "fam19_nwpu", "fam20_dvl"]
OUT = TRAIN / "leaks"
BL = TRAIN / "blocklist"
# Folders under converted/ that are not evaluation data: probes/ holds the linear-probe experiment's training
# images (docs/04 §8: e.g. AID's 7,000 non-test images and EuroSAT train), so matching them is not a leak.
NON_EVAL_PREFIXES = ("probes/",)
MAX_EVAL_IMAGES_PER_HASH = 20
MAX_BUCKET = 5000
T_WHOLE, T_TILE = 6, 5


def _fp(rel: str):
    try:
        return rel, fingerprint(TRAIN / rel), None
    except Exception as exc:  # noqa: BLE001
        return rel, None, f"{type(exc).__name__}: {exc}"


def candidate_pairs(eval_h: np.ndarray, q: np.ndarray):
    """(query index, eval index) pairs that share at least one 16-bit block."""
    out_q, out_e = [], []
    for b in range(4):
        ke = ((eval_h >> np.uint64(16 * b)) & np.uint64(0xFFFF)).astype(np.int64)
        order = np.argsort(ke, kind="stable")
        ks = ke[order]
        kq = ((q >> np.uint64(16 * b)) & np.uint64(0xFFFF)).astype(np.int64)
        lo, hi = np.searchsorted(ks, kq, "left"), np.searchsorted(ks, kq, "right")
        n = hi - lo
        n[n > MAX_BUCKET] = 0
        if n.sum() == 0:
            continue
        qi = np.repeat(np.arange(len(q)), n)
        starts = np.repeat(lo, n)
        offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        out_q.append(qi)
        out_e.append(order[starts + offs])
    if not out_q:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    qi, ei = np.concatenate(out_q), np.concatenate(out_e)
    pairs = np.unique(np.stack([qi, ei], 1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="write rows/final without likely-leaked rows")
    p.add_argument("--keep-previous", metavar="NAME",
                   help="first copy the current rows/final to rows/final_NAME (once), so an earlier run's data stays")
    args = p.parse_args()
    if args.apply and args.keep_previous and (ROWS / "final").is_dir():
        import shutil

        dest = ROWS / f"final_{args.keep_previous}"
        if not dest.exists():
            shutil.copytree(ROWS / "final", dest)
            print(f"kept the previous rows/final as {dest}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows_by_file = {f: list(read_jsonl(ROWS / f"{f}.jsonl")) for f in FILES if (ROWS / f"{f}.jsonl").exists()}
    images = sorted({im for rows in rows_by_file.values() for r in rows for im in r["images"]})
    print(f"{sum(map(len, rows_by_file.values()))} rows in {list(rows_by_file)}; {len(images)} training images", flush=True)

    fps = {}
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))) as pool:
        for k, (rel, fp, err) in enumerate(pool.imap_unordered(_fp, images, chunksize=32), 1):
            fps[rel] = fp
            if k % 10000 == 0:
                print(f"  fingerprinted {k}/{len(images)} ({time.time() - t0:.0f}s)", flush=True)
    with open(OUT / "train_fingerprints.jsonl", "w") as f:
        for rel in images:
            f.write(json.dumps({"path": rel, **(fps[rel] or {"error": True})}) + "\n")

    ev = np.load(BL / "eval_hashes.npz")
    eval_meta = [json.loads(l) for l in open(BL / "eval_images.jsonl")]
    is_eval = np.array([not m["path"].startswith(NON_EVAL_PREFIXES) for m in eval_meta])
    keep_h = is_eval[ev["image"]]
    eval_h, eval_img, eval_grid = ev["hash"][keep_h], ev["image"][keep_h], ev["grid"][keep_h]
    eval_sha = {m["sha256"]: k for k, m in enumerate(eval_meta) if m["sha256"] and is_eval[k]}
    print(f"  eval images {int(is_eval.sum())} (ignoring {int((~is_eval).sum())} non-eval: {NON_EVAL_PREFIXES})",
          flush=True)

    # training hash table
    q_h, q_img, q_grid = [], [], []
    for k, rel in enumerate(images):
        for g, _i, _j, h in (fps[rel] or {}).get("hashes", []):
            q_h.append(h), q_img.append(k), q_grid.append(g)
    q_h = np.array(q_h, dtype=np.uint64)
    q_img, q_grid = np.array(q_img), np.array(q_grid)
    print(f"  {len(q_h)} training hashes vs {len(eval_h)} eval hashes", flush=True)

    evidence: dict[tuple[int, int], list] = collections.defaultdict(list)
    chunk = 20000
    for s in range(0, len(q_h), chunk):
        qi, ei = candidate_pairs(eval_h, q_h[s:s + chunk])
        if len(qi) == 0:
            continue
        d = hamming(q_h[s:s + chunk][qi], eval_h[ei])
        whole = (q_grid[s + qi] == 1) & (eval_grid[ei] == 1)
        keep = np.where(whole, d <= T_WHOLE, d <= T_TILE)
        qi, ei, d, whole = qi[keep], ei[keep], d[keep], whole[keep]
        # drop featureless hashes: a training hash matching too many different eval images
        per_q = collections.defaultdict(set)
        for a, e in zip(qi.tolist(), eval_img[ei].tolist()):
            per_q[a].add(e)
        for a, e_idx, dist, w in zip(qi.tolist(), ei.tolist(), d.tolist(), whole.tolist()):
            if len(per_q[a]) > MAX_EVAL_IMAGES_PER_HASH:
                continue
            evidence[(int(q_img[s + a]), int(eval_img[e_idx]))].append((int(dist), bool(w)))
        print(f"  matched hashes {min(s + chunk, len(q_h))}/{len(q_h)} ({time.time() - t0:.0f}s)", flush=True)

    likely, possible = {}, {}

    def small(size) -> bool:  # under 256 px a side: no tile hashes, so only the whole-image hash can match
        return bool(size) and min(size) < 256

    for (ti, ei_), ev_ in evidence.items():
        dists = sorted(x[0] for x in ev_)
        whole_d = min((x[0] for x in ev_ if x[1]), default=99)
        n = len(ev_)
        no_tiles = small((fps[images[ti]] or {}).get("size")) or small(eval_meta[ei_]["size"])
        # Real duplicates and crops agree on many hashes (typically all 14). A single whole-image hash at distance
        # 4-6 alone is what unrelated, featureless images produce (seen against SIS-Bench drone frames).
        if no_tiles:
            hit = whole_d <= 3
        else:
            hit = whole_d <= 2 or (whole_d <= T_WHOLE and n >= 3) or n >= 3 and dists[0] <= 3 or n >= 5
        verdict = "likely" if hit else "possible"
        (likely if verdict == "likely" else possible).setdefault(images[ti], []).append(
            {"eval": eval_meta[ei_]["path"], "benchmarks": eval_meta[ei_]["benchmarks"], "n": len(ev_),
             "min_dist": dists[0], "whole_dist": whole_d if whole_d < 99 else None})
    for rel in images:
        fp = fps[rel]
        if fp and fp["sha256"] in eval_sha:
            k = eval_sha[fp["sha256"]]
            likely.setdefault(rel, []).append({"eval": eval_meta[k]["path"], "benchmarks": eval_meta[k]["benchmarks"],
                                               "exact": True})
    with open(OUT / "matches.jsonl", "w") as f:
        for tag, d in (("likely", likely), ("possible", possible)):
            for rel, ms in sorted(d.items()):
                f.write(json.dumps({"train": rel, "verdict": tag, "matches": ms}) + "\n")

    src = lambda rel: rel.split("/")[1]  # noqa: E731 - images/<source>/...
    kind = lambda m: "exact" if m.get("exact") else "whole" if m.get("whole_dist") is not None else "tile"  # noqa: E731
    combos = collections.Counter()
    examples = collections.defaultdict(list)
    for rel, ms in likely.items():
        for m in ms:
            for b in m["benchmarks"]:
                key = (src(rel), b, kind(m))
                combos[key] += 1
                if len(examples[key]) < 4:
                    examples[key].append(f"{rel} <-> {m['eval']} (n={m.get('n')}, min={m.get('min_dist')}, "
                                         f"whole={m.get('whole_dist')})")
    with open(OUT / "examples.txt", "w") as f:
        for key, n in combos.most_common():
            f.write(f"{key[0]} vs {key[1]} [{key[2]}]: {n}\n")
            for e in examples[key]:
                f.write(f"    {e}\n")
    print("likely leaks by (training source, eval benchmark, kind):",
          {f"{a}|{b}|{c}": n for (a, b, c), n in combos.most_common(30)})
    print("likely leaks by training source:", dict(collections.Counter(src(r) for r in likely)))
    print("possible by training source:", dict(collections.Counter(src(r) for r in possible)))
    print("likely leaks by eval benchmark:",
          dict(collections.Counter(b for ms in likely.values() for m in ms for b in m["benchmarks"]).most_common(20)))

    if args.apply:
        final = ROWS / "final"
        manifest = {}
        for name, rows in rows_by_file.items():
            kept = [r for r in rows if not any(im in likely for im in r["images"])]
            write_jsonl(final / f"{name}.jsonl", kept)
            manifest[name] = {"rows": len(kept), "dropped": len(rows) - len(kept),
                              "train": sum(r["split"] == "train" for r in kept),
                              "dev": sum(r["split"] == "dev" for r in kept)}
        (final / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print("final:", json.dumps(manifest))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
