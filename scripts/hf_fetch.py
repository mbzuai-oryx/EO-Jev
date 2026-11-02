"""Paced, resumable Hugging Face dataset downloader (stays under the 1,000-requests / 5-min per-user quota).

snapshot_download on this node either bursts into 429s on many-file repos or, worse, returns a partial local
folder as if complete. This lists each repo once, then fetches only files that are missing or the wrong size,
at most --rate files per second, sleeping on 429. It verifies every file's size at the end.

    python scripts/hf_fetch.py <repo> <local name> [--allow PATTERN ...]   # files land in $EOJEV_DATASETS/raw/<name>
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("name")
    p.add_argument("--allow", nargs="*")
    p.add_argument("--rate", type=float, default=3.0, help="max files per second")
    args = p.parse_args()

    dest = Path(os.environ["EOJEV_DATASETS"]) / "raw" / args.name
    api = HfApi()
    while True:
        try:
            info = api.dataset_info(args.repo, files_metadata=True)
            break
        except HfHubHTTPError as e:
            if getattr(e.response, "status_code", None) != 429:
                raise
            print("429 listing repo; sleeping 120 s", flush=True)
            time.sleep(120)
    files = [(s.rfilename, s.size or 0) for s in info.siblings
             if not args.allow or any(fnmatch.fnmatch(s.rfilename, pat) for pat in args.allow)]
    todo = [(f, size) for f, size in files if not (dest / f).is_file() or (dest / f).stat().st_size != size]
    total_gb = sum(size for _, size in files) / 1e9
    print(f"{args.repo}: {len(files)} files ({total_gb:.2f} GB), {len(todo)} to fetch", flush=True)

    for i, (f, size) in enumerate(todo, 1):
        failures = 0
        while True:
            started = time.monotonic()
            try:
                hf_hub_download(args.repo, f, repo_type="dataset", local_dir=str(dest), revision=info.sha)
                break
            except HfHubHTTPError as e:
                if getattr(e.response, "status_code", None) != 429:
                    raise
                print(f"429 at file {i}/{len(todo)}; sleeping 120 s", flush=True)
                time.sleep(120)
            except (OSError, ConnectionError) as e:  # dropped connections, truncated bodies (httpx errors), timeouts
                failures += 1
                if failures > 8:
                    raise
                print(f"transient error at file {i}/{len(todo)} ({type(e).__name__}); retry {failures} in 30 s", flush=True)
                time.sleep(30)
            except Exception as e:  # noqa: BLE001 - httpx.RemoteProtocolError etc. are not OSError subclasses
                if "httpx" not in type(e).__module__ and "httpcore" not in type(e).__module__:
                    raise
                failures += 1
                if failures > 8:
                    raise
                print(f"transient error at file {i}/{len(todo)} ({type(e).__name__}); retry {failures} in 30 s", flush=True)
                time.sleep(30)
        time.sleep(max(0.0, 1.0 / args.rate - (time.monotonic() - started)))
        if i % 500 == 0:
            print(f"  {i}/{len(todo)} files", flush=True)

    bad = [f for f, size in files if not (dest / f).is_file() or (dest / f).stat().st_size != size]
    if bad:
        print(f"INCOMPLETE {args.repo}: {len(bad)} files missing or wrong size, e.g. {bad[:3]}", flush=True)
        sys.exit(1)
    print(f"OK {args.repo} -> {dest} ({len(files)} files verified)", flush=True)


if __name__ == "__main__":
    main()
