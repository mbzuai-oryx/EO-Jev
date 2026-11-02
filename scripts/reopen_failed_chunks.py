"""Reopen the chunks of a run dir (bench/runners/vlm.py or d3.py layout) that need work, so that rerunning the runner on
the same dir redoes only what is missing:
- error records ("status": "error") are dropped from their chunk file, and that chunk's .done and .lock removed;
- locks without a .done marker (a task that was killed mid-chunk) are removed; the chunk's finished rows are kept.
Prints "<dir> <rows dropped> <chunks reopened> <chunks done>"; exit code 0 if nothing needed reopening, 3 otherwise.

    python scripts/reopen_failed_chunks.py <run dir> [--dry]
"""

import json
import sys
from pathlib import Path


def main() -> int:
    run = Path(sys.argv[1])
    dry = "--dry" in sys.argv
    dropped = reopened = 0
    for path in sorted(run.glob("chunk*.jsonl")):
        lines = path.read_text().splitlines(keepends=True)
        keep = [ln for ln in lines if json.loads(ln).get("status") != "error"]
        if len(keep) < len(lines):
            dropped += len(lines) - len(keep)
            reopened += 1
            if not dry:
                path.with_suffix(".jsonl.tmp").write_text("".join(keep))
                path.with_suffix(".jsonl.tmp").replace(path)
                for marker in (path.with_suffix(".done"), path.with_suffix(".lock")):
                    marker.unlink(missing_ok=True)
    for lock in sorted(run.glob("chunk*.lock")):
        if not lock.with_suffix(".done").exists():
            reopened += 1
            if not dry:
                lock.unlink()
    n_done = len(list(run.glob("chunk*.done")))
    print(f"{run} {dropped} {reopened} {n_done}")
    return 3 if (reopened or n_done < 64) else 0


if __name__ == "__main__":
    sys.exit(main())
