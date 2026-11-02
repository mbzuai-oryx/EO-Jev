"""Write the rows whose results are errors (status != ok, or any question errored) to a new rows file, for a
retry pass into <run dir>/retry (the scorer keeps a row's latest successful result).

    python scripts/retry_errors.py <rows.jsonl> <run dir> <out rows.jsonl>
"""

import glob
import json
import sys

rows_path, run_dir, out = sys.argv[1:4]
ok = set()
bad = set()
for f in glob.glob(f"{run_dir}/chunk*.jsonl") + glob.glob(f"{run_dir}/retry/chunk*.jsonl"):
    for line in open(f):
        r = json.loads(line)
        if r["status"] == "ok" and not any("error" in a for a in r["answers"].values()):
            ok.add(r["id"])
        else:
            bad.add(r["id"])
todo = bad - ok
with open(out, "w") as fh:
    for line in open(rows_path):
        if json.loads(line)["id"] in todo:
            fh.write(line)
print(f"{len(todo)} rows to retry -> {out}")
