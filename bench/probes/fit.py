"""Fit linear probes on extracted features and compare with zero-shot readout accuracy.

For each feature set (decision-state hidden, pooled vision), a multinomial logistic regression on standardized
features is fit on the train split (C chosen on a 10% held-out slice of train) and scored on the test split.
Also reported: a nearest-class-mean (prototype) readout, and the model's own zero-shot accuracy on the test split.

    python -m bench.probes.fit <features dir> [<features dir> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load(d: Path) -> dict[str, np.ndarray]:
    parts = [np.load(f) for f in sorted(d.glob("chunk*.npz"))]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


def probe(xtr, ytr, xte, yte) -> dict:
    scaler = StandardScaler().fit(xtr)
    xtr, xte = scaler.transform(xtr), scaler.transform(xte)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(xtr))
    cut = len(idx) // 10
    val, fit_idx = idx[:cut], idx[cut:]
    best = max((LogisticRegression(C=c, max_iter=2000).fit(xtr[fit_idx], ytr[fit_idx]).score(xtr[val], ytr[val]), c)
               for c in (0.001, 0.01, 0.1, 1.0))
    clf = LogisticRegression(C=best[1], max_iter=3000).fit(xtr, ytr)
    classes = np.unique(ytr)
    means = np.stack([xtr[ytr == k].mean(0) for k in classes])
    proto = classes[np.argmax(xte @ means.T - 0.5 * (means ** 2).sum(1), axis=1)]
    return {"linear_probe_acc": float(clf.score(xte, yte)), "C": best[1], "prototype_acc": float((proto == yte).mean())}


def main() -> None:
    report = {}
    for d in map(Path, sys.argv[1:]):
        f = load(d)
        tr, te = f["split"] == "train", f["split"] == "test"
        entry = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
                 "zero_shot_acc_test": float((f["pred"][te] == f["labels"][te]).mean())}
        for feat in ("hidden", "vision"):
            x = f[feat].astype(np.float32)
            entry[feat] = probe(x[tr], f["labels"][tr], x[te], f["labels"][te])
        report[str(d)] = entry
        print(d.name, json.dumps(entry), flush=True)
    out = Path(sys.argv[1]).parent / "probe_report.json"
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
