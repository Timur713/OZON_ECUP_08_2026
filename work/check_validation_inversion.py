#!/usr/bin/env python
"""Is fold-378 validation anti-correlated with true real-target quality?

Real-target standalone RMSLE comes from exactly measured public moments and is
therefore ground truth about the competition window.  Fold-378 validation is
the criterion every offline gate in this project has used.  If they disagree in
rank, the whole admission protocol has been selecting against the real target.
"""
import json, os
from pathlib import Path
import numpy as np

ROOT = Path("/home/ubuntu/ecup")
WORK = ROOT / "work"
MAT = WORK / "mat"

real = json.loads((WORK / "real_target_standalone.json").read_text())

gmv = np.load(MAT / "gmv.npy", mmap_mode="r")
nusers, ndays = gmv.shape
cumulative = np.zeros((nusers, ndays + 1), dtype=np.float64)
np.cumsum(gmv, axis=1, out=cumulative[:, 1:])
anchor = 378
target = np.log1p(cumulative[:, anchor + 31] - cumulative[:, anchor + 1]).astype(np.float64)
del cumulative

rng = np.random.default_rng(20260825)
calib = np.sort(rng.choice(nusers, nusers // 5, replace=False))
mask = np.ones(nusers, bool); mask[calib] = False
score_idx = np.flatnonzero(mask)

def fold378(pred):
    design = np.column_stack([pred, np.ones(len(pred))])
    coef = np.linalg.lstsq(design[calib], target[calib], rcond=None)[0]
    fitted = np.clip(design[score_idx] @ coef, 0, None)
    return float(np.sqrt(np.mean((target[score_idx] - fitted) ** 2)))

rows = []
for name in sorted(real):
    for cand in (f"{name}_val.npy", f"AVG_{name}_val.npy", f"{name}_valpred.npy"):
        path = WORK / cand
        if path.exists():
            pred = np.load(path).astype(np.float64)
            if len(pred) != nusers:
                continue
            rows.append((name, fold378(pred), real[name]))
            break

print(f"paired bases: {len(rows)}")
print(f"{'base':16s} {'fold378':>10s} {'REAL target':>12s}")
for name, f, r in sorted(rows, key=lambda x: x[1]):
    print(f"{name:16s} {f:10.6f} {r:12.6f}")

if len(rows) >= 4:
    f = np.array([r[1] for r in rows]); t = np.array([r[2] for r in rows])
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])
    print()
    print(f"Pearson  fold378 vs real = {np.corrcoef(f, t)[0,1]:+.4f}")
    print(f"Spearman fold378 vs real = {spearman(f, t):+.4f}")
    print()
    print("Positive means fold378 ranks models the same way the real target does.")
    print("Negative means every gate built on fold378 selected against the target.")
    json.dump({"rows": [{"base": n, "fold378": f_, "real": r_} for n, f_, r_ in rows],
               "pearson": float(np.corrcoef(f, t)[0,1]),
               "spearman": spearman(f, t)},
              open(WORK / "validation_inversion_audit.json", "w"), indent=2)
