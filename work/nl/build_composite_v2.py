#!/usr/bin/env python
"""234 - a second composite, from models the first one never saw.

Frozen key: work/234_composite_v2_preregister.json. The regularisation is
chosen by the composite's own out-of-fold score over a grid fixed before any
member existed, because the first composite's weights, at the pool's lambda,
transferred at only one sixth and fold-378 idiosyncrasy is the likely reason.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
DIV = Path(__file__).resolve().parents[1] / "div"
GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
SEED = 20260828

truth = np.load(OUT / "truth378.npy").astype(np.float64)
n = len(truth)
used = set(json.loads((OUT / "227_composite_members.json").read_text())["members"])
names, val_cols, final_cols = [], [], []
for path in sorted(DIV.glob("*_val.npy")):
    stem = path.stem.replace("_val", "")
    if stem in used or stem.startswith("composite"):
        continue
    final = path.with_name(path.name.replace("_val.npy", "_final.npy"))
    if not final.exists():
        continue
    v = np.load(path).astype(np.float64)
    f = np.load(final).astype(np.float64)
    if v.shape != truth.shape or f.shape != truth.shape:
        continue
    if not (np.isfinite(v).all() and np.isfinite(f).all()) or v.std() < 1e-6:
        continue
    names.append(stem)
    val_cols.append(v)
    final_cols.append(f)
print(f"{len(names)} members not used by composite51")
if len(names) < 5:
    raise SystemExit("too few new members; wait for the bank queue")

val = np.column_stack(val_cols + [np.ones(n)])
final = np.column_stack(final_cols + [np.ones(n)])
folds = np.random.default_rng(SEED).permutation(n) % 5


def out_of_fold(lam):
    penalty = np.eye(val.shape[1]) * lam
    penalty[-1, -1] = 0.0
    prediction = np.zeros(n)
    for k in range(5):
        score = np.flatnonzero(folds == k)
        fit = np.flatnonzero(folds != k)
        x = val[fit]
        prediction[score] = val[score] @ np.linalg.solve(
            x.T @ x / len(fit) + penalty, x.T @ truth[fit] / len(fit)
        )
    residual = truth - np.clip(prediction, 0, None)
    return float(np.sqrt(np.mean(residual ** 2))), prediction


scores = {}
best = None
for lam in GRID:
    score, prediction = out_of_fold(lam)
    scores[str(lam)] = score
    print(f"  lam={lam:<7g} composite out-of-fold RMSLE = {score:.7f}")
    if best is None or score < best[1]:
        best = (lam, score, prediction)
lam, score, oof = best
print(f"chosen lambda {lam} at {score:.7f}")

penalty = np.eye(val.shape[1]) * lam
penalty[-1, -1] = 0.0
weights = np.linalg.solve(val.T @ val / n + penalty, val.T @ truth / n)
np.save(DIV / "compositeV2_val.npy", oof.astype(np.float32))
np.save(DIV / "compositeV2_final.npy", (final @ weights).astype(np.float32))
(OUT / "234_composite_v2_members.json").write_text(json.dumps({
    "members": names, "grid": list(GRID), "out_of_fold_by_lambda": scores,
    "chosen_lambda": lam, "chosen_out_of_fold": score,
}, indent=2) + "\n")
print("written compositeV2_val.npy and compositeV2_final.npy")
