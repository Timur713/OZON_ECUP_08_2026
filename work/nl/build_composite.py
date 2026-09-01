#!/usr/bin/env python
"""227 - compress every extra trained model into ONE composite base.

Fifty-one models trained this session are each below the admission threshold on
their own, but together they move the fold-378 out-of-fold score from 1.6645302
to 1.6639663. Delivering that through the moment apparatus one base at a time
would cost fifty-one probes. Compressing them into a single vector costs one
probe and one degree of freedom.

The composite's weights are fitted on fold-378 labels, which is offline and
leaderboard-free, but it means the composite has SEEN those labels. Screening it
against the pool at the same fold would therefore be optimistic, so the
composite is built out of fold: for each fifth of users the weights come from
the other four fifths. The vector that goes to anchor 408 uses weights fitted on
all users, which is the honest deployment object, while the vector that is
SCREENED is the out-of-fold one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
DIV = Path(__file__).resolve().parents[1] / "div"
LAM = 0.003
SEED = 20260828

truth = np.load(OUT / "truth378.npy").astype(np.float64)
n = len(truth)
names, val_cols, final_cols = [], [], []
for path in sorted(DIV.glob("*_val.npy")):
    final = path.with_name(path.name.replace("_val.npy", "_final.npy"))
    if not final.exists():
        continue
    v = np.load(path).astype(np.float64)
    f = np.load(final).astype(np.float64)
    if v.shape != truth.shape or f.shape != truth.shape:
        continue
    if not (np.isfinite(v).all() and np.isfinite(f).all()) or v.std() < 1e-6:
        continue
    names.append(path.stem.replace("_val", ""))
    val_cols.append(v)
    final_cols.append(f)
print(f"{len(names)} models with both vectors")

val = np.column_stack(val_cols + [np.ones(n)])
final = np.column_stack(final_cols + [np.ones(n)])
penalty = np.eye(val.shape[1]) * LAM
penalty[-1, -1] = 0.0

folds = np.random.default_rng(SEED).permutation(n) % 5
composite_oof = np.zeros(n)
for k in range(5):
    score = np.flatnonzero(folds == k)
    fit = np.flatnonzero(folds != k)
    x = val[fit]
    weights = np.linalg.solve(
        x.T @ x / len(fit) + penalty, x.T @ truth[fit] / len(fit)
    )
    composite_oof[score] = val[score] @ weights
residual = truth - np.clip(composite_oof, 0, None)
print(f"composite out-of-fold RMSLE at 378: {np.sqrt(np.mean(residual ** 2)):.7f}")

full_weights = np.linalg.solve(
    val.T @ val / n + penalty, val.T @ truth / n
)
composite_final = final @ full_weights
np.save(DIV / "composite51_val.npy", composite_oof.astype(np.float32))
np.save(DIV / "composite51_final.npy", composite_final.astype(np.float32))
(OUT / "227_composite_members.json").write_text(json.dumps({
    "members": names, "lambda": LAM,
    "composite_oof_rmsle_fold378": float(np.sqrt(np.mean(residual ** 2))),
    "weights": {k: float(w) for k, w in zip(names + ["const"], full_weights)},
}, indent=2) + "\n")
print(f"final composite mean={composite_final.mean():.4f} sd={composite_final.std():.4f}")
