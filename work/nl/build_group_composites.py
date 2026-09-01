#!/usr/bin/env python
"""236 - one composite per training family.

Frozen key: work/236_group_composites_preregister.json. The partition is by
family and is fixed in that file; the regularisation of each group composite is
chosen from a fixed grid by its own out-of-fold score.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
DIV = Path(__file__).resolve().parents[1] / "div"
GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
SEED = 20260828

GROUPS = {
    "shape": lambda s: "shape" in s,
    "anchors": lambda s: re.search(r"st\d\d", s) is not None or s.startswith("divB") or s.startswith("divH"),
    "repr": lambda s: any(k in s for k in ("daynorm", "cumulative", "diff", "occurrence", "rankday")),
    "dailycal": lambda s: "daily" in s or "cal" in s or "gift" in s,
    "capacity": lambda s: s.startswith("divR") or "w192" in s or "w256" in s or "w384" in s,
    "hurdle": lambda s: s.startswith("divV") or "prob" in s or "magn" in s,
}


def group_of(stem):
    for name, test in GROUPS.items():
        if test(stem):
            return name
    return "other"


truth = np.load(OUT / "truth378.npy").astype(np.float64)
n = len(truth)
folds = np.random.default_rng(SEED).permutation(n) % 5
members = {}
for path in sorted(DIV.glob("*_val.npy")):
    stem = path.stem.replace("_val", "")
    if stem.startswith("composite"):
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
    members.setdefault(group_of(stem), []).append((stem, v, f))

report = {}
for name, rows in sorted(members.items()):
    if len(rows) < 3:
        print(f"{name}: only {len(rows)} members, skipped")
        continue
    val = np.column_stack([r[1] for r in rows] + [np.ones(n)])
    final = np.column_stack([r[2] for r in rows] + [np.ones(n)])
    best = None
    for lam in GRID:
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
        rmsle = float(np.sqrt(np.mean(residual ** 2)))
        if best is None or rmsle < best[1]:
            best = (lam, rmsle, prediction)
    lam, rmsle, oof = best
    penalty = np.eye(val.shape[1]) * lam
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(val.T @ val / n + penalty, val.T @ truth / n)
    np.save(DIV / f"grp_{name}_val.npy", oof.astype(np.float32))
    np.save(DIV / f"grp_{name}_final.npy", (final @ weights).astype(np.float32))
    report[name] = {"members": [r[0] for r in rows], "count": len(rows),
                    "chosen_lambda": lam, "out_of_fold_rmsle": rmsle}
    print(f"{name:10s} {len(rows):3d} members  lam={lam:<6g} oof={rmsle:.7f}")

(OUT / "236_group_composites.json").write_text(json.dumps(report, indent=2) + "\n")
