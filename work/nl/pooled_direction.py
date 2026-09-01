#!/usr/bin/env python
"""151 — is there ONE common conditional direction, or four fold-specific ones?

Frozen reading key: work/151_pooled_direction_preregister.json.
Reuses the fold state persisted by 150; no leaderboard score is read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
FOLDS = [288, 318, 348, 378]
CROSS_FOLDS = 5
SEED = 20260828

state = {
    fold: dict(
        standard=np.load(OUT / f"state_{fold}_standard.npy").astype(np.float64),
        truth=np.load(OUT / f"state_{fold}_truth.npy").astype(np.float64),
        affine=np.load(OUT / f"state_{fold}_affine.npy").astype(np.float64),
        beta=np.load(OUT / f"state_{fold}_beta.npy").astype(np.float64),
    )
    for fold in FOLDS
}
n = len(state[FOLDS[0]]["truth"])
fold_of = np.random.default_rng(SEED).permutation(n) % CROSS_FOLDS


def rmsle(prediction, truth):
    residual = truth - np.clip(prediction, 0.0, None)
    return float(np.sqrt(np.mean(residual * residual)))


def out_of_fold_linear(design, truth):
    prediction = np.zeros(n)
    for k in range(CROSS_FOLDS):
        score_index = np.flatnonzero(fold_of == k)
        fit_index = np.flatnonzero(fold_of != k)
        x = design[fit_index]
        coef = np.linalg.solve(
            x.T @ x / len(fit_index) + np.eye(design.shape[1]) * 1e-9,
            x.T @ truth[fit_index] / len(fit_index),
        )
        prediction[score_index] = design[score_index] @ coef
    return prediction


def gain(target, direction):
    design = np.column_stack([state[target]["affine"], direction, np.ones(n)])
    corrected = out_of_fold_linear(design, state[target]["truth"])
    baseline = rmsle(state[target]["affine"], state[target]["truth"])
    return baseline - rmsle(corrected, state[target]["truth"])


pooled = {}
for target in FOLDS:
    others = [f for f in FOLDS if f != target]
    direction_beta = np.mean([state[f]["beta"] for f in others], axis=0)
    pooled[str(target)] = gain(target, state[target]["standard"] @ direction_beta)

# Same pooling, but with each source beta normalised first, so that one loud
# fold cannot dominate the average purely through its scale.
pooled_unit = {}
for target in FOLDS:
    others = [f for f in FOLDS if f != target]
    direction_beta = np.mean(
        [state[f]["beta"] / np.linalg.norm(state[f]["beta"]) for f in others], axis=0
    )
    pooled_unit[str(target)] = gain(target, state[target]["standard"] @ direction_beta)

worst = min(pooled.values())
worst_unit = min(pooled_unit.values())
verdict = (
    "promote" if worst >= 0.00008 and worst_unit >= 0.00008 else "closed"
)
report = {
    "tag": "151_pooled_conditional_direction",
    "leave_one_fold_out_pooled_gain": pooled,
    "leave_one_fold_out_pooled_gain_unit_normalised": pooled_unit,
    "worst_fold_gain": worst,
    "worst_fold_gain_unit_normalised": worst_unit,
    "threshold": 0.00008,
    "verdict": verdict,
}
(OUT / "151_pooled_direction.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
