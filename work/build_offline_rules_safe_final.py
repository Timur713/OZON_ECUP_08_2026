#!/usr/bin/env python
"""Build a fixed six-family fallback without leaderboard-derived quantities.

No public score, recovered target moment, competition mean, or frozen final is
read while constructing this file.  Lambda is selected on historical labels by
fixed 50k-fit -> 200k-score splits, then the nonnegative stack is refit on the
full historical validation anchor.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUTPUT = ROOT / "submissions" / "119_offline_rules_safe_6model.csv"
META = WORK / "119_offline_rules_safe_6model_meta.json"
LAMBDAS = (0.001, 0.003, 0.01, 0.03)
REPEATS = 96
SEED = 20260825


def load(name: str) -> np.ndarray:
    return np.load(WORK / name).astype(np.float64)


families = {
    "GBD262": (load("v4_262_valpred.npy"), load("AVG_GBD.npy")),
    "W120_seed_average": (
        np.mean([load(f"w120{seed}_val.npy") for seed in "abc"], axis=0),
        np.mean([load(f"w120{seed}_final.npy") for seed in "abc"], axis=0),
    ),
    "TCN180_two_head": (load("tcn180two_val.npy"), load("tcn180two_final.npy")),
    "TCN365_growing_anchor": (
        load("tcn365v336_val.npy"), load("tcn365v336_final.npy"),
    ),
    "TCN409": (load("tcn409_val.npy"), load("tcn409_final.npy")),
    "TCN409_replication": (load("w409c_val.npy"), load("w409c_final.npy")),
}
names = list(families)
validation = np.column_stack([
    *[families[name][0] for name in names],
    np.ones(len(next(iter(families.values()))[0])),
])
final = np.column_stack([
    *[families[name][1] for name in names],
    np.ones(len(next(iter(families.values()))[1])),
])
gmv = np.load(WORK / "mat" / "gmv.npy", mmap_mode="r")
truth = np.log1p(gmv[:, 379:409].sum(axis=1, dtype=np.float64))
all_index = np.arange(len(truth))


def fit(index: np.ndarray, lam: float):
    design = validation[index, :-1]
    target = truth[index]
    design_mean = design.mean(axis=0)
    target_mean = target.mean()
    centered_design = design - design_mean
    centered_target = target - target_mean
    scale = np.sqrt(len(index))
    augmented_design = np.vstack([
        centered_design / scale,
        np.sqrt(lam) * np.eye(len(names)),
    ])
    augmented_target = np.r_[centered_target / scale, np.zeros(len(names))]
    model_weights, _ = nnls(
        augmented_design, augmented_target, maxiter=10_000
    )
    intercept = target_mean - design_mean @ model_weights
    weights = np.r_[model_weights, intercept]
    if not np.isfinite(weights).all():
        raise RuntimeError("nonnegative ridge returned non-finite weights")
    return weights


rng = np.random.default_rng(SEED)
splits = []
for _ in range(REPEATS):
    public = rng.choice(len(truth), 50_000, replace=False)
    private_mask = np.ones(len(truth), dtype=bool)
    private_mask[public] = False
    splits.append((public, all_index[private_mask]))

cv_rows = []
for lam in LAMBDAS:
    scores = []
    active_counts = []
    for fit_index, score_index in splits:
        weights = fit(fit_index, lam)
        prediction = np.clip(validation[score_index] @ weights, 0, None)
        scores.append(float(np.sqrt(np.mean(
            (truth[score_index] - prediction) ** 2
        ))))
        active_counts.append(int(np.sum(weights[:-1] > 1e-8)))
    cv_rows.append({
        "lambda": lam,
        "mean_independent_private": float(np.mean(scores)),
        "private_score_se": float(np.std(scores, ddof=1) / np.sqrt(REPEATS)),
        "mean_active_components": float(np.mean(active_counts)),
    })

selected = min(cv_rows, key=lambda row: row["mean_independent_private"])
weights = fit(all_index, selected["lambda"])
validation_prediction = np.clip(validation @ weights, 0, None)
validation_score = float(np.sqrt(np.mean(
    (truth - validation_prediction) ** 2
)))
final_log = np.clip(final @ weights, 0, None)
prediction = np.expm1(final_log)
if not np.isfinite(prediction).all() or np.any(prediction < 0):
    raise ValueError("invalid offline fallback prediction")
uids = np.load(WORK / "mat" / "uids.npy")
with OUTPUT.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["user_id", "predict"])
    writer.writerows((int(user_id), float(value)) for user_id, value in zip(uids, prediction))

sha256 = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
report = {
    "file": str(OUTPUT),
    "sha256": sha256,
    "construction": "historical_validation_only_nonnegative_ridge",
    "uses_public_scores": False,
    "uses_recovered_moments": False,
    "uses_competition_target_mean": False,
    "families": names,
    "lambda_grid": list(LAMBDAS),
    "lambda_selection": "minimum mean independent 200k score over fixed 96 splits",
    "cv": cv_rows,
    "selected_lambda": selected["lambda"],
    "validation_score_full_250k": validation_score,
    "active_components": int(np.sum(weights[:-1] > 1e-8)),
    "weights": dict(zip(names + ["const"], weights.tolist())),
    "final_mean_log1p": float(final_log.mean()),
    "final_std_log1p": float(final_log.std()),
    "prediction_min": float(prediction.min()),
    "prediction_max": float(prediction.max()),
    "rows": len(prediction),
    "warning": (
        "Historical validation is January while the competition target is a "
        "gift season; this file is a rules-safe fallback, not the expected "
        "best competition final."
    ),
}
META.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
