#!/usr/bin/env python
"""Build a leaderboard-free challenger without the dominant w409c replica.

The structural constraint is fixed in advance: combine the tree model, the
short-window seed average, and three independently trained TCN mechanisms, but
exclude TCN409_replication.  Lambda is chosen on fixed historical 50k-fit to
independent-200k-score customer splits.  The future mean is forecast from
historical labels only.  No leaderboard-derived value or submission is read.
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
OUTPUT = ROOT / "submissions" / "122_offline_diverse_no_replica.csv"
META = WORK / "122_offline_diverse_no_replica_meta.json"
LAMBDAS = (0.001, 0.003, 0.01, 0.03)
REPEATS = 96
SEED = 20260825
YEAR = 365.2425


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
}
names = list(families)
validation = np.column_stack([families[name][0] for name in names])
final = np.column_stack([families[name][1] for name in names])
gmv = np.load(WORK / "mat" / "gmv.npy", mmap_mode="r")
truth = np.log1p(gmv[:, 379:409].sum(axis=1, dtype=np.float64))
all_index = np.arange(len(truth))


def fit(index: np.ndarray, lam: float) -> np.ndarray:
    design = validation[index]
    target = truth[index]
    design_mean = design.mean(axis=0)
    target_mean = float(target.mean())
    centered_design = design - design_mean
    centered_target = target - target_mean
    root_n = np.sqrt(len(index))
    coefficients, _ = nnls(
        np.vstack([
            centered_design / root_n,
            np.sqrt(lam) * np.eye(len(names)),
        ]),
        np.r_[centered_target / root_n, np.zeros(len(names))],
        maxiter=10_000,
    )
    intercept = target_mean - design_mean @ coefficients
    return np.r_[coefficients, intercept]


rng = np.random.default_rng(SEED)
splits = []
for _ in range(REPEATS):
    fit_index = rng.choice(len(truth), 50_000, replace=False)
    score_mask = np.ones(len(truth), dtype=bool)
    score_mask[fit_index] = False
    splits.append((fit_index, all_index[score_mask]))

cv_rows = []
for lam in LAMBDAS:
    scores = []
    active_counts = []
    for fit_index, score_index in splits:
        weights = fit(fit_index, lam)
        prediction = np.clip(
            validation[score_index] @ weights[:-1] + weights[-1], 0, None
        )
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
validation_prediction = np.clip(
    validation @ weights[:-1] + weights[-1], 0, None
)
validation_score = float(np.sqrt(np.mean(
    (truth - validation_prediction) ** 2
)))
raw_final_log = np.clip(final @ weights[:-1] + weights[-1], 0, None)

# Recompute the frozen historical-only mean forecast (one annual harmonic,
# lambda=1.0), without reading candidate 120 or its metadata.
cumulative = np.zeros((gmv.shape[0], gmv.shape[1] + 1), dtype=np.float64)
np.cumsum(gmv, axis=1, out=cumulative[:, 1:])
mean_target = np.full(gmv.shape[1], np.nan, dtype=np.float64)
for anchor in range(29, 379):
    target = np.log1p(cumulative[:, anchor + 31] - cumulative[:, anchor + 1])
    mean_target[anchor] = target.mean()


def mean_features(anchor: int) -> np.ndarray:
    midpoint = anchor + 15
    phase = 2 * np.pi * midpoint / YEAR
    return np.asarray([
        mean_target[anchor - 30],
        mean_target[anchor - 60],
        mean_target[anchor - 90],
        (anchor - 200) / 200,
        np.sin(phase),
        np.cos(phase),
    ])


train_anchors = np.arange(119, 379)
mean_design = np.asarray([mean_features(anchor) for anchor in train_anchors])
mean_truth = mean_target[train_anchors]
mean_query = mean_features(408)
mean_location = mean_design.mean(axis=0)
mean_scale = mean_design.std(axis=0) + 1e-8
mean_design = (mean_design - mean_location) / mean_scale
mean_query = (mean_query - mean_location) / mean_scale
mean_design = np.column_stack([mean_design, np.ones(len(mean_design))])
mean_query = np.r_[mean_query, 1.0]
mean_penalty = np.eye(mean_design.shape[1])
mean_penalty[-1, -1] = 0
mean_weights = np.linalg.solve(
    mean_design.T @ mean_design / len(mean_design) + mean_penalty,
    mean_design.T @ mean_truth / len(mean_design),
)
forecast_mean = float(mean_query @ mean_weights)

shift = forecast_mean - raw_final_log.mean()
for _ in range(20):
    final_log = np.clip(raw_final_log + shift, 0, None)
    delta = forecast_mean - final_log.mean()
    shift += delta
    if abs(delta) < 1e-13:
        break
final_log = np.clip(raw_final_log + shift, 0, None)
prediction = np.expm1(final_log)
if not np.isfinite(prediction).all() or np.any(prediction < 0):
    raise RuntimeError("invalid final prediction")

uids = np.load(WORK / "mat" / "uids.npy")
with OUTPUT.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["user_id", "predict"])
    writer.writerows(
        (int(user_id), float(value))
        for user_id, value in zip(uids, prediction)
    )

sha256 = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
report = {
    "file": str(OUTPUT),
    "sha256": sha256,
    "construction": "historical_validation_only_no_w409c_replication",
    "uses_public_scores": False,
    "uses_recovered_moments": False,
    "uses_competition_target_mean": False,
    "structural_constraint": "exclude dominant TCN409_replication",
    "families": names,
    "lambda_grid": list(LAMBDAS),
    "cv_protocol": "96 fixed 50k-fit to independent-200k-score user splits",
    "cv": cv_rows,
    "selected_lambda": selected["lambda"],
    "validation_score_full_250k": validation_score,
    "active_components": int(np.sum(weights[:-1] > 1e-8)),
    "weights": dict(zip(names + ["const"], weights.tolist())),
    "raw_final_mean_log1p": float(raw_final_log.mean()),
    "historical_only_forecast_mean_log1p": forecast_mean,
    "applied_shift_before_clipping": float(shift),
    "output_mean_log1p": float(final_log.mean()),
    "output_std_log1p": float(final_log.std()),
    "prediction_min": float(prediction.min()),
    "prediction_max": float(prediction.max()),
    "rows": len(prediction),
}
META.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
