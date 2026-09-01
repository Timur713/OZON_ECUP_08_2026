#!/usr/bin/env python
"""Calibrate offline fallback with a leaderboard-free rolling mean forecast.

The global mean model uses only historical labels whose 30-day window is fully
observed.  Hyperparameters are selected by rolling-origin backtests that leave
a 30-day label-availability gap.  No public score or recovered moment is read.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
SOURCE = ROOT / "submissions" / "119_offline_rules_safe_6model.csv"
OUTPUT = ROOT / "submissions" / "120_offline_rules_safe_meanforecast.csv"
META = WORK / "120_offline_rules_safe_meanforecast_meta.json"
LAMBDAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
HARMONICS = (1, 2, 3, 4)
YEAR = 365.2425


gmv = np.load(WORK / "mat" / "gmv.npy", mmap_mode="r")
cumulative = np.zeros((gmv.shape[0], gmv.shape[1] + 1), dtype=np.float64)
np.cumsum(gmv, axis=1, out=cumulative[:, 1:])
# `mean_target[t]` is mean log1p GMV for days t+1..t+30.  It is observable
# only through t=378 because the provided history ends at day 408.
mean_target = np.full(gmv.shape[1], np.nan, dtype=np.float64)
for anchor in range(29, 379):
    target = np.log1p(cumulative[:, anchor + 31] - cumulative[:, anchor + 1])
    mean_target[anchor] = target.mean()


def features(anchor: int, harmonics: int) -> np.ndarray:
    midpoint = anchor + 15
    values = [
        mean_target[anchor - 30],
        mean_target[anchor - 60],
        mean_target[anchor - 90],
        (anchor - 200) / 200,
    ]
    for harmonic in range(1, harmonics + 1):
        phase = 2 * np.pi * harmonic * midpoint / YEAR
        values.extend([np.sin(phase), np.cos(phase)])
    return np.asarray(values, dtype=np.float64)


def fit_predict(train_anchors, test_anchor, lam, harmonics):
    design = np.asarray([features(anchor, harmonics) for anchor in train_anchors])
    target = mean_target[train_anchors]
    query = features(test_anchor, harmonics)
    location = design.mean(axis=0)
    scale = design.std(axis=0) + 1e-8
    normalized = (design - location) / scale
    query = (query - location) / scale
    normalized = np.column_stack([normalized, np.ones(len(normalized))])
    query = np.r_[query, 1.0]
    penalty = np.eye(normalized.shape[1]) * lam
    penalty[-1, -1] = 0
    weights = np.linalg.solve(
        normalized.T @ normalized / len(normalized) + penalty,
        normalized.T @ target / len(normalized),
    )
    return float(query @ weights)


cv = []
for harmonics in HARMONICS:
    for lam in LAMBDAS:
        errors = []
        # At every origin, train labels end at least 30 days before the target
        # anchor, matching real label availability.
        for test_anchor in range(210, 379, 7):
            train_anchors = np.arange(119, test_anchor - 29)
            errors.append(
                fit_predict(train_anchors, test_anchor, lam, harmonics)
                - mean_target[test_anchor]
            )
        errors = np.asarray(errors)
        cv.append({
            "harmonics": harmonics,
            "lambda": lam,
            "rolling_rmse": float(np.sqrt(np.mean(errors * errors))),
            "rolling_bias": float(errors.mean()),
            "folds": len(errors),
        })

selected = min(cv, key=lambda row: row["rolling_rmse"])
forecast_mean = fit_predict(
    np.arange(119, 379),
    408,
    selected["lambda"],
    selected["harmonics"],
)

table = pl.read_csv(SOURCE)
source_log = np.log1p(np.clip(table["predict"].to_numpy(), 0, None)).astype(np.float64)
shift = forecast_mean - np.clip(source_log, 0, None).mean()
for _ in range(12):
    calibrated = np.clip(source_log + shift, 0, None)
    delta = forecast_mean - calibrated.mean()
    shift += delta
    if abs(delta) < 1e-13:
        break
calibrated = np.clip(source_log + shift, 0, None)
prediction = np.expm1(calibrated)
if not np.isfinite(prediction).all() or np.any(prediction < 0):
    raise ValueError("invalid calibrated offline fallback")
with OUTPUT.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["user_id", "predict"])
    writer.writerows(
        (int(user_id), float(value))
        for user_id, value in zip(table["user_id"], prediction)
    )

sha256 = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
report = {
    "file": str(OUTPUT),
    "sha256": sha256,
    "source": str(SOURCE),
    "construction": "historical_only_rolling_mean_forecast",
    "uses_public_scores": False,
    "uses_recovered_moments": False,
    "cv_grid": cv,
    "selected": selected,
    "forecast_mean_log1p": forecast_mean,
    "source_mean_log1p": float(source_log.mean()),
    "applied_shift_before_clipping": float(shift),
    "output_mean_log1p": float(calibrated.mean()),
    "output_std_log1p": float(calibrated.std()),
    "prediction_min": float(prediction.min()),
    "prediction_max": float(prediction.max()),
    "rows": len(prediction),
}
META.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
