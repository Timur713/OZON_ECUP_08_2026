#!/usr/bin/env python
"""Build a historical-only stack with a fixed per-model concentration cap.

This candidate is frozen before any 120/122 leaderboard response.  It retains
all six offline families but constrains every model coefficient to [0, 0.35],
preventing the independent w409c replication from dominating the ensemble.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUTPUT = ROOT / "submissions" / "123_offline_capped_w035.csv"
META = WORK / "123_offline_capped_w035_meta.json"
LAMBDA = 0.001
MAX_WEIGHT = 0.35
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
validation = np.column_stack([families[name][0] for name in names])
final = np.column_stack([families[name][1] for name in names])
gmv = np.load(WORK / "mat" / "gmv.npy", mmap_mode="r")
truth = np.log1p(gmv[:, 379:409].sum(axis=1, dtype=np.float64))
all_index = np.arange(len(truth))


def fit(index: np.ndarray) -> np.ndarray:
    design = validation[index]
    target = truth[index]
    location = design.mean(axis=0)
    target_mean = float(target.mean())
    centered = design - location
    centered_target = target - target_mean

    root_n = np.sqrt(len(index))
    result = lsq_linear(
        np.vstack([
            centered / root_n,
            np.sqrt(LAMBDA) * np.eye(len(names)),
        ]),
        np.r_[centered_target / root_n, np.zeros(len(names))],
        bounds=(0.0, MAX_WEIGHT),
        method="bvls",
        tol=1e-12,
        max_iter=10_000,
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"bounded ridge failed: {result.message}")
    intercept = target_mean - location @ result.x
    return np.r_[result.x, intercept]


rng = np.random.default_rng(SEED)
splits = []
for _ in range(REPEATS):
    fit_index = rng.choice(len(truth), 50_000, replace=False)
    score_mask = np.ones(len(truth), dtype=bool)
    score_mask[fit_index] = False
    splits.append((fit_index, all_index[score_mask]))

private_scores = []
public_scores = []
cap_fractions = np.zeros(len(names), dtype=np.float64)
for fit_index, score_index in splits:
    weights = fit(fit_index)
    public_prediction = np.clip(
        validation[fit_index] @ weights[:-1] + weights[-1], 0, None
    )
    private_prediction = np.clip(
        validation[score_index] @ weights[:-1] + weights[-1], 0, None
    )
    public_scores.append(float(np.sqrt(np.mean(
        (truth[fit_index] - public_prediction) ** 2
    ))))
    private_scores.append(float(np.sqrt(np.mean(
        (truth[score_index] - private_prediction) ** 2
    ))))
    cap_fractions += weights[:-1] >= MAX_WEIGHT - 1e-8

weights = fit(all_index)
validation_prediction = np.clip(
    validation @ weights[:-1] + weights[-1], 0, None
)
validation_score = float(np.sqrt(np.mean(
    (truth - validation_prediction) ** 2
)))
raw_final_log = np.clip(final @ weights[:-1] + weights[-1], 0, None)

# Candidate 120's metadata contains only the frozen historical-only forecast;
# the construction never reads 120 predictions, scores, or competition data.
mean_meta = json.loads(
    (WORK / "120_offline_rules_safe_meanforecast_meta.json").read_text()
)
if mean_meta.get("uses_public_scores") or mean_meta.get("uses_recovered_moments"):
    raise RuntimeError("mean forecast metadata is not leaderboard-free")
forecast_mean = float(mean_meta["forecast_mean_log1p"])
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
    raise RuntimeError("invalid prediction")

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
    "construction": "historical_only_bounded_nonnegative_ridge",
    "frozen_before_120_122_scores": True,
    "uses_public_scores": False,
    "uses_recovered_moments": False,
    "uses_competition_target_mean": False,
    "lambda": LAMBDA,
    "max_model_weight": MAX_WEIGHT,
    "families": names,
    "cv_protocol": "96 fixed 50k-fit to independent-200k-score user splits",
    "mean_public_score": float(np.mean(public_scores)),
    "mean_independent_private": float(np.mean(private_scores)),
    "private_score_se": float(
        np.std(private_scores, ddof=1) / np.sqrt(REPEATS)
    ),
    "cap_fraction_by_model": dict(zip(
        names, (cap_fractions / REPEATS).tolist()
    )),
    "validation_score_full_250k": validation_score,
    "weights": dict(zip(names + ["const"], weights.tolist())),
    "historical_only_forecast_mean_log1p": forecast_mean,
    "raw_final_mean_log1p": float(raw_final_log.mean()),
    "applied_shift_before_clipping": float(shift),
    "output_mean_log1p": float(final_log.mean()),
    "output_std_log1p": float(final_log.std()),
    "prediction_min": float(prediction.min()),
    "prediction_max": float(prediction.max()),
    "rows": len(prediction),
}
META.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
