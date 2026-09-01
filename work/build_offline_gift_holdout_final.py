#!/usr/bin/env python
"""Build a leaderboard-free gift-season/user-holdout final.

The shape model is fitted only on 50,000 customers excluded from training of
the validation networks, for the historical 14-Feb--15-Mar target window.
Regularisation is selected by fixed 10k-fit -> 40k-score customer splits.
The future global level is forecast from historical labels with a 30-day
availability gap.  No leaderboard score, recovered moment, or competition
submission is read by this script.
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
OUTPUT = ROOT / "submissions" / "121_offline_gift_holdout.csv"
META = WORK / "121_offline_gift_holdout_meta.json"
LAMBDAS = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
REPEATS = 96
SEED = 20260825
YEAR = 365.2425


def load_components(stem: str, split: str) -> tuple[list[str], list[np.ndarray]]:
    path = WORK / f"{stem}_{split}_server_best_val_components.npz"
    if split == "full":
        path = WORK / f"{stem}_full_server_components.npz"
    names: list[str] = []
    arrays: list[np.ndarray] = []
    with np.load(path) as values:
        for head in ("combined", "hurdle", "direct"):
            names.append(f"{stem}_{head}")
            arrays.append(values[head].astype(np.float64))
    return names, arrays


names: list[str] = []
validation_columns: list[np.ndarray] = []
final_columns: list[np.ndarray] = []
for model in ("cls43hold", "cls43exacthold"):
    val_names, val_arrays = load_components(model, "val")
    full_names, full_arrays = load_components(model, "full")
    if val_names != full_names:
        raise RuntimeError(f"head mismatch for {model}")
    names.extend(val_names)
    validation_columns.extend(val_arrays)
    final_columns.extend(full_arrays)

validation = np.column_stack(validation_columns)
final = np.column_stack(final_columns)
holdout_users = np.load(WORK / "cls43hold_val_server_val_users.npy").astype(int)
exact_users = np.load(WORK / "cls43exacthold_val_server_val_users.npy").astype(int)
if not np.array_equal(holdout_users, exact_users):
    raise RuntimeError("historical user holdouts are not identical")
if validation.shape != (len(holdout_users), len(names)):
    raise RuntimeError("unexpected validation matrix shape")

gmv = np.load(WORK / "mat" / "gmv.npy", mmap_mode="r")
# Anchor 43 predicts days 44..73 (zero-based), i.e. 14-Feb--15-Mar 2025.
truth = np.log1p(gmv[holdout_users, 44:74].sum(axis=1, dtype=np.float64))


def fit(index: np.ndarray, lam: float) -> dict[str, np.ndarray | float]:
    design = validation[index]
    target = truth[index]
    location = design.mean(axis=0)
    scale = design.std(axis=0) + 1e-8
    target_mean = float(target.mean())
    standardized = (design - location) / scale
    centered_target = target - target_mean
    root_n = np.sqrt(len(index))
    coefficients, _ = nnls(
        np.vstack([
            standardized / root_n,
            np.sqrt(lam) * np.eye(len(names)),
        ]),
        np.r_[centered_target / root_n, np.zeros(len(names))],
        maxiter=10_000,
    )
    return {
        "location": location,
        "scale": scale,
        "target_mean": target_mean,
        "coefficients": coefficients,
    }


def predict(matrix: np.ndarray, model: dict[str, np.ndarray | float]) -> np.ndarray:
    return np.clip(
        (matrix - model["location"]) / model["scale"]
        @ model["coefficients"]
        + model["target_mean"],
        0,
        None,
    )


rng = np.random.default_rng(SEED)
all_index = np.arange(len(truth))
splits: list[tuple[np.ndarray, np.ndarray]] = []
for _ in range(REPEATS):
    fit_index = rng.choice(len(truth), 10_000, replace=False)
    score_mask = np.ones(len(truth), dtype=bool)
    score_mask[fit_index] = False
    splits.append((fit_index, all_index[score_mask]))

cv_rows = []
for lam in LAMBDAS:
    scores = []
    active_counts = []
    positive_fractions = np.zeros(len(names), dtype=np.float64)
    for fit_index, score_index in splits:
        model = fit(fit_index, lam)
        prediction = predict(validation[score_index], model)
        scores.append(float(np.sqrt(np.mean(
            (truth[score_index] - prediction) ** 2
        ))))
        active = model["coefficients"] > 1e-8
        active_counts.append(int(active.sum()))
        positive_fractions += active
    cv_rows.append({
        "lambda": lam,
        "mean_independent_private": float(np.mean(scores)),
        "private_score_se": float(np.std(scores, ddof=1) / np.sqrt(REPEATS)),
        "mean_active_components": float(np.mean(active_counts)),
        "component_positive_fraction": dict(zip(
            names, (positive_fractions / REPEATS).tolist()
        )),
    })

selected = min(cv_rows, key=lambda row: row["mean_independent_private"])
shape_model = fit(all_index, selected["lambda"])
validation_prediction = predict(validation, shape_model)
validation_score = float(np.sqrt(np.mean(
    (truth - validation_prediction) ** 2
)))
raw_final_log = predict(final, shape_model)

# Historical-only mean forecast.  The model/hyperparameters are identical to
# the independently frozen forecast in candidate 120, but recomputed here.
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


def forecast_mean(train_anchors: np.ndarray, test_anchor: int) -> float:
    design = np.asarray([mean_features(anchor) for anchor in train_anchors])
    target = mean_target[train_anchors]
    query = mean_features(test_anchor)
    location = design.mean(axis=0)
    scale = design.std(axis=0) + 1e-8
    standardized = (design - location) / scale
    query = (query - location) / scale
    standardized = np.column_stack([standardized, np.ones(len(standardized))])
    query = np.r_[query, 1.0]
    penalty = np.eye(standardized.shape[1])
    penalty[-1, -1] = 0
    weights = np.linalg.solve(
        standardized.T @ standardized / len(standardized) + penalty,
        standardized.T @ target / len(standardized),
    )
    return float(query @ weights)


target_mean = forecast_mean(np.arange(119, 379), 408)
shift = target_mean - raw_final_log.mean()
for _ in range(20):
    final_log = np.clip(raw_final_log + shift, 0, None)
    delta = target_mean - final_log.mean()
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
    "construction": "historical_gift_season_user_holdout_nonnegative_ridge",
    "uses_public_scores": False,
    "uses_recovered_moments": False,
    "uses_competition_target_mean": False,
    "historical_target": "2025-02-14 through 2025-03-15",
    "validation_customers": len(holdout_users),
    "cv_protocol": "96 fixed 10k-fit to independent-40k-score user splits",
    "families": names,
    "lambda_grid": list(LAMBDAS),
    "cv": cv_rows,
    "selected_lambda": selected["lambda"],
    "validation_score_full_50k": validation_score,
    "shape_location": dict(zip(names, shape_model["location"].tolist())),
    "shape_scale": dict(zip(names, shape_model["scale"].tolist())),
    "standardized_coefficients": dict(zip(
        names, shape_model["coefficients"].tolist()
    )),
    "historical_holdout_target_mean": shape_model["target_mean"],
    "raw_final_mean_log1p": float(raw_final_log.mean()),
    "historical_only_forecast_mean_log1p": target_mean,
    "applied_shift_before_clipping": float(shift),
    "output_mean_log1p": float(final_log.mean()),
    "output_std_log1p": float(final_log.std()),
    "prediction_min": float(prediction.min()),
    "prediction_max": float(prediction.max()),
    "rows": len(prediction),
  }
META.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
