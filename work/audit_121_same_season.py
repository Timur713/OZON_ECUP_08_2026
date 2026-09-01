#!/usr/bin/env python
"""Paired same-season stability audit for frozen candidate 121.

The candidate and its public reading key already exist.  This audit does not
read leaderboard scores or change the candidate.  On each fixed 10k-fit / 40k-
score split it compares the frozen nonnegative six-head ridge construction to
an intentionally strong oracle baseline: the best independently calibrated
single head on that score fold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUTPUT = WORK / "121_same_season_stability_audit.json"
LAM = 0.003
REPEATS = 96
SEED = 20260825


def load_components(stem: str) -> tuple[list[str], list[np.ndarray]]:
    path = WORK / f"{stem}_val_server_best_val_components.npz"
    names: list[str] = []
    arrays: list[np.ndarray] = []
    with np.load(path) as values:
        for head in ("combined", "hurdle", "direct"):
            names.append(f"{stem}_{head}")
            arrays.append(values[head].astype(np.float64))
    return names, arrays


names: list[str] = []
columns: list[np.ndarray] = []
for stem in ("cls43hold", "cls43exacthold"):
    local_names, local_columns = load_components(stem)
    names.extend(local_names)
    columns.extend(local_columns)
design = np.column_stack(columns)

holdout_users = np.load(WORK / "cls43hold_val_server_val_users.npy").astype(int)
exact_users = np.load(WORK / "cls43exacthold_val_server_val_users.npy").astype(int)
if not np.array_equal(holdout_users, exact_users):
    raise RuntimeError("historical user holdouts differ")
gmv = np.load(WORK / "mat" / "gmv.npy", mmap_mode="r")
truth = np.log1p(gmv[holdout_users, 44:74].sum(axis=1, dtype=np.float64))
if design.shape != (len(truth), len(names)) or not np.isfinite(design).all():
    raise RuntimeError("invalid validation design")


def fit(index: np.ndarray, matrix: np.ndarray, lam: float) -> dict[str, np.ndarray | float]:
    local = matrix[index]
    target = truth[index]
    location = local.mean(axis=0)
    scale = local.std(axis=0) + 1e-8
    target_mean = float(target.mean())
    standardized = (local - location) / scale
    root_n = np.sqrt(len(index))
    coefficients, _ = nnls(
        np.vstack([standardized / root_n, np.sqrt(lam) * np.eye(local.shape[1])]),
        np.r_[target - target_mean, np.zeros(local.shape[1])] / root_n,
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


def rmsle_log(index: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((truth[index] - prediction) ** 2)))


rng = np.random.default_rng(SEED)
all_index = np.arange(len(truth))
rows = []
for repeat in range(REPEATS):
    fit_index = rng.choice(len(truth), 10_000, replace=False)
    score_mask = np.ones(len(truth), dtype=bool)
    score_mask[fit_index] = False
    score_index = all_index[score_mask]

    ensemble_model = fit(fit_index, design, LAM)
    ensemble_prediction = predict(design[score_index], ensemble_model)
    ensemble_score = rmsle_log(score_index, ensemble_prediction)

    single_scores = []
    for column in range(design.shape[1]):
        single_model = fit(fit_index, design[:, column:column + 1], 0.0)
        single_prediction = predict(
            design[score_index, column:column + 1], single_model
        )
        single_scores.append(rmsle_log(score_index, single_prediction))
    oracle_score = min(single_scores)
    rows.append({
        "repeat": repeat,
        "ensemble_score": ensemble_score,
        "oracle_single_score": oracle_score,
        "gain_over_oracle_single": oracle_score - ensemble_score,
        "oracle_single": names[int(np.argmin(single_scores))],
    })

gains = np.asarray([row["gain_over_oracle_single"] for row in rows])
report = {
    "candidate": "121_offline_gift_holdout.csv",
    "uses_public_scores": False,
    "target": "2025-02-14 through 2025-03-15",
    "protocol": "96 fixed 10k-fit to independent-40k-score user splits",
    "baseline": "best independently calibrated single head chosen by score-fold oracle",
    "lambda": LAM,
    "heads": names,
    "mean_gain_over_oracle_single": float(gains.mean()),
    "gain_standard_error": float(gains.std(ddof=1) / np.sqrt(REPEATS)),
    "positive_gain_splits": int(np.count_nonzero(gains > 0)),
    "positive_gain_fraction": float(np.mean(gains > 0)),
    "gain_p05": float(np.quantile(gains, 0.05)),
    "gain_p50": float(np.quantile(gains, 0.50)),
    "gain_p95": float(np.quantile(gains, 0.95)),
    "rows": rows,
}
OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
