#!/usr/bin/env python
"""Paired 96-split risk audit for frozen offline candidates 120 and 122."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear, nnls


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUTPUT = WORK / "120_122_paired_risk96.json"
REPEATS = 96
SEED = 20260825


def load(name: str) -> np.ndarray:
    return np.load(WORK / name).astype(np.float64)


columns = {
    "GBD262": load("v4_262_valpred.npy"),
    "W120_seed_average": np.mean([
        load(f"w120{seed}_val.npy") for seed in "abc"
    ], axis=0),
    "TCN180_two_head": load("tcn180two_val.npy"),
    "TCN365_growing_anchor": load("tcn365v336_val.npy"),
    "TCN409": load("tcn409_val.npy"),
    "TCN409_replication": load("w409c_val.npy"),
}
names = list(columns)
design = np.column_stack(list(columns.values()))
gmv = np.load(WORK / "mat" / "gmv.npy", mmap_mode="r")
truth = np.log1p(gmv[:, 379:409].sum(axis=1, dtype=np.float64))
all_index = np.arange(len(truth))


def fit(index: np.ndarray, selected: np.ndarray, lam: float) -> np.ndarray:
    local = design[index][:, selected]
    target = truth[index]
    location = local.mean(axis=0)
    target_mean = float(target.mean())
    root_n = np.sqrt(len(index))
    coefficients, _ = nnls(
        np.vstack([
            (local - location) / root_n,
            np.sqrt(lam) * np.eye(len(selected)),
        ]),
        np.r_[(target - target_mean) / root_n, np.zeros(len(selected))],
        maxiter=10_000,
    )
    return np.r_[coefficients, target_mean - location @ coefficients]


def fit_capped(index: np.ndarray, lam: float, cap: float) -> np.ndarray:
    local = design[index]
    target = truth[index]
    location = local.mean(axis=0)
    target_mean = float(target.mean())
    root_n = np.sqrt(len(index))
    result = lsq_linear(
        np.vstack([
            (local - location) / root_n,
            np.sqrt(lam) * np.eye(local.shape[1]),
        ]),
        np.r_[(target - target_mean) / root_n, np.zeros(local.shape[1])],
        bounds=(0.0, cap),
        method="bvls",
        tol=1e-12,
        max_iter=10_000,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return np.r_[result.x, target_mean - location @ result.x]


def score(index: np.ndarray, selected: np.ndarray, weights: np.ndarray) -> float:
    prediction = np.clip(
        design[index][:, selected] @ weights[:-1] + weights[-1], 0, None
    )
    return float(np.sqrt(np.mean((truth[index] - prediction) ** 2)))


specs = {
    "120_all_six_l001": (np.arange(6), 0.001),
    "122_no_replication_l003": (np.arange(5), 0.003),
}
rng = np.random.default_rng(SEED)
rows = []
for _ in range(REPEATS):
    public = rng.choice(len(truth), 50_000, replace=False)
    private_mask = np.ones(len(truth), dtype=bool)
    private_mask[public] = False
    private = all_index[private_mask]
    row = {}
    for tag, (selected, lam) in specs.items():
        weights = fit(public, selected, lam)
        row[tag] = {
            "public": score(public, selected, weights),
            "private": score(private, selected, weights),
            "active": int(np.sum(weights[:-1] > 1e-8)),
        }
    capped = fit_capped(public, 0.001, 0.35)
    row["123_all_six_l001_cap035"] = {
        "public": score(public, np.arange(6), capped),
        "private": score(private, np.arange(6), capped),
        "active": int(np.sum(capped[:-1] > 1e-8)),
    }
    rows.append(row)

primary = np.asarray([row["120_all_six_l001"]["private"] for row in rows])
challenger = np.asarray([
    row["122_no_replication_l003"]["private"] for row in rows
])
delta = challenger - primary
capped_private = np.asarray([
    row["123_all_six_l001_cap035"]["private"] for row in rows
])
capped_delta = capped_private - primary
report = {
    "scope": "paired historical user-split audit; not an absolute competition-private forecast",
    "columns": names,
    "repeats": REPEATS,
    "public_users": 50_000,
    "private_users": 200_000,
    "solvers": {
        tag: {
            "lambda": lam,
            "columns": [names[index] for index in selected],
            "mean_public": float(np.mean([row[tag]["public"] for row in rows])),
            "mean_private": float(np.mean([row[tag]["private"] for row in rows])),
            "mean_active": float(np.mean([row[tag]["active"] for row in rows])),
        }
        for tag, (selected, lam) in specs.items()
    },
    "challenger_minus_primary_private": {
        "mean": float(delta.mean()),
        "se": float(delta.std(ddof=1) / np.sqrt(REPEATS)),
        "min": float(delta.min()),
        "median": float(np.median(delta)),
        "max": float(delta.max()),
        "p10": float(np.quantile(delta, 0.10)),
        "p90": float(np.quantile(delta, 0.90)),
        "challenger_win_fraction": float(np.mean(delta < 0)),
    },
    "capped_minus_primary_private": {
        "mean": float(capped_delta.mean()),
        "se": float(capped_delta.std(ddof=1) / np.sqrt(REPEATS)),
        "min": float(capped_delta.min()),
        "median": float(np.median(capped_delta)),
        "max": float(capped_delta.max()),
        "p10": float(np.quantile(capped_delta, 0.10)),
        "p90": float(np.quantile(capped_delta, 0.90)),
        "capped_win_fraction": float(np.mean(capped_delta < 0)),
    },
    "capped_solver": {
        "tag": "123_all_six_l001_cap035",
        "lambda": 0.001,
        "max_model_weight": 0.35,
        "columns": names,
        "mean_public": float(np.mean([
            row["123_all_six_l001_cap035"]["public"] for row in rows
        ])),
        "mean_private": float(capped_private.mean()),
        "mean_active": float(np.mean([
            row["123_all_six_l001_cap035"]["active"] for row in rows
        ])),
    },
    "decision": (
        "120 remains clean primary; 123 is preferred clean insurance because "
        "the concentration cap retains much more historical quality; 122 is "
        "the severe model-shift reserve"
    ),
}
OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
