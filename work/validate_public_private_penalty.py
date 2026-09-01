#!/usr/bin/env python
"""Empirically audit public-to-private ridge optimism on 50k/200k user splits."""
import json
import os

import numpy as np

ROOT = "/Users/timur/Desktop/dev/OZON_ECUP_2026_3"
WORK = os.path.join(ROOT, "work")
PUBLIC_USERS = 50_000
REPEATS = 24
LAMBDAS = (0.001, 0.003, 0.01, 0.03)


def load(name):
    return np.load(os.path.join(WORK, name)).astype(np.float64)


columns = {
    "gbdt262": load("v4_262_valpred.npy"),
    "gbdt159": load("gbdt_v5_val.npy"),
    "seq180": load("seq_val.npy"),
    "tcn45": load("tcn45_val.npy"),
    "tcn90": load("tcn90_val.npy"),
    "tcn180two": load("tcn180two_val.npy"),
    "tcn270": load("tcn270_val.npy"),
    "tcn365": load("tcn365_val.npy"),
    "tcn365b": load("tcn365b_val.npy"),
    "tcn365v336": load("tcn365v336_val.npy"),
    "tcn409": load("tcn409_val.npy"),
    "gru180": load("gru180_val.npy"),
    "W45": np.mean([load(f"w45{seed}_val.npy") for seed in "abcd"], axis=0),
    "W60": np.mean([load(f"w60{seed}_val.npy") for seed in "abc"], axis=0),
    "W90": np.mean([load(f"w90{seed}_val.npy") for seed in "abc"], axis=0),
    "W120": np.mean([load(f"w120{seed}_val.npy") for seed in "abc"], axis=0),
    "W150": load("w150a_val.npy"),
    "W180": np.mean([load(f"w180{seed}_val.npy") for seed in "ab"], axis=0),
    "W210": load("w210a_val.npy"),
    "W270": load("w270a_val.npy"),
    "W300": load("w300a_val.npy"),
    "W365": np.mean([load(f"w365{seed}_val.npy") for seed in "ab"], axis=0),
    "W409": load("w409a_val.npy"),
    "cls300": load("cls300_val_server_val.npy"),
    "cls409": load("cls409_val_server_val.npy"),
}

gmv = np.load(os.path.join(WORK, "mat", "gmv.npy"), mmap_mode="r")
truth = np.log1p(gmv[:, 379:409].sum(axis=1, dtype=np.float64))
design = np.column_stack([*columns.values(), np.ones_like(truth)])
assert design.shape[0] == len(truth) and np.isfinite(design).all()


def solve(index, lam):
    x = design[index]
    y = truth[index]
    gram = x.T @ x / len(index)
    rhs = x.T @ y / len(index)
    penalty = np.eye(design.shape[1]) * lam
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(gram + penalty, rhs)
    degrees = np.trace(gram @ np.linalg.inv(gram + penalty))
    return weights, float(degrees)


def rmse(index, weights):
    residual = truth[index] - design[index] @ weights
    return float(np.sqrt(np.mean(residual * residual)))


rng = np.random.default_rng(20260825)
all_index = np.arange(len(truth))
report = []
for lam in LAMBDAS:
    oracle_weights, oracle_df = solve(all_index, lam)
    oracle_score = rmse(all_index, oracle_weights)
    rows = []
    for _ in range(REPEATS):
        public = rng.choice(len(truth), PUBLIC_USERS, replace=False)
        private_mask = np.ones(len(truth), dtype=bool)
        private_mask[public] = False
        private = all_index[private_mask]
        weights, degrees = solve(public, lam)
        public_score = rmse(public, weights)
        private_score = rmse(private, weights)
        population_score = rmse(all_index, weights)
        oracle_public = rmse(public, oracle_weights)
        oracle_private = rmse(private, oracle_weights)
        rows.append((
            degrees,
            public_score,
            private_score,
            population_score,
            oracle_public - public_score,
            private_score - oracle_private,
        ))
    rows = np.asarray(rows)
    (
        mean_df,
        mean_public,
        mean_private,
        mean_population,
        mean_public_optimism,
        mean_private_excess,
    ) = rows.mean(axis=0)
    report.append({
        "lambda": lam,
        "bases": len(columns),
        "repeats": REPEATS,
        "oracle_score_250k": oracle_score,
        "oracle_df_250k": oracle_df,
        "mean_df_50k": mean_df,
        "mean_public_score": mean_public,
        "mean_private_score": mean_private,
        "mean_population_score": mean_population,
        "private_minus_public": mean_private - mean_public,
        "population_minus_public": mean_population - mean_public,
        "private_minus_population": mean_private - mean_population,
        "public_optimism_vs_oracle": mean_public_optimism,
        "private_excess_vs_oracle": mean_private_excess,
        "fitted_public_to_private_penalty": mean_public_optimism + mean_private_excess,
        "fitted_penalty_sd": float((rows[:, 4] + rows[:, 5]).std(ddof=1)),
        "fitted_penalty_se": float((rows[:, 4] + rows[:, 5]).std(ddof=1) / np.sqrt(REPEATS)),
        "half_penalty_prediction": oracle_score * mean_df / (2 * PUBLIC_USERS),
        "full_gap_prediction": oracle_score * mean_df / PUBLIC_USERS,
        "private_minus_public_sd": float((rows[:, 2] - rows[:, 1]).std(ddof=1)),
    })

print(json.dumps(report, indent=2))
