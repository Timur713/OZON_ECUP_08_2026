#!/usr/bin/env python
"""159 — 155 and 156 redone against the right baseline.

155 and 156 compared a deployment-faithful augmented solve against a MATCHED
50k baseline.  That charges the augmented arm for the Gram/right-hand-side
mismatch which 130 already pays too, so it understates every arm by the 0.000626
that 157 measured.  The decision-relevant comparison keeps the mismatch on both
sides: what does the block add ON TOP of the solve the project actually
performs?

Every arm below solves with the Gram over all 250k and a right-hand side from
the public 50k.  Gains are against the base pool solved the same way.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

OUT = Path(__file__).resolve().parent
BASE_LAM = 0.003
REPEATS = 48
PUBLIC_USERS = 50_000
SEED = 20260828
K_GRID = [6, 12, 20, 30]
NEW_LAMBDAS = [0.003, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
WEIGHT = 0.30
RANK_FOLDS = [288, 318, 348]

base = np.load(OUT / "base378.npy").astype(np.float64)
truth = np.load(OUT / "truth378.npy").astype(np.float64)
hist = np.load(OUT / "hist378.npy").astype(np.float64)
stack = np.load(OUT / "oof378_ridge.npy").astype(np.float64)
unique = json.loads((OUT / "hist_unique.json").read_text())["kept_indices"]
ranking = np.mean(
    [np.abs(np.load(OUT / f"state_{f}_beta.npy")) for f in RANK_FOLDS], axis=0
)
masked = np.full(len(ranking), -np.inf)
masked[unique] = ranking[unique]
order = np.argsort(-masked)[:max(K_GRID)]
target_mean, target_sd = stack.mean(), stack.std()


def probe_transform(column):
    scaled = (column - column.mean()) / (column.std() + 1e-12) * target_sd + target_mean
    shift = brentq(
        lambda v: np.clip(scaled + v, 0, None).mean() - target_mean, -50, 50
    )
    return np.clip(scaled + shift, 0, None)


columns = np.column_stack([probe_transform(hist[:, c]) for c in order])
n = len(truth)
ext = np.hstack([base, np.ones((n, 1))])
gram_ext_all = ext.T @ ext / n
probes = [(1 - WEIGHT) * stack + WEIGHT * columns[:, j]
          for j in range(columns.shape[1])]
probe_second_all = np.array([float(np.mean(p * p)) for p in probes])
stack_second_all = float(np.mean(stack * stack))
all_index = np.arange(n)
rng = np.random.default_rng(SEED)
splits = []
for _ in range(REPEATS):
    public = rng.choice(n, PUBLIC_USERS, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[public] = False
    splits.append((public, all_index[mask]))

proxy_delta = np.empty((REPEATS, columns.shape[1]))
for i, (public, _) in enumerate(splits):
    base_delta = (stack_second_all - float(np.mean(stack[public] ** 2))) / 2
    for j in range(columns.shape[1]):
        probe_second_public = float(np.mean(probes[j][public] ** 2))
        proxy_delta[i, j] = (
            (probe_second_all[j] - probe_second_public) / 2
            - (1 - WEIGHT) * base_delta
        ) / WEIGHT


def rmsle(design, weights, index):
    residual = truth[index] - design[index] @ weights
    return float(np.sqrt(np.mean(residual * residual)))


penalty_base = np.eye(ext.shape[1]) * BASE_LAM
penalty_base[-1, -1] = 0
baseline = np.empty(REPEATS)
for i, (public, private) in enumerate(splits):
    rhs = ext[public].T @ truth[public] / len(public)
    baseline[i] = rmsle(
        ext, np.linalg.solve(gram_ext_all + penalty_base, rhs), private
    )

report = {
    "tag": "159_deployment_baseline",
    "baseline_is": "the 25-base pool solved the way the project actually solves it",
    "baseline_mean_private": float(baseline.mean()),
    "by_k": {},
}
for k in K_GRID:
    values = columns[:, :k]
    coefficients = np.linalg.solve(gram_ext_all, ext.T @ values / n)
    perpendicular = values - ext @ coefficients
    covariance = perpendicular.T @ perpendicular / n
    eigenvalues, vectors = np.linalg.eigh(covariance)
    whitener = vectors / np.sqrt(np.maximum(eigenvalues, 1e-12))
    whitened = perpendicular @ whitener
    row = {}

    def evaluate(block, new_lambda, with_proxy, transform_delta):
        design = np.hstack([ext[:, :-1], block, ext[:, -1:]])
        size = design.shape[1]
        gram_all = design.T @ design / n
        penalty = np.zeros((size, size))
        penalty[np.arange(base.shape[1]), np.arange(base.shape[1])] = BASE_LAM
        index = np.arange(base.shape[1], base.shape[1] + k)
        penalty[index, index] = new_lambda
        scores = np.empty(REPEATS)
        for i, (public, private) in enumerate(splits):
            x = design[public]
            rhs = x.T @ truth[public] / len(public)
            if with_proxy:
                rhs[index] += transform_delta(proxy_delta[i, :k])
            scores[i] = rmsle(
                design, np.linalg.solve(gram_all + penalty, rhs), private
            )
        gain = baseline - scores
        return float(gain.mean()), float((gain > 0).mean())

    mean_gain, positive = evaluate(values, BASE_LAM, True, lambda d: d)
    row["raw_columns_as_153_would_have_done"] = {
        "mean_private_gain": mean_gain, "positive_fraction": positive
    }
    row["orthogonal_whitened_by_new_lambda"] = {}
    for new_lambda in NEW_LAMBDAS:
        mean_gain, positive = evaluate(
            whitened, new_lambda, True, lambda d: whitener.T @ d
        )
        row["orthogonal_whitened_by_new_lambda"][str(new_lambda)] = {
            "mean_private_gain": mean_gain, "positive_fraction": positive
        }
    best = max(
        row["orthogonal_whitened_by_new_lambda"],
        key=lambda key:
        row["orthogonal_whitened_by_new_lambda"][key]["mean_private_gain"],
    )
    row["best_new_lambda"] = float(best)
    mean_gain, positive = evaluate(
        whitened, float(best), False, lambda d: whitener.T @ d
    )
    row["orthogonal_whitened_without_proxy_error"] = {
        "mean_private_gain": mean_gain, "positive_fraction": positive
    }
    report["by_k"][str(k)] = row
    raw = row["raw_columns_as_153_would_have_done"]
    orth = row["orthogonal_whitened_by_new_lambda"][best]
    print(f"K={k:3d} raw={raw['mean_private_gain']:+.7f} "
          f"pos={raw['positive_fraction']:.2f} | orth(lam={best:>5s})="
          f"{orth['mean_private_gain']:+.7f} pos={orth['positive_fraction']:.2f} | "
          f"orth_no_proxy={mean_gain:+.7f} pos={positive:.2f}", flush=True)

(OUT / "159_deployment_baseline.json").write_text(json.dumps(report, indent=2) + "\n")
