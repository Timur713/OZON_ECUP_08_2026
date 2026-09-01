#!/usr/bin/env python
"""156 — can the conditional block be saved by changing its parameterisation?

Frozen key: work/156_orthogonal_block_preregister.json.  Every arm simulates
the deployment solve faithfully: Gram over all 250k users, right-hand side from
the public 50k, and the E[v^2] proxy error where it really occurs.
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
NEW_LAMBDAS = [0.0, 0.003, 0.03, 0.1, 0.3, 1.0, 3.0]
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
all_index = np.arange(n)
ext = np.hstack([base, np.ones((n, 1))])
gram_ext = ext.T @ ext / n
probes = [(1 - WEIGHT) * stack + WEIGHT * columns[:, j]
          for j in range(columns.shape[1])]
probe_second_all = np.array([float(np.mean(p * p)) for p in probes])
stack_second_all = float(np.mean(stack * stack))

rng = np.random.default_rng(SEED)
splits = []
for _ in range(REPEATS):
    public = rng.choice(n, PUBLIC_USERS, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[public] = False
    splits.append((public, all_index[mask]))


def rmsle(design, weights, index):
    residual = truth[index] - design[index] @ weights
    return float(np.sqrt(np.mean(residual * residual)))


base_private = np.empty(REPEATS)
for i, (public, private) in enumerate(splits):
    x = ext[public]
    penalty = np.eye(ext.shape[1]) * BASE_LAM
    penalty[-1, -1] = 0
    weights = np.linalg.solve(
        x.T @ x / len(public) + penalty, x.T @ truth[public] / len(public)
    )
    base_private[i] = rmsle(ext, weights, private)

report = {"tag": "156_orthogonal_whitened_block", "base_mean_private":
          float(base_private.mean()), "by_k": {}}
for k in K_GRID:
    values = columns[:, :k]
    coefficients = np.linalg.solve(gram_ext, ext.T @ values / n)
    perpendicular = values - ext @ coefficients
    covariance = perpendicular.T @ perpendicular / n
    eigenvalues, vectors = np.linalg.eigh(covariance)
    whitener = vectors / np.sqrt(np.maximum(eigenvalues, 1e-12))
    whitened = perpendicular @ whitener
    # ext keeps its intercept in the last column, so the new block goes first
    design = np.hstack([ext[:, :-1], whitened, ext[:, -1:]])
    size = design.shape[1]
    gram_all = design.T @ design / n
    row = {"conditioning_of_new_block": float(
        eigenvalues.max() / max(eigenvalues.min(), 1e-12))}

    def run(new_lambda, with_proxy):
        scores = np.empty(REPEATS)
        penalty = np.zeros((size, size))
        penalty[np.arange(base.shape[1]), np.arange(base.shape[1])] = BASE_LAM
        index = np.arange(base.shape[1], base.shape[1] + k)
        penalty[index, index] = new_lambda
        for i, (public, private) in enumerate(splits):
            x = design[public]
            rhs = x.T @ truth[public] / len(public)
            if with_proxy:
                stack_second_public = float(np.mean(stack[public] ** 2))
                base_delta = (stack_second_all - stack_second_public) / 2
                raw_delta = np.empty(k)
                for j in range(k):
                    probe_second_public = float(np.mean(probes[j][public] ** 2))
                    raw_delta[j] = (
                        (probe_second_all[j] - probe_second_public) / 2
                        - (1 - WEIGHT) * base_delta
                    ) / WEIGHT
                # the same probes, expressed in the transformed basis
                ext_delta = np.zeros(ext.shape[1])
                ext_delta[-1] = 0.0
                rhs[index] += whitener.T @ (raw_delta - coefficients.T @ ext_delta)
            weights = np.linalg.solve(gram_all + penalty, rhs)
            scores[i] = rmsle(design, weights, private)
        gain = base_private - scores
        return float(gain.mean()), float((gain > 0).mean())

    row["D_and_E_by_new_lambda"] = {}
    for new_lambda in NEW_LAMBDAS:
        mean_gain, positive = run(new_lambda, True)
        row["D_and_E_by_new_lambda"][str(new_lambda)] = {
            "mean_private_gain": mean_gain, "positive_fraction": positive
        }
    best_lambda = max(
        row["D_and_E_by_new_lambda"],
        key=lambda key: row["D_and_E_by_new_lambda"][key]["mean_private_gain"],
    )
    row["best_new_lambda"] = float(best_lambda)
    mean_gain, positive = run(float(best_lambda), False)
    row["F_no_proxy_error_at_best_lambda"] = {
        "mean_private_gain": mean_gain, "positive_fraction": positive
    }
    report["by_k"][str(k)] = row
    best = row["D_and_E_by_new_lambda"][best_lambda]
    print(f"K={k:3d} cond={row['conditioning_of_new_block']:.1f} "
          f"bestlam={best_lambda:>5s} D/E={best['mean_private_gain']:+.7f} "
          f"pos={best['positive_fraction']:.2f} | "
          f"F={mean_gain:+.7f} pos={positive:.2f}", flush=True)

(OUT / "156_orthogonal_block.json").write_text(json.dumps(report, indent=2) + "\n")
