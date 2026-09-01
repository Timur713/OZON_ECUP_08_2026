#!/usr/bin/env python
"""157 — does the Gram/right-hand-side mismatch hurt the EXISTING pool?

155 showed that solving with the 250k Gram against a public-50k right-hand side
costs 0.0019 once raw historical columns are in the design.  If the same
mismatch also hurt the 25-base pool on its own, every private estimate the
project has published would be optimistic.  This measures it at K = 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
LAM = 0.003
REPEATS = 48
PUBLIC_USERS = 50_000
SEED = 20260828

base = np.load(OUT / "base378.npy").astype(np.float64)
truth = np.load(OUT / "truth378.npy").astype(np.float64)
n = len(truth)
design = np.hstack([base, np.ones((n, 1))])
gram_all = design.T @ design / n
penalty = np.eye(design.shape[1]) * LAM
penalty[-1, -1] = 0
all_index = np.arange(n)
rng = np.random.default_rng(SEED)
splits = []
for _ in range(REPEATS):
    public = rng.choice(n, PUBLIC_USERS, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[public] = False
    splits.append((public, all_index[mask]))


def rmsle(weights, index):
    residual = truth[index] - design[index] @ weights
    return float(np.sqrt(np.mean(residual * residual)))


matched = np.empty(REPEATS)
mismatched = np.empty(REPEATS)
oracle = np.empty(REPEATS)
oracle_weights = np.linalg.solve(
    gram_all + penalty, design.T @ truth / n
)
correlations = np.corrcoef(base.T)
for i, (public, private) in enumerate(splits):
    x = design[public]
    rhs = x.T @ truth[public] / len(public)
    matched[i] = rmsle(np.linalg.solve(x.T @ x / len(public) + penalty, rhs), private)
    mismatched[i] = rmsle(np.linalg.solve(gram_all + penalty, rhs), private)
    oracle[i] = rmsle(oracle_weights, private)

report = {
    "tag": "157_base_pool_mismatch",
    "note": "K = 0, the 25 admitted validation columns only",
    "matched_gram_and_rhs_from_50k": float(matched.mean()),
    "mismatched_gram_250k_rhs_50k": float(mismatched.mean()),
    "population_oracle_both_from_250k": float(oracle.mean()),
    "cost_of_the_mismatch": float((mismatched - matched).mean()),
    "cost_of_fitting_on_50k_at_all": float((matched - oracle).mean()),
    "mismatch_hurts_in_fraction_of_splits": float((mismatched > matched).mean()),
    "median_abs_correlation_between_bases": float(
        np.median(np.abs(correlations[np.triu_indices_from(correlations, 1)]))
    ),
    "min_abs_correlation_between_bases": float(
        np.min(np.abs(correlations[np.triu_indices_from(correlations, 1)]))
    ),
}
(OUT / "157_base_pool_mismatch.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
