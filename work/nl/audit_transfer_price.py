#!/usr/bin/env python
"""158 — re-measure the public-to-private price per degree of freedom under the
solve the project actually performs.

The standing 0.0000395 per df was calibrated on MATCHED fits, where the Gram
and the right-hand side both come from the same 50k users.  The real solve uses
the Gram over all 250k with a right-hand side recovered from public moments.
157 showed that mismatch costs 0.000626 on the 25-base pool alone.  This
measures the price per df directly in both settings, as
(private score - fitted public score) / degrees of freedom.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
REPEATS = 48
PUBLIC_USERS = 50_000
SEED = 20260828
LAMBDAS = [0.001, 0.003, 0.01, 0.03, 0.1]

base = np.load(OUT / "base378.npy").astype(np.float64)
truth = np.load(OUT / "truth378.npy").astype(np.float64)
n = len(truth)
design = np.hstack([base, np.ones((n, 1))])
gram_all = design.T @ design / n
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


report = {"tag": "158_transfer_price_under_the_real_solve", "by_lambda": {}}
for lam in LAMBDAS:
    penalty = np.eye(design.shape[1]) * lam
    penalty[-1, -1] = 0
    matched_gap = np.empty(REPEATS)
    mismatched_gap = np.empty(REPEATS)
    degrees = np.empty(REPEATS)
    for i, (public, private) in enumerate(splits):
        x = design[public]
        gram_public = x.T @ x / len(public)
        rhs = x.T @ truth[public] / len(public)
        degrees[i] = float(np.trace(
            gram_public @ np.linalg.inv(gram_public + penalty)
        ))
        weights = np.linalg.solve(gram_public + penalty, rhs)
        matched_gap[i] = rmsle(weights, private) - rmsle(weights, public)
        weights = np.linalg.solve(gram_all + penalty, rhs)
        mismatched_gap[i] = rmsle(weights, private) - rmsle(weights, public)
    report["by_lambda"][str(lam)] = {
        "degrees_of_freedom": float(degrees.mean()),
        "matched_price_per_df": float(matched_gap.mean() / degrees.mean()),
        "real_solve_price_per_df": float(mismatched_gap.mean() / degrees.mean()),
        "ratio": float(mismatched_gap.mean() / matched_gap.mean()),
        "matched_total_gap": float(matched_gap.mean()),
        "real_solve_total_gap": float(mismatched_gap.mean()),
    }
    row = report["by_lambda"][str(lam)]
    print(f"lam={lam:<6g} df={row['degrees_of_freedom']:5.2f} "
          f"matched={row['matched_price_per_df']:.3e} "
          f"real={row['real_solve_price_per_df']:.3e} "
          f"ratio={row['ratio']:.2f}", flush=True)

(OUT / "158_transfer_price.json").write_text(json.dumps(report, indent=2) + "\n")
