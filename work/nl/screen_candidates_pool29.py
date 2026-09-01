#!/usr/bin/env python
"""162 — screen candidate bases against the admitted pool at fold 378.

Gate (frozen in work/161_aggressive_round_register.json):
  conditional gain beyond the 27-column pool >= 0.00008,
  positive candidate weight in at least 90 of 96 splits,
  correlation with the stack >= 0.99 so the moment apparatus stays in the
  region that 155-159 showed to be safe.
Both the matched solve and the deployment-faithful solve are reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
LAM = 0.003
REPEATS = 96
PUBLIC_USERS = 50_000
SEED = 20260828

parser = argparse.ArgumentParser()
parser.add_argument("candidates", nargs="+", help="paths to *_val.npy vectors")
parser.add_argument("--out", default="162_screen.json")
args = parser.parse_args()

pool = np.load(OUT / "pool29_378.npy").astype(np.float64)
truth = np.load(OUT / "truth378.npy").astype(np.float64)
n = len(truth)
ones = np.ones((n, 1))
base_design = np.hstack([pool, ones])
gram_base_all = base_design.T @ base_design / n
all_index = np.arange(n)
rng = np.random.default_rng(SEED)
splits = []
for _ in range(REPEATS):
    public = rng.choice(n, PUBLIC_USERS, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[public] = False
    splits.append((public, all_index[mask]))


def penalty_for(size):
    penalty = np.eye(size) * LAM
    penalty[-1, -1] = 0.0
    return penalty


def rmsle(design, weights, index):
    residual = truth[index] - design[index] @ weights
    return float(np.sqrt(np.mean(residual * residual)))


stack = np.zeros(n)
folds = rng.permutation(n) % 5
for k in range(5):
    score_index = np.flatnonzero(folds == k)
    fit_index = np.flatnonzero(folds != k)
    x = base_design[fit_index]
    stack[score_index] = base_design[score_index] @ np.linalg.solve(
        x.T @ x / len(fit_index) + penalty_for(base_design.shape[1]),
        x.T @ truth[fit_index] / len(fit_index),
    )
print(f"pool stack OOF RMSLE {np.sqrt(np.mean((truth - stack) ** 2)):.7f}")

base_matched = np.empty(REPEATS)
base_deploy = np.empty(REPEATS)
for i, (public, private) in enumerate(splits):
    x = base_design[public]
    rhs = x.T @ truth[public] / len(public)
    size = base_design.shape[1]
    base_matched[i] = rmsle(
        base_design,
        np.linalg.solve(x.T @ x / len(public) + penalty_for(size), rhs),
        private,
    )
    base_deploy[i] = rmsle(
        base_design,
        np.linalg.solve(gram_base_all + penalty_for(size), rhs),
        private,
    )

results = {}
for path in args.candidates:
    vector = np.load(path).astype(np.float64)
    if vector.shape != truth.shape or not np.isfinite(vector).all():
        results[Path(path).stem] = {"error": f"bad vector {vector.shape}"}
        continue
    design = np.hstack([pool, vector[:, None], ones])
    size = design.shape[1]
    gram_all = design.T @ design / n
    matched = np.empty(REPEATS)
    deploy = np.empty(REPEATS)
    weights_seen = np.empty(REPEATS)
    for i, (public, private) in enumerate(splits):
        x = design[public]
        rhs = x.T @ truth[public] / len(public)
        w = np.linalg.solve(x.T @ x / len(public) + penalty_for(size), rhs)
        matched[i] = rmsle(design, w, private)
        weights_seen[i] = w[-2]
        deploy[i] = rmsle(
            design, np.linalg.solve(gram_all + penalty_for(size), rhs), private
        )
    matched_gain = base_matched - matched
    deploy_gain = base_deploy - deploy
    correlation = float(np.corrcoef(vector, stack)[0, 1])
    row = {
        "standalone_rmsle": float(np.sqrt(np.mean((truth - np.clip(vector, 0, None)) ** 2))),
        "correlation_with_stack": correlation,
        "matched_conditional_gain": float(matched_gain.mean()),
        "matched_standard_error": float(matched_gain.std(ddof=1) / np.sqrt(REPEATS)),
        "matched_positive_splits": int((matched_gain > 0).sum()),
        "deployment_conditional_gain": float(deploy_gain.mean()),
        "deployment_positive_splits": int((deploy_gain > 0).sum()),
        "positive_weight_splits": int((weights_seen > 0).sum()),
        "mean_weight": float(weights_seen.mean()),
    }
    row["gate"] = (
        "PASS" if row["matched_conditional_gain"] >= 0.00008
        and max(row["positive_weight_splits"], REPEATS - row["positive_weight_splits"]) >= 90
        and correlation >= 0.99
        else "FAIL"
    )
    results[Path(path).stem] = row
    print(f"{Path(path).stem:22s} standalone={row['standalone_rmsle']:.5f} "
          f"corr={correlation:.4f} matched={row['matched_conditional_gain']:+.7f} "
          f"deploy={row['deployment_conditional_gain']:+.7f} "
          f"w>0 {row['positive_weight_splits']}/{REPEATS} {row['gate']}", flush=True)

(OUT / args.out).write_text(json.dumps({
    "pool": "25 audit columns + w409c + decay_s93",
    "base_matched_private": float(base_matched.mean()),
    "base_deployment_private": float(base_deploy.mean()),
    "candidates": results,
}, indent=2) + "\n")
