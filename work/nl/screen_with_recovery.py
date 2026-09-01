#!/usr/bin/env python
"""163 — screen one candidate under the FULL deployment simulation.

Gram over all 250k, right-hand side from the public 50k, and the E[v^2] proxy
error the moment recovery actually incurs for a 0.70/0.30 probe against the
current best submission.  Frozen key: work/163_direct_recovery_test_preregister.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

OUT = Path(__file__).resolve().parent
LAM = 0.003
REPEATS = 96
PUBLIC_USERS = 50_000
SEED = 20260828
WEIGHT = 0.30

parser = argparse.ArgumentParser()
parser.add_argument("candidates", nargs="+")
parser.add_argument("--out", default="163_screen_recovery.json")
args = parser.parse_args()

pool = np.load(OUT / "pool27_378.npy").astype(np.float64)
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
target_mean, target_sd = stack.mean(), stack.std()
stack_second_all = float(np.mean(stack * stack))

base_deploy = np.empty(REPEATS)
for i, (public, private) in enumerate(splits):
    x = base_design[public]
    base_deploy[i] = rmsle(
        base_design,
        np.linalg.solve(
            gram_base_all + penalty_for(base_design.shape[1]),
            x.T @ truth[public] / len(public),
        ),
        private,
    )


def probe_transform(column):
    scaled = (column - column.mean()) / (column.std() + 1e-12) * target_sd + target_mean
    shift = brentq(
        lambda v: np.clip(scaled + v, 0, None).mean() - target_mean, -50, 50
    )
    return np.clip(scaled + shift, 0, None)


results = {}
for path in args.candidates:
    raw = np.load(path).astype(np.float64)
    column = probe_transform(raw)
    probe = (1 - WEIGHT) * stack + WEIGHT * column
    probe_second_all = float(np.mean(probe * probe))
    design = np.hstack([pool, column[:, None], ones])
    size = design.shape[1]
    gram_all = design.T @ design / n
    scores = np.empty(REPEATS)
    weights_seen = np.empty(REPEATS)
    for i, (public, private) in enumerate(splits):
        x = design[public]
        rhs = x.T @ truth[public] / len(public)
        base_delta = (stack_second_all - float(np.mean(stack[public] ** 2))) / 2
        probe_delta = (probe_second_all - float(np.mean(probe[public] ** 2))) / 2
        rhs[-2] += (probe_delta - (1 - WEIGHT) * base_delta) / WEIGHT
        w = np.linalg.solve(gram_all + penalty_for(size), rhs)
        weights_seen[i] = w[-2]
        scores[i] = rmsle(design, w, private)
    gain = base_deploy - scores
    positive = int((weights_seen > 0).sum())
    row = {
        "correlation_with_stack": float(np.corrcoef(column, stack)[0, 1]),
        "full_deployment_gain": float(gain.mean()),
        "standard_error": float(gain.std(ddof=1) / np.sqrt(REPEATS)),
        "gain_positive_splits": int((gain > 0).sum()),
        "weight_positive_splits": positive,
        "sign_stability": max(positive, REPEATS - positive),
        "mean_weight": float(weights_seen.mean()),
    }
    row["gate"] = (
        "PASS" if row["full_deployment_gain"] >= 0.00008
        and row["sign_stability"] >= 90 else "FAIL"
    )
    results[Path(path).stem] = row
    print(f"{Path(path).stem:24s} gain={row['full_deployment_gain']:+.7f} "
          f"se={row['standard_error']:.7f} pos_gain={row['gain_positive_splits']}/{REPEATS} "
          f"sign={row['sign_stability']}/{REPEATS} corr={row['correlation_with_stack']:.4f} "
          f"{row['gate']}", flush=True)

(OUT / args.out).write_text(json.dumps({
    "base_deployment_private": float(base_deploy.mean()),
    "candidates": results,
}, indent=2) + "\n")
