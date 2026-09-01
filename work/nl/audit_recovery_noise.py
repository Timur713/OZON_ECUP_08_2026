#!/usr/bin/env python
"""155 — price the one error source the 152 audit does not simulate.

The moment recovery reads E[v^2] on all 250 000 users but the score S is a
public-50k quantity, so the recovered moment carries a bias term
(E_all[v^2] - E_public[v^2]) / 2 that a plain fit-50k audit never sees.  For
prediction columns this has always been negligible; raw historical columns have
different tails, so it is measured rather than assumed.

Three arms on the same 48 splits at fold 378:
  A  plain audit        Gram and right-hand side both from the public 50k
  B  deployment, clean  Gram from all 250k, right-hand side from the public 50k
  C  deployment, real   as B, plus the exact proxy error on the new columns
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

OUT = Path(__file__).resolve().parent
LAM = 0.003
REPEATS = 48
PUBLIC_USERS = 50_000
SEED = 20260828
K_GRID = [6, 12, 20, 30]
WEIGHT = 0.30
RANK_FOLDS = [288, 318, 348]

base = np.load(OUT / "base378.npy").astype(np.float64)
truth = np.load(OUT / "truth378.npy").astype(np.float64)
hist = np.load(OUT / "hist378.npy").astype(np.float64)
stack = np.load(OUT / "oof378_ridge.npy").astype(np.float64)
names = json.loads((OUT / "hist_keys.json").read_text())
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
ones = np.ones((n, 1))
rng = np.random.default_rng(SEED)
splits = []
for _ in range(REPEATS):
    public = rng.choice(n, PUBLIC_USERS, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[public] = False
    splits.append((public, all_index[mask]))

# The proxy error of column j on public set P, exactly as the recovery incurs
# it: probe = (1-w) * stack + w * column, and E[probe^2] is read on all users.
probes = [(1 - WEIGHT) * stack + WEIGHT * columns[:, j]
          for j in range(columns.shape[1])]
probe_second_all = np.array([float(np.mean(p * p)) for p in probes])
stack_second_all = float(np.mean(stack * stack))


def solve(gram, rhs, size):
    penalty = np.eye(size) * LAM
    penalty[-1, -1] = 0
    return np.linalg.solve(gram + penalty, rhs)


def score(design, weights, index):
    residual = truth[index] - design[index] @ weights
    return float(np.sqrt(np.mean(residual * residual)))


report = {"tag": "155_recovery_noise", "weight": WEIGHT, "by_k": {}}
for k in K_GRID:
    design = np.hstack([base, columns[:, :k], ones])
    size = design.shape[1]
    gram_all = design.T @ design / n
    arm_a = np.empty(REPEATS)
    arm_b = np.empty(REPEATS)
    arm_c = np.empty(REPEATS)
    base_arm = np.empty(REPEATS)
    base_design = np.hstack([base, ones])
    base_gram_all = base_design.T @ base_design / n
    for i, (public, private) in enumerate(splits):
        x = design[public]
        gram_public = x.T @ x / len(public)
        rhs_public = x.T @ truth[public] / len(public)
        arm_a[i] = score(design, solve(gram_public, rhs_public, size), private)
        arm_b[i] = score(design, solve(gram_all, rhs_public, size), private)

        rhs_proxy = rhs_public.copy()
        stack_second_public = float(np.mean(stack[public] ** 2))
        base_delta = (stack_second_all - stack_second_public) / 2
        for j in range(k):
            probe_second_public = float(np.mean(probes[j][public] ** 2))
            delta_probe = (probe_second_all[j] - probe_second_public) / 2
            rhs_proxy[base.shape[1] + j] += (
                delta_probe - (1 - WEIGHT) * base_delta
            ) / WEIGHT
        arm_c[i] = score(design, solve(gram_all, rhs_proxy, size), private)

        xb = base_design[public]
        base_arm[i] = score(
            base_design,
            solve(xb.T @ xb / len(public), xb.T @ truth[public] / len(public),
                  base_design.shape[1]),
            private,
        )
    report["by_k"][str(k)] = {
        "A_plain_audit_gain": float((base_arm - arm_a).mean()),
        "B_deployment_clean_gain": float((base_arm - arm_b).mean()),
        "C_deployment_with_proxy_error_gain": float((base_arm - arm_c).mean()),
        "cost_of_the_proxy": float((arm_c - arm_b).mean()),
        "C_positive_fraction": float(((base_arm - arm_c) > 0).mean()),
    }
    row = report["by_k"][str(k)]
    print(f"K={k:3d} A={row['A_plain_audit_gain']:+.7f} "
          f"B={row['B_deployment_clean_gain']:+.7f} "
          f"C={row['C_deployment_with_proxy_error_gain']:+.7f} "
          f"proxy_cost={row['cost_of_the_proxy']:+.7f} "
          f"pos={row['C_positive_fraction']:.2f}", flush=True)

(OUT / "155_recovery_noise.json").write_text(json.dumps(report, indent=2) + "\n")
