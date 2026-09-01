#!/usr/bin/env python
"""160 — measure the two final recipes against each other on held-out users.

Frozen key: work/160_final_pair_direct_preregister.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

OUT = Path(__file__).resolve().parent
REPEATS = 200
PUBLIC_USERS = 50_000
SEED = 20260828

base = np.load(OUT / "base378.npy").astype(np.float64)
truth = np.load(OUT / "truth378.npy").astype(np.float64)
n = len(truth)
design = np.hstack([base, np.ones((n, 1))])
size = design.shape[1]
gram_all = design.T @ design / n
all_index = np.arange(n)
rng = np.random.default_rng(SEED)
splits = []
for _ in range(REPEATS):
    public = rng.choice(n, PUBLIC_USERS, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[public] = False
    splits.append((public, all_index[mask]))


def penalty_matrix(lam):
    penalty = np.eye(size) * lam
    penalty[-1, -1] = 0.0
    return penalty


def solve_free(gram, rhs, lam):
    return np.linalg.solve(gram + penalty_matrix(lam), rhs)


def solve_nonnegative(gram, rhs, lam):
    system = gram + penalty_matrix(lam)
    start = solve_free(gram, rhs, lam)
    start[:-1] = np.maximum(start[:-1], 0.0)
    result = minimize(
        lambda w: 0.5 * w @ system @ w - rhs @ w,
        start,
        jac=lambda w: system @ w - rhs,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * (size - 1) + [(None, None)],
        options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 10_000, "maxls": 50},
    )
    return result.x


def rmsle(weights, index):
    residual = truth[index] - design[index] @ weights
    return float(np.sqrt(np.mean(residual * residual)))


rows = {"matched": {"130": [], "147": []}, "deployment": {"130": [], "147": []}}
active_counts = []
for public, private in splits:
    x = design[public]
    gram_public = x.T @ x / len(public)
    rhs = x.T @ truth[public] / len(public)
    for label, gram in (("matched", gram_public), ("deployment", gram_all)):
        rows[label]["130"].append(rmsle(solve_free(gram, rhs, 0.003), private))
        weights = solve_nonnegative(gram, rhs, 0.001)
        rows[label]["147"].append(rmsle(weights, private))
        if label == "deployment":
            active_counts.append(int((weights[:-1] > 1e-8).sum()))

report = {"tag": "160_final_pair_direct_comparison", "repeats": REPEATS,
          "mean_active_components_of_147_recipe": float(np.mean(active_counts)),
          "arms": {}}
for label in ("matched", "deployment"):
    a = np.asarray(rows[label]["130"])
    b = np.asarray(rows[label]["147"])
    difference = b - a
    report["arms"][label] = {
        "mean_private_130_recipe": float(a.mean()),
        "mean_private_147_recipe": float(b.mean()),
        "mean_difference_147_minus_130": float(difference.mean()),
        "standard_error": float(difference.std(ddof=1) / np.sqrt(REPEATS)),
        "fraction_of_splits_where_147_wins": float((difference < 0).mean()),
    }
    row = report["arms"][label]
    print(f"{label:11s} 130={row['mean_private_130_recipe']:.7f} "
          f"147={row['mean_private_147_recipe']:.7f} "
          f"diff={row['mean_difference_147_minus_130']:+.7f} "
          f"se={row['standard_error']:.7f} "
          f"147_wins={row['fraction_of_splits_where_147_wins']:.3f}", flush=True)

deployment = report["arms"]["deployment"]
if deployment["mean_difference_147_minus_130"] < -deployment["standard_error"] \
        and deployment["fraction_of_splits_where_147_wins"] > 0.5:
    report["verdict"] = "reverses_the_roles"
elif deployment["mean_difference_147_minus_130"] > deployment["standard_error"]:
    report["verdict"] = "confirms_130_as_slot_one"
else:
    report["verdict"] = "inconclusive"
(OUT / "160_final_pair_direct.json").write_text(json.dumps(report, indent=2) + "\n")
print("verdict:", report["verdict"])
