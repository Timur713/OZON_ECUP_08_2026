#!/usr/bin/env python
"""152, second condition — is the paid conditional gain a property of fold 378
or of every fold?

Same fit-50k / score-200k protocol at all four folds of the 150 surrogate, with
the surrogate stack as the base column so the four folds are comparable to each
other.  Also compares raw top-K columns against label-free PCA directions,
which cost the same degrees of freedom but are chosen without any target.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
FOLDS = [288, 318, 348, 378]
RANK_FOLDS = [288, 318, 348]
LAM = 0.003
REPEATS = 48
PUBLIC_USERS = 50_000
SEED = 20260828
K_GRID = [6, 12, 24, 76]

names = json.loads((OUT / "hist_keys.json").read_text())
ranking = np.mean(
    [np.abs(np.load(OUT / f"state_{f}_beta.npy")) for f in RANK_FOLDS], axis=0
)
order = np.argsort(-ranking)


def fit_model(design, truth, fit_index, score_index):
    x = design[fit_index]
    gram = x.T @ x / len(fit_index)
    rhs = x.T @ truth[fit_index] / len(fit_index)
    penalty = np.eye(design.shape[1]) * LAM
    penalty[-1, -1] = 0
    weights = np.linalg.solve(gram + penalty, rhs)
    residual = truth[score_index] - design[score_index] @ weights
    return float(np.sqrt(np.mean(residual * residual)))


report = {"tag": "152_fold_robustness", "folds": {}, "pca_fold378": {}}
for fold in FOLDS:
    truth = np.load(OUT / f"state_{fold}_truth.npy").astype(np.float64)
    stack = np.load(OUT / f"state_{fold}_affine.npy").astype(np.float64)
    hist = np.load(OUT / f"state_{fold}_standard.npy").astype(np.float64)
    n = len(truth)
    all_index = np.arange(n)
    rng = np.random.default_rng(SEED)
    splits = []
    for _ in range(REPEATS):
        public = rng.choice(n, PUBLIC_USERS, replace=False)
        mask = np.ones(n, dtype=bool)
        mask[public] = False
        splits.append((public, all_index[mask]))
    base_design = np.column_stack([stack, np.ones(n)])
    base_private = np.array(
        [fit_model(base_design, truth, p, q) for p, q in splits]
    )
    row = {}
    for k in K_GRID:
        design = np.column_stack([stack, hist[:, order[:k]], np.ones(n)])
        private = np.array([fit_model(design, truth, p, q) for p, q in splits])
        gain = base_private - private
        row[str(k)] = {
            "mean_private_gain": float(gain.mean()),
            "standard_error": float(gain.std(ddof=1) / np.sqrt(REPEATS)),
            "positive_fraction": float((gain > 0).mean()),
        }
        print(f"fold {fold} K={k:3d} private={gain.mean():+.7f} "
              f"pos={(gain > 0).mean():.2f}", flush=True)
    report["folds"][str(fold)] = row
    del hist, stack, truth

# Label-free PCA directions at fold 378 against the real 25-base pool.
base = np.load(OUT / "base378.npy").astype(np.float64)
truth = np.load(OUT / "truth378.npy").astype(np.float64)
hist = np.load(OUT / "hist378.npy").astype(np.float64)
hist = (hist - hist.mean(0)) / (hist.std(0) + 1e-9)
n = len(truth)
all_index = np.arange(n)
rng = np.random.default_rng(SEED)
splits = []
for _ in range(REPEATS):
    public = rng.choice(n, PUBLIC_USERS, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[public] = False
    splits.append((public, all_index[mask]))
ones = np.ones((n, 1))
base_design = np.hstack([base, ones])
base_private = np.array([fit_model(base_design, truth, p, q) for p, q in splits])
covariance = hist.T @ hist / n
values, vectors = np.linalg.eigh(covariance)
vectors = vectors[:, ::-1]
scores = hist @ vectors
scores /= scores.std(0) + 1e-9
for k in K_GRID:
    design = np.hstack([base, scores[:, :k], ones])
    private = np.array([fit_model(design, truth, p, q) for p, q in splits])
    gain = base_private - private
    report["pca_fold378"][str(k)] = {
        "mean_private_gain": float(gain.mean()),
        "standard_error": float(gain.std(ddof=1) / np.sqrt(REPEATS)),
        "positive_fraction": float((gain > 0).mean()),
    }
    print(f"PCA378 K={k:3d} private={gain.mean():+.7f} "
          f"pos={(gain > 0).mean():.2f}", flush=True)

worst = {
    k: min(report["folds"][str(f)][k]["mean_private_gain"] for f in FOLDS)
    for k in map(str, K_GRID)
}
report["worst_fold_by_k"] = worst
report["condition_two_threshold"] = 0.00020
report["condition_two_pass_k"] = [k for k, v in worst.items() if v >= 0.00020]
(OUT / "152_fold_robustness.json").write_text(json.dumps(report, indent=2) + "\n")
print("\nworst fold by K:", json.dumps(worst, indent=2))
print("K values passing condition two:", report["condition_two_pass_k"])
