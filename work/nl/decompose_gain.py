#!/usr/bin/env python
"""Where does the fold-378 nonlinear gain come from?

Separates three explanations that the single blend number conflates:
  L1  linear ridge over the 25 bases                       (current recipe)
  L2  linear ridge over bases plus the 44 user features    (missing linear terms)
  L3  L2 plus per-feature-bucket rescaling of the stack    (conditional slope)
  NL  gradient boosting over bases plus features           (full nonlinearity)
All are scored out of fold on the same user partition.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "nl"
FOLDS = 5
SEED = 20260828

base = np.load(OUT / "base378.npy").astype(np.float64)
feat = np.load(OUT / "feat378.npy").astype(np.float64)
truth = np.load(OUT / "truth378.npy").astype(np.float64)
nl = np.load(OUT / "oof378_nl_full.npy").astype(np.float64)
fkeys = json.loads((OUT / "feat378_keys.json").read_text())

n = len(truth)
fold_of = np.random.default_rng(SEED).permutation(n) % FOLDS
standard = (feat - feat.mean(0)) / (feat.std(0) + 1e-9)


def oof(design, lam):
    prediction = np.zeros(n)
    penalty = np.eye(design.shape[1]) * lam
    penalty[-1, -1] = 0
    for fold in range(FOLDS):
        score_index = np.flatnonzero(fold_of == fold)
        fit_index = np.flatnonzero(fold_of != fold)
        x = design[fit_index]
        coef = np.linalg.solve(
            x.T @ x / len(fit_index) + penalty,
            x.T @ truth[fit_index] / len(fit_index),
        )
        prediction[score_index] = design[score_index] @ coef
    residual = truth - np.clip(prediction, 0.0, None)
    return float(np.sqrt(np.mean(residual * residual))), prediction


ones = np.ones((n, 1))
l1, pred_l1 = oof(np.hstack([base, ones]), 0.003)
l2, pred_l2 = oof(np.hstack([base, standard, ones]), 0.003)

# Conditional slope: let the fitted stack be rescaled inside deciles of the
# strongest historical feature, which is what a tree stacker does first.
driver = feat[:, fkeys.index("gmv_d180")]
edges = np.quantile(driver, np.linspace(0, 1, 11)[1:-1])
bucket = np.searchsorted(edges, driver)
indicators = np.zeros((n, 10))
indicators[np.arange(n), bucket] = 1.0
l3, _ = oof(
    np.hstack([base, standard, indicators[:, 1:], indicators * pred_l1[:, None], ones]),
    0.003,
)

blend, _ = oof(np.column_stack([pred_l1, nl, np.ones(n)]), 1e-9)
report = {
    "L1_bases_only": l1,
    "L2_bases_plus_linear_features": l2,
    "L3_plus_decile_conditional_slope": l3,
    "NL_blend_ridge_plus_boosted_stacker": blend,
    "gain_L2_over_L1": l1 - l2,
    "gain_L3_over_L1": l1 - l3,
    "gain_NL_over_L1": l1 - blend,
    "gain_NL_over_L2": l2 - blend,
}
(OUT / "gain_decomposition_fold378.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
