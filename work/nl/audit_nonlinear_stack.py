#!/usr/bin/env python
"""Does a nonlinear stacker beat the linear ridge on fold 378?

Existence test only: 5-fold user cross-fit over all 250 000 users with known
labels.  Both stackers see exactly the same rows, so the difference is
attributable to functional form and to the extra user features, not to sample
size.  Leaderboard scores are not used anywhere.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "nl"
LAM = 0.003
FOLDS = 5
SEED = 20260828

base = np.load(OUT / "base378.npy").astype(np.float64)
feat = np.load(OUT / "feat378.npy").astype(np.float64)
truth = np.load(OUT / "truth378.npy").astype(np.float64)
keys = json.loads((OUT / "base378_keys.json").read_text())
fkeys = json.loads((OUT / "feat378_keys.json").read_text())

n = len(truth)
rng = np.random.default_rng(SEED)
fold_of = rng.permutation(n) % FOLDS


def rmsle(pred, index):
    residual = truth[index] - np.clip(pred, 0.0, None)
    return float(np.sqrt(np.mean(residual * residual)))


def ridge_fit(design, fit_index):
    x = design[fit_index]
    gram = x.T @ x / len(fit_index)
    rhs = x.T @ truth[fit_index] / len(fit_index)
    penalty = np.eye(design.shape[1]) * LAM
    penalty[-1, -1] = 0
    return np.linalg.solve(gram + penalty, rhs)


design = np.column_stack([base, np.ones(n)])
stack_oof = np.zeros(n)
nl_oof = np.zeros(n)
nlb_oof = np.zeros(n)

params = dict(
    objective="regression",
    metric="l2",
    learning_rate=0.03,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    num_threads=8,
    verbose=-1,
    max_bin=127,
)
matrix_full = np.column_stack([base, feat])
names = keys + fkeys
start = time.time()
for fold in range(FOLDS):
    score_index = np.flatnonzero(fold_of == fold)
    fit_index = np.flatnonzero(fold_of != fold)
    weights = ridge_fit(design, fit_index)
    stack_oof[score_index] = design[score_index] @ weights

    # (a) nonlinear stacker over base predictions only
    model = lgb.train(
        params,
        lgb.Dataset(base[fit_index], truth[fit_index], feature_name=keys),
        num_boost_round=600,
    )
    nlb_oof[score_index] = model.predict(base[score_index])

    # (b) nonlinear stacker over base predictions plus user features
    model = lgb.train(
        params,
        lgb.Dataset(matrix_full[fit_index], truth[fit_index], feature_name=names),
        num_boost_round=600,
    )
    nl_oof[score_index] = model.predict(matrix_full[score_index])
    print(f"fold {fold} done {time.time() - start:.0f}s", flush=True)

all_index = np.arange(n)
report = {
    "linear_ridge_oof": rmsle(stack_oof, all_index),
    "nonlinear_bases_only_oof": rmsle(nlb_oof, all_index),
    "nonlinear_bases_plus_features_oof": rmsle(nl_oof, all_index),
}

# Best linear blend of the ridge with each nonlinear stacker, also out of fold.
for tag, vector in (("bases_only", nlb_oof), ("bases_plus_features", nl_oof)):
    blend_design = np.column_stack([stack_oof, vector, np.ones(n)])
    blend_oof = np.zeros(n)
    for fold in range(FOLDS):
        score_index = np.flatnonzero(fold_of == fold)
        fit_index = np.flatnonzero(fold_of != fold)
        x = blend_design[fit_index]
        coef = np.linalg.solve(
            x.T @ x / len(fit_index) + np.eye(3) * 1e-9,
            x.T @ truth[fit_index] / len(fit_index),
        )
        blend_oof[score_index] = blend_design[score_index] @ coef
    report[f"blend_ridge_plus_{tag}_oof"] = rmsle(blend_oof, all_index)
    report[f"correlation_ridge_vs_{tag}"] = float(
        np.corrcoef(stack_oof, vector)[0, 1]
    )

report["gain_blend_bases_only"] = (
    report["linear_ridge_oof"] - report["blend_ridge_plus_bases_only_oof"]
)
report["gain_blend_bases_plus_features"] = (
    report["linear_ridge_oof"] - report["blend_ridge_plus_bases_plus_features_oof"]
)
(OUT / "nonlinear_stack_fold378.json").write_text(json.dumps(report, indent=2) + "\n")
np.save(OUT / "oof378_ridge.npy", stack_oof.astype(np.float32))
np.save(OUT / "oof378_nl_bases.npy", nlb_oof.astype(np.float32))
np.save(OUT / "oof378_nl_full.npy", nl_oof.astype(np.float32))
print(json.dumps(report, indent=2))
