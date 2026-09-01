#!/usr/bin/env python
"""154 — surrogate betas and ranking over the extended column pool.

Same procedure as 150: one surrogate stack per fold, trained on the eight
anchors ending 30 days before the fold, affine-recalibrated out of fold, then
beta fitted on the residual.  Only the column definition changes.
Frozen key: work/154_extended_pool_preregister.json.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

import histfeat2

OUT = Path(__file__).resolve().parent
MAT = Path(histfeat2.MAT)
FOLDS = [288, 318, 348, 378]
TRAIN_ANCHORS = 8
ANCHOR_STEP = 12
HORIZON = 30
CROSS_FOLDS = 5
SEED = 20260828

GMV = np.load(MAT / "gmv.npy", mmap_mode="r")
PARAMS = dict(
    objective="regression", metric="l2", learning_rate=0.05, num_leaves=127,
    min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8,
    bagging_freq=1, lambda_l2=10.0, num_threads=10, verbose=-1, max_bin=127,
)
ROUNDS = 300


def target_at(anchor):
    window = GMV[:, anchor + 1:anchor + 1 + HORIZON]
    return np.log1p(window.sum(axis=1, dtype=np.float64)).astype(np.float32)


start = time.time()
names = None
for fold in FOLDS:
    anchors = [fold - HORIZON - ANCHOR_STEP * i for i in range(TRAIN_ANCHORS)][::-1]
    frames = histfeat2.build(anchors + [fold])
    names = frames[fold][1]
    train_x = np.vstack([frames[a][0] for a in anchors])
    train_y = np.concatenate([target_at(a) for a in anchors]).astype(np.float64)
    for a in anchors:
        del frames[a]
    gc.collect()
    model = lgb.train(
        PARAMS, lgb.Dataset(train_x, train_y, feature_name=names),
        num_boost_round=ROUNDS,
    )
    del train_x, train_y
    gc.collect()

    fold_x = frames[fold][0]
    truth = target_at(fold).astype(np.float64)
    raw = model.predict(fold_x).astype(np.float64)
    standard = ((fold_x - fold_x.mean(0)) / (fold_x.std(0) + 1e-9)).astype(np.float64)
    n = len(truth)
    fold_of = np.random.default_rng(SEED).permutation(n) % CROSS_FOLDS
    design = np.column_stack([raw, np.ones(n)])
    affine = np.zeros(n)
    for k in range(CROSS_FOLDS):
        score_index = np.flatnonzero(fold_of == k)
        fit_index = np.flatnonzero(fold_of != k)
        x = design[fit_index]
        coef = np.linalg.solve(
            x.T @ x / len(fit_index) + np.eye(2) * 1e-9,
            x.T @ truth[fit_index] / len(fit_index),
        )
        affine[score_index] = design[score_index] @ coef
    beta = np.linalg.solve(
        standard.T @ standard / n + np.eye(standard.shape[1]) * 1e-6,
        standard.T @ (truth - affine) / n,
    )
    np.save(OUT / f"state2_{fold}_beta.npy", beta)
    residual = truth - np.clip(affine, 0, None)
    print(f"fold {fold}: affine {np.sqrt(np.mean(residual ** 2)):.6f} "
          f"[{time.time() - start:.0f}s]", flush=True)
    del fold_x, standard, frames, raw, affine, truth
    gc.collect()

(OUT / "hist2_keys.json").write_text(
    "[\n" + ",\n".join(f'  "{k}"' for k in names) + "\n]\n"
)
for anchor in (378, 408):
    frames = histfeat2.build([anchor])
    np.save(OUT / f"hist2_{anchor}.npy", frames[anchor][0])
    del frames
    gc.collect()
    print("materialised anchor", anchor, flush=True)
print("DONE", flush=True)
