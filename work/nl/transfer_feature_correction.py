#!/usr/bin/env python
"""150 — does the feature-conditional stack correction transfer across folds?

Frozen reading key: work/150_feature_conditional_preregister.json.
No leaderboard score is read anywhere in this file.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[2]))
MAT = ROOT / "work" / "mat"
OUT = Path(os.environ.get("ECUP_OUT", ROOT / "work" / "nl"))
OUT.mkdir(parents=True, exist_ok=True)

FOLDS = [288, 318, 348, 378]
TRAIN_ANCHORS = 8
ANCHOR_STEP = 12
HORIZON = 30
CROSS_FOLDS = 5
SEED = 20260828
WINDOWS = (7, 14, 30, 60, 90, 180, 365)
CHANNELS = (("gmv", True), ("to_ord", False), ("searches", False),
            ("to_cart", False), ("active", False))
RECENCY_CHANNELS = ("gmv", "to_ord", "active")

cumulative = {}
presence_cumulative = {}
last_seen = {}
first_seen = {}
for name, _ in CHANNELS:
    matrix = np.load(MAT / f"{name}.npy", mmap_mode="r")
    users, days = matrix.shape
    block = np.zeros((users, days + 1), dtype=np.float32)
    np.cumsum(matrix, axis=1, dtype=np.float32, out=block[:, 1:])
    cumulative[name] = block
    positive = np.asarray(matrix, dtype=np.float32) > 0
    block = np.zeros((users, days + 1), dtype=np.float32)
    np.cumsum(positive, axis=1, dtype=np.float32, out=block[:, 1:])
    presence_cumulative[name] = block
    if name in RECENCY_CHANNELS:
        index = np.arange(days, dtype=np.float32)
        last_seen[name] = np.maximum.accumulate(
            np.where(positive, index, -1.0), axis=1
        ).astype(np.float32)
        first = np.where(positive, index, np.inf)
        first_seen[name] = np.minimum.accumulate(first, axis=1).astype(np.float32)
    del matrix, positive
NUSERS = users
GMV = np.load(MAT / "gmv.npy", mmap_mode="r")

FEATURE_NAMES = []
for name, _ in CHANNELS:
    FEATURE_NAMES += [f"{name}_s{w}" for w in WINDOWS]
    FEATURE_NAMES += [f"{name}_d{w}" for w in WINDOWS]
    if name in RECENCY_CHANNELS:
        FEATURE_NAMES += [f"{name}_recency", f"{name}_age"]


def features_at(anchor):
    blocks = []
    for name, log_scale in CHANNELS:
        block = cumulative[name]
        presence = presence_cumulative[name]
        for w in WINDOWS:
            value = block[:, anchor + 1] - block[:, max(0, anchor - w + 1)]
            blocks.append(np.log1p(value) if log_scale else value)
        for w in WINDOWS:
            blocks.append(presence[:, anchor + 1] - presence[:, max(0, anchor - w + 1)])
        if name in RECENCY_CHANNELS:
            last = last_seen[name][:, anchor]
            first = first_seen[name][:, anchor]
            blocks.append(np.where(last < 0, 999.0, anchor - last))
            blocks.append(np.where(np.isfinite(first), anchor - first, 999.0))
    return np.column_stack(blocks).astype(np.float32)


def target_at(anchor):
    window = GMV[:, anchor + 1:anchor + 1 + HORIZON]
    return np.log1p(window.sum(axis=1, dtype=np.float64)).astype(np.float32)


PARAMS = dict(
    objective="regression", metric="l2", learning_rate=0.05, num_leaves=127,
    min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8,
    bagging_freq=1, lambda_l2=10.0, num_threads=10, verbose=-1, max_bin=127,
)
ROUNDS = 300

rng = np.random.default_rng(SEED)
fold_of = rng.permutation(NUSERS) % CROSS_FOLDS


def rmsle(prediction, truth):
    residual = truth - np.clip(prediction, 0.0, None)
    return float(np.sqrt(np.mean(residual * residual)))


def out_of_fold_linear(design, truth, lam=1e-6):
    prediction = np.zeros(len(truth))
    penalty = np.eye(design.shape[1]) * lam
    penalty[-1, -1] = 0
    for k in range(CROSS_FOLDS):
        score_index = np.flatnonzero(fold_of == k)
        fit_index = np.flatnonzero(fold_of != k)
        x = design[fit_index]
        coef = np.linalg.solve(
            x.T @ x / len(fit_index) + penalty,
            x.T @ truth[fit_index] / len(fit_index),
        )
        prediction[score_index] = design[score_index] @ coef
    return prediction


state = {}
start = time.time()
for fold in FOLDS:
    anchors = [fold - HORIZON - ANCHOR_STEP * i for i in range(TRAIN_ANCHORS)][::-1]
    train_x = np.vstack([features_at(a) for a in anchors])
    train_y = np.concatenate([target_at(a) for a in anchors]).astype(np.float64)
    model = lgb.train(
        PARAMS,
        lgb.Dataset(train_x, train_y, feature_name=FEATURE_NAMES),
        num_boost_round=ROUNDS,
    )
    del train_x, train_y
    fold_x = features_at(fold)
    fold_truth = target_at(fold).astype(np.float64)
    raw = model.predict(fold_x).astype(np.float64)
    standard = ((fold_x - fold_x.mean(0)) / (fold_x.std(0) + 1e-9)).astype(np.float64)

    affine_design = np.column_stack([raw, np.ones(NUSERS)])
    affine = out_of_fold_linear(affine_design, fold_truth)
    residual = fold_truth - affine
    beta = np.linalg.solve(
        standard.T @ standard / NUSERS + np.eye(standard.shape[1]) * 1e-6,
        standard.T @ residual / NUSERS,
    )
    state[fold] = dict(
        standard=standard, truth=fold_truth, affine=affine, beta=beta,
        raw_rmsle=rmsle(raw, fold_truth), affine_rmsle=rmsle(affine, fold_truth),
        anchors=anchors,
    )
    np.save(OUT / f"state_{fold}_standard.npy", standard.astype(np.float32))
    np.save(OUT / f"state_{fold}_truth.npy", fold_truth.astype(np.float32))
    np.save(OUT / f"state_{fold}_affine.npy", affine.astype(np.float32))
    np.save(OUT / f"state_{fold}_beta.npy", beta)
    print(f"fold {fold}: raw {state[fold]['raw_rmsle']:.6f} "
          f"affine {state[fold]['affine_rmsle']:.6f} [{time.time() - start:.0f}s]",
          flush=True)
    del fold_x

random_beta = rng.normal(size=len(FEATURE_NAMES))
random_beta *= np.linalg.norm(state[FOLDS[0]]["beta"]) / np.linalg.norm(random_beta)

pairs = {}
for source in FOLDS:
    for target in FOLDS:
        direction = state[target]["standard"] @ state[source]["beta"]
        design = np.column_stack([
            state[target]["affine"], direction, np.ones(NUSERS)
        ])
        corrected = out_of_fold_linear(design, state[target]["truth"])
        pairs[f"{source}->{target}"] = (
            state[target]["affine_rmsle"] - rmsle(corrected, state[target]["truth"])
        )

control = {}
for target in FOLDS:
    direction = state[target]["standard"] @ random_beta
    design = np.column_stack([state[target]["affine"], direction, np.ones(NUSERS)])
    corrected = out_of_fold_linear(design, state[target]["truth"])
    control[str(target)] = (
        state[target]["affine_rmsle"] - rmsle(corrected, state[target]["truth"])
    )

cross = {k: v for k, v in pairs.items() if k.split("->")[0] != k.split("->")[1]}
self_gain = {k: v for k, v in pairs.items() if k.split("->")[0] == k.split("->")[1]}
mean_cross = float(np.mean(list(cross.values())))
positive = int(sum(v > 0 for v in cross.values()))
correlation = {}
for i, a in enumerate(FOLDS):
    for b in FOLDS[i + 1:]:
        correlation[f"{a}~{b}"] = float(
            np.corrcoef(state[a]["beta"], state[b]["beta"])[0, 1]
        )

verdict = (
    "transfers" if mean_cross >= 0.00030 and positive >= 9 else
    "closed" if mean_cross < 0.00008 or positive < 9 else
    "diagnostic"
)
report = {
    "tag": "150_feature_conditional_correction",
    "folds": FOLDS,
    "raw_rmsle": {str(f): state[f]["raw_rmsle"] for f in FOLDS},
    "affine_rmsle": {str(f): state[f]["affine_rmsle"] for f in FOLDS},
    "self_gain_positive_control": self_gain,
    "cross_fold_gain": cross,
    "random_direction_negative_control": control,
    "mean_cross_fold_gain": mean_cross,
    "positive_cross_pairs": positive,
    "beta_correlation_between_folds": correlation,
    "verdict": verdict,
}
(OUT / "150_feature_conditional_transfer.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
