#!/usr/bin/env python
"""Pair- and user-safe adjacent-window residual-correlation diagnostic.

For every adjacent 30-day target-window pair, both windows are excluded from
training labels.  Two-fold user cross-fitting prevents a customer's labels or
target-window calibration row from influencing that customer's residuals.
This is a leaderboard-free diagnostic and does not produce a forecast or CSV.
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

import feats4


ROOT = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get("ECUP_OUT", ROOT / "work"))
TAG = os.environ.get("ECUP_TAG", "residcurve_pairsafe")
NUSERS = int(os.environ.get("ECUP_NUSERS", "250000"))
TRAIN_FRAC = float(os.environ.get("ECUP_TRAIN_FRAC", "0.30"))
MAX_ROUNDS = int(os.environ.get("ECUP_MAX_ROUNDS", "300"))
MAX_PAIRS = int(os.environ.get("ECUP_MAX_PAIRS", "0"))
THREADS = int(os.environ.get("ECUP_LGB_THREADS", "24"))
SEED = 20260825
TRAIN_ANCHORS = tuple(range(120, 349, 12))
EVAL_ANCHORS = tuple(range(138, 379, 30))
PAIRS = tuple(zip(EVAL_ANCHORS[:-1], EVAL_ANCHORS[1:]))


def overlaps(left_anchor, right_anchor):
    left = (left_anchor + 1, left_anchor + 30)
    right = (right_anchor + 1, right_anchor + 30)
    return not (left[1] < right[0] or right[1] < left[0])


def affine_residual(prediction, truth, calibration, evaluation):
    design = np.column_stack([
        prediction[calibration], np.ones(len(calibration))
    ])
    slope, intercept = np.linalg.lstsq(
        design, truth[calibration], rcond=None
    )[0]
    calibrated = np.clip(prediction[evaluation] * slope + intercept, 0, None)
    return truth[evaluation] - calibrated, float(slope), float(intercept)


def correlation(left, right):
    return float(np.corrcoef(left, right)[0, 1])


def block_correlations(left, right, block_ids, blocks):
    values = np.empty(blocks, dtype=np.float64)
    for block in range(blocks):
        index = block_ids == block
        values[block] = correlation(left[index], right[index])
    return values


def main():
    if not 0 < TRAIN_FRAC <= 1:
        raise ValueError("ECUP_TRAIN_FRAC must be in (0, 1]")
    WORK.mkdir(parents=True, exist_ok=True)
    started = time.time()
    pairs = PAIRS[:MAX_PAIRS] if MAX_PAIRS else PAIRS
    required = sorted(set(TRAIN_ANCHORS) | {a for pair in pairs for a in pair})
    print(f"building feats4 anchors={required}", flush=True)
    matrix, feature_names = feats4.build(required, verbose=True)
    matrix = matrix.reshape(len(required), NUSERS, -1)
    anchor_row = {anchor: row for row, anchor in enumerate(required)}
    target_matrix = np.log1p(feats4.targets(required)).reshape(
        len(required), NUSERS
    ).astype(np.float32)
    truth = {
        anchor: target_matrix[anchor_row[anchor]].astype(np.float64)
        for anchor in {a for pair in pairs for a in pair}
    }

    rng = np.random.default_rng(SEED)
    user_fold = np.empty(NUSERS, dtype=np.int8)
    user_fold[rng.permutation(NUSERS)] = np.arange(NUSERS) % 2
    residual_pairs = []
    rows = []
    params = {
        "objective": "regression",
        "learning_rate": 0.04,
        "num_leaves": 127,
        "min_data_in_leaf": 250,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 15.0,
        "num_threads": THREADS,
        "verbosity": -1,
        "max_bin": 255,
    }

    for pair_index, pair in enumerate(pairs):
        training_anchors = [
            anchor for anchor in TRAIN_ANCHORS
            if not any(overlaps(anchor, heldout) for heldout in pair)
        ]
        pair_residuals = [np.empty(NUSERS, dtype=np.float32) for _ in pair]
        fold_rows = []
        for evaluation_fold in (0, 1):
            evaluation = np.flatnonzero(user_fold == evaluation_fold)
            calibration = np.flatnonzero(user_fold != evaluation_fold)
            sample_size = max(1, int(round(len(calibration) * TRAIN_FRAC)))
            sample_rng = np.random.default_rng(
                SEED + 10_000 * pair_index + evaluation_fold
            )
            sampled_users = np.sort(
                sample_rng.choice(calibration, sample_size, replace=False)
            )
            train_rows = [anchor_row[anchor] for anchor in training_anchors]
            x_train = np.ascontiguousarray(
                matrix[train_rows][:, sampled_users].reshape(
                    -1, matrix.shape[-1]
                )
            )
            y_train = np.concatenate([
                target_matrix[anchor_row[anchor], sampled_users]
                for anchor in training_anchors
            ]).astype(np.float32)
            model = lgb.train(
                params | {
                    "seed": SEED + 100 * pair_index + evaluation_fold,
                    "bagging_seed": SEED + 100 * pair_index + evaluation_fold,
                    "feature_fraction_seed": (
                        SEED + 100 * pair_index + evaluation_fold
                    ),
                },
                lgb.Dataset(x_train, y_train, feature_name=feature_names),
                num_boost_round=MAX_ROUNDS,
            )
            calibration_rows = []
            for side, anchor in enumerate(pair):
                x_eval = np.ascontiguousarray(matrix[anchor_row[anchor]])
                prediction = model.predict(x_eval).astype(np.float64)
                residual, slope, intercept = affine_residual(
                    prediction, truth[anchor], calibration, evaluation
                )
                pair_residuals[side][evaluation] = residual.astype(np.float32)
                calibration_rows.append({
                    "anchor": anchor,
                    "slope": slope,
                    "intercept": intercept,
                })
                del x_eval, prediction, residual
            fold_rows.append({
                "evaluation_fold": evaluation_fold,
                "training_users": len(sampled_users),
                "calibration_users": len(calibration),
                "evaluation_users": len(evaluation),
                "calibration": calibration_rows,
            })
            del x_train, y_train, model
            gc.collect()

        observed = correlation(*pair_residuals)
        residual_pairs.append(np.stack(pair_residuals))
        rows.append({
            "anchors": list(pair),
            "training_anchors": training_anchors,
            "correlation": observed,
            "folds": fold_rows,
        })
        print(
            f"pair={pair} corr={observed:+.6f} "
            f"elapsed={time.time()-started:.0f}s",
            flush=True,
        )

    residual_pairs = np.stack(residual_pairs).astype(np.float32)
    np.save(WORK / f"{TAG}_residual_pairs.npy", residual_pairs)
    observed_values = np.asarray([row["correlation"] for row in rows])

    # Deterministic controls.  The negative control destroys user alignment;
    # the positive control injects a stable component targeting corr~=0.01.
    control_rng = np.random.default_rng(SEED + 77)
    negative_values = []
    positive_values = []
    for pair_residual in residual_pairs:
        left = pair_residual[0].astype(np.float64)
        right = pair_residual[1].astype(np.float64)
        shuffled = right[control_rng.permutation(NUSERS)]
        negative_values.append(correlation(left, shuffled))
        shared = control_rng.standard_normal(NUSERS)
        covariance_scale = np.sqrt(
            0.01 * np.sqrt(np.var(left) * np.var(shuffled)) / 0.99
        )
        positive_values.append(correlation(
            left + covariance_scale * shared,
            shuffled + covariance_scale * shared,
        ))

    blocks = min(200, max(10, NUSERS // 500))
    block_ids = np.empty(NUSERS, dtype=np.int16)
    block_ids[control_rng.permutation(NUSERS)] = np.arange(NUSERS) % blocks
    block_matrix = np.stack([
        block_correlations(pair[0], pair[1], block_ids, blocks)
        for pair in residual_pairs
    ])
    bootstrap_rng = np.random.default_rng(SEED + 99)
    draws = 5000
    bootstrap = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected_pairs = bootstrap_rng.integers(0, len(pairs), len(pairs))
        selected_blocks = bootstrap_rng.integers(0, blocks, blocks)
        bootstrap[draw] = block_matrix[
            selected_pairs[:, None], selected_blocks[None, :]
        ].mean()

    report = {
        "tag": TAG,
        "uses_public_scores": False,
        "uses_recovered_moments": False,
        "uses_competition_target_mean": False,
        "pair_safe": True,
        "user_crossfit_folds": 2,
        "training_user_fraction_within_calibration_half": TRAIN_FRAC,
        "features": len(feature_names),
        "boost_rounds": MAX_ROUNDS,
        "pairs": rows,
        "mean_adjacent_residual_correlation": float(observed_values.mean()),
        "pair_positive_count": int(np.count_nonzero(observed_values > 0)),
        "pair_count": len(observed_values),
        "pair_standard_deviation": (
            float(observed_values.std(ddof=1)) if len(observed_values) > 1 else 0.0
        ),
        "two_way_block_bootstrap_p05": float(np.quantile(bootstrap, 0.05)),
        "two_way_block_bootstrap_p50": float(np.quantile(bootstrap, 0.50)),
        "two_way_block_bootstrap_p95": float(np.quantile(bootstrap, 0.95)),
        "negative_control_mean": float(np.mean(negative_values)),
        "positive_control_mean": float(np.mean(positive_values)),
        "control_thresholds": {
            "maximum_absolute_negative": 0.003,
            "minimum_positive": 0.006,
            "maximum_positive": 0.014,
        },
        "decision_threshold": {
            "minimum_mean_correlation": 0.003,
            "minimum_positive_pairs": max(1, int(np.ceil(0.75 * len(pairs)))),
            "minimum_bootstrap_p05": 0.0,
        },
        "elapsed_seconds": time.time() - started,
    }
    control_thresholds = report["control_thresholds"]
    report["controls_valid"] = bool(
        abs(report["negative_control_mean"])
        <= control_thresholds["maximum_absolute_negative"]
        and control_thresholds["minimum_positive"]
        <= report["positive_control_mean"]
        <= control_thresholds["maximum_positive"]
    )
    threshold = report["decision_threshold"]
    report["supports_stable_user_residual_branch"] = bool(
        report["controls_valid"]
        and report["mean_adjacent_residual_correlation"]
        >= threshold["minimum_mean_correlation"]
        and report["pair_positive_count"] >= threshold["minimum_positive_pairs"]
        and report["two_way_block_bootstrap_p05"]
        > threshold["minimum_bootstrap_p05"]
    )
    (WORK / f"{TAG}_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
