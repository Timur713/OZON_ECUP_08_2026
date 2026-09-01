#!/usr/bin/env python
"""Out-of-time residual correction for the multi-anchor growth model.

The correction available for a fold uses only residuals from earlier folds.
Configuration selection is performed on forward fold 342 by default; fold 378
remains an untouched temporal holdout.  Calibration uses the fixed 20%
customer split and scoring uses the independent 80%.  The branch is accepted
only if the selection fold does not regress and holdout RMSLE improves by at
least 0.00015.  No leaderboard result is read and no CSV is produced.
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
BASE_TAG = os.environ.get("ECUP_BASE_TAG", "multi_anchor_growth")
TAG = os.environ.get("ECUP_TAG", "residual_growth")
FINAL_ANCHOR = 408
NUSERS = int(os.environ.get("ECUP_NUSERS", "250000"))
NUM_THREADS = int(os.environ.get("ECUP_LGB_THREADS", "20"))
MAX_BOOST = int(os.environ.get("ECUP_MAX_BOOST", "200"))
CHECKPOINTS = tuple(value for value in (50, 100, 200) if value <= MAX_BOOST)
SHRINKAGES = (0.10, 0.25, 0.50, 0.75, 1.00)
MIN_HOLDOUT_GAIN = 0.00015
VALIDATION_ONLY = os.environ.get("ECUP_VALIDATION_ONLY", "0") == "1"


def calibrated_score(
    prediction: np.ndarray,
    truth: np.ndarray,
    public: np.ndarray,
    private: np.ndarray,
) -> tuple[float, float, float]:
    design = np.column_stack([prediction[public], np.ones(len(public))])
    slope, intercept = np.linalg.lstsq(
        design, truth[public], rcond=None
    )[0]
    calibrated = np.clip(prediction[private] * slope + intercept, 0, None)
    score = float(np.sqrt(np.mean((truth[private] - calibrated) ** 2)))
    return score, float(slope), float(intercept)


def residual_history(residuals: np.ndarray, index: int) -> np.ndarray:
    """Features known before predicting residuals at ``index``."""
    past = residuals[:index]
    if len(past) == 0:
        raise ValueError("residual history requires at least one earlier fold")
    last = past[-1]
    mean = past.mean(axis=0)
    median = np.median(past, axis=0)
    absolute_mean = np.abs(past).mean(axis=0)
    if len(past) > 1:
        trend = past[-1] - past[-2]
    else:
        trend = np.zeros_like(last)
    return np.column_stack(
        [last, mean, median, absolute_mean, last - mean, trend]
    ).astype(np.float32)


def design_for_fold(
    features: np.ndarray,
    base_predictions: np.ndarray,
    residuals: np.ndarray,
    index: int,
) -> np.ndarray:
    history = residual_history(residuals, index)
    return np.ascontiguousarray(
        np.column_stack(
            [features[index], base_predictions[index].astype(np.float32), history]
        ),
        dtype=np.float32,
    )


def train_correction(
    features: np.ndarray,
    base_predictions: np.ndarray,
    residuals: np.ndarray,
    target_indices: list[int],
    seed: int,
) -> lgb.Booster:
    x = np.vstack(
        [design_for_fold(features, base_predictions, residuals, i) for i in target_indices]
    )
    y = np.concatenate([residuals[i] for i in target_indices]).astype(np.float32)
    min_leaf = min(1000, max(20, len(y) // 200))
    params = {
        "objective": "regression",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": min_leaf,
        "feature_fraction": 0.65,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l1": 5.0,
        "lambda_l2": 50.0,
        "max_bin": 127,
        "num_threads": NUM_THREADS,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "verbosity": -1,
    }
    model = lgb.train(params, lgb.Dataset(x, y), num_boost_round=MAX_BOOST)
    del x, y
    gc.collect()
    return model


def main() -> None:
    started = time.time()
    oof_file = WORK / f"{BASE_TAG}_oof.npz"
    final_file = WORK / f"{BASE_TAG}_final.npy"
    with np.load(oof_file) as archive:
        folds = archive["folds"].astype(int).tolist()
        base_predictions = archive["predictions"].astype(np.float32)
        direct_predictions = archive["direct_predictions"].astype(np.float32)
        hurdle_predictions = archive["hurdle_predictions"].astype(np.float32)
        public = archive["public_users"].astype(int)
    if len(folds) < 4:
        raise ValueError("at least four temporal folds are required")
    if base_predictions.shape != (len(folds), NUSERS):
        raise ValueError(
            f"unexpected OOF shape {base_predictions.shape}; "
            f"expected {(len(folds), NUSERS)}"
        )
    if not VALIDATION_ONLY:
        final_base = np.load(final_file).astype(np.float32)
        with np.load(WORK / f"{BASE_TAG}_final_components.npz") as archive:
            final_direct = archive["direct"].astype(np.float32)
            final_hurdle = archive["hurdle"].astype(np.float32)
    truth = np.log1p(feats4.targets(folds)).reshape(len(folds), NUSERS).astype(np.float32)
    residuals = truth - base_predictions
    private_mask = np.ones(NUSERS, dtype=bool)
    private_mask[public] = False
    private = np.flatnonzero(private_mask)

    feature_anchors = folds if VALIDATION_ONLY else folds + [FINAL_ANCHOR]
    print(f"building residual features anchors={feature_anchors}", flush=True)
    feature_matrix, feature_names = feats4.build(feature_anchors, verbose=True)
    feature_matrix = feature_matrix.reshape(len(feature_anchors), NUSERS, -1)
    fold_feature_matrix = feature_matrix if VALIDATION_ONLY else feature_matrix[:-1]
    features = np.concatenate(
        [
            fold_feature_matrix,
            direct_predictions[..., None],
            hurdle_predictions[..., None],
            (direct_predictions - hurdle_predictions)[..., None],
        ],
        axis=2,
    )
    if not VALIDATION_ONLY:
        final_features = np.column_stack(
            [
                feature_matrix[-1],
                final_direct,
                final_hurdle,
                final_direct - final_hurdle,
            ]
        ).astype(np.float32)
    del feature_matrix
    gc.collect()

    evaluation_indices = list(range(2, len(folds)))
    selection_indices = evaluation_indices[:-1]
    holdout_index = evaluation_indices[-1]
    if not selection_indices:
        raise ValueError("residual selection requires a fold before the holdout")
    base_rows = {}
    correction_predictions: dict[int, dict[int, np.ndarray]] = {}
    for index in evaluation_indices:
        base_score, slope, intercept = calibrated_score(
            base_predictions[index], truth[index], public, private
        )
        base_rows[index] = {
            "fold": folds[index],
            "score": base_score,
            "slope": slope,
            "intercept": intercept,
        }
        # Row i predicts residual_i from residual history strictly before i.
        training_indices = list(range(1, index))
        model = train_correction(
            features,
            base_predictions,
            residuals,
            training_indices,
            seed=30_000 + folds[index],
        )
        x_eval = design_for_fold(features, base_predictions, residuals, index)
        correction_predictions[index] = {
            checkpoint: model.predict(x_eval, num_iteration=checkpoint).astype(np.float32)
            for checkpoint in CHECKPOINTS
        }
        print(
            f"fold={folds[index]} train_residual_folds="
            f"{[folds[i] for i in training_indices]} base={base_score:.6f} "
            f"elapsed={time.time()-started:.0f}s",
            flush=True,
        )
        del model, x_eval
        gc.collect()

    candidates = []
    # A transparent persistence baseline checks whether a complex learner is
    # actually necessary.
    for shrinkage in SHRINKAGES:
        rows = []
        for index in evaluation_indices:
            prediction = base_predictions[index] + shrinkage * residuals[index - 1]
            score, slope, intercept = calibrated_score(
                prediction, truth[index], public, private
            )
            rows.append({
                "fold": folds[index],
                "score": score,
                "gain_vs_base": base_rows[index]["score"] - score,
                "slope": slope,
                "intercept": intercept,
            })
        candidates.append({
            "kind": "persistence",
            "iteration": None,
            "shrinkage": shrinkage,
            "folds": rows,
        })
    for checkpoint in CHECKPOINTS:
        for shrinkage in SHRINKAGES:
            rows = []
            for index in evaluation_indices:
                prediction = (
                    base_predictions[index]
                    + shrinkage * correction_predictions[index][checkpoint]
                )
                score, slope, intercept = calibrated_score(
                    prediction, truth[index], public, private
                )
                rows.append({
                    "fold": folds[index],
                    "score": score,
                    "gain_vs_base": base_rows[index]["score"] - score,
                    "slope": slope,
                    "intercept": intercept,
                })
            candidates.append({
                "kind": "lightgbm",
                "iteration": checkpoint,
                "shrinkage": shrinkage,
                "folds": rows,
            })

    for candidate in candidates:
        scores = np.asarray([row["score"] for row in candidate["folds"]])
        gains = np.asarray([row["gain_vs_base"] for row in candidate["folds"]])
        candidate["mean_score"] = float(scores.mean())
        candidate["worst_score"] = float(scores.max())
        candidate["mean_gain_vs_base"] = float(gains.mean())
        candidate["worst_gain_vs_base"] = float(gains.min())
        selection_rows = [
            row for row in candidate["folds"]
            if row["fold"] in {folds[i] for i in selection_indices}
        ]
        candidate["selection_objective"] = float(
            np.mean([row["score"] for row in selection_rows])
        )
    selected = min(candidates, key=lambda row: row["selection_objective"])
    selected_by_fold = {row["fold"]: row for row in selected["folds"]}
    selection_gains = [
        selected_by_fold[folds[i]]["gain_vs_base"] for i in selection_indices
    ]
    holdout_gain = selected_by_fold[folds[holdout_index]]["gain_vs_base"]
    accepted = bool(
        min(selection_gains) >= 0.0
        and holdout_gain >= MIN_HOLDOUT_GAIN
    )

    if selected["kind"] == "lightgbm":
        holdout_correction = correction_predictions[holdout_index][
            selected["iteration"]
        ]
    else:
        holdout_correction = residuals[holdout_index - 1]
    holdout_prediction = (
        base_predictions[holdout_index]
        + selected["shrinkage"] * holdout_correction
    ).astype(np.float64)
    validation_file = WORK / f"{TAG}_val.npy"
    np.save(validation_file, holdout_prediction)

    candidate_file = None
    if not VALIDATION_ONLY:
        # Train one forward-valid correction for the real forecast. Saving the
        # vector does not make it a submission candidate unless the gate passes.
        final_training_indices = list(range(1, len(folds)))
        if selected["kind"] == "lightgbm":
            final_model = train_correction(
                features,
                base_predictions,
                residuals,
                final_training_indices,
                seed=40_408,
            )
            final_history = residual_history(residuals, len(folds))
            final_design = np.ascontiguousarray(
                np.column_stack([final_features, final_base, final_history]),
                dtype=np.float32,
            )
            final_correction = final_model.predict(
                final_design, num_iteration=selected["iteration"]
            ).astype(np.float32)
            del final_model, final_design
        else:
            final_correction = residuals[-1]
        final_prediction = (
            final_base + selected["shrinkage"] * final_correction
        ).astype(np.float64)
        candidate_file = WORK / f"{TAG}_candidate_final.npy"
        np.save(candidate_file, final_prediction)

    report = {
        "tag": TAG,
        "base_tag": BASE_TAG,
        "validation_only": VALIDATION_ONLY,
        "uses_public_scores": False,
        "uses_recovered_moments": False,
        "folds": folds,
        "evaluation_folds": [folds[i] for i in evaluation_indices],
        "selection_folds": [folds[i] for i in selection_indices],
        "untouched_holdout_fold": folds[holdout_index],
        "protocol": "train residual correction only on strictly earlier temporal folds",
        "features": len(feature_names) + 10,
        "head_features": ["direct", "hurdle", "direct_minus_hurdle"],
        "residual_history_features": [
            "last", "mean", "median", "absolute_mean", "last_minus_mean", "trend"
        ],
        "base_scores": list(base_rows.values()),
        "selection_grid": {
            "iterations": list(CHECKPOINTS),
            "shrinkages": list(SHRINKAGES),
            "minimum_untouched_holdout_gain": MIN_HOLDOUT_GAIN,
            "requires_selection_folds_nonnegative": True,
        },
        "selected": selected,
        "accepted": accepted,
        "validation_file": str(validation_file),
        "candidate_file": str(candidate_file) if candidate_file else None,
        "elapsed_seconds": time.time() - started,
    }
    report_file = WORK / f"{TAG}_report.json"
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
