#!/usr/bin/env python
"""Multi-anchor LightGBM with temporal OOF and 50k->200k model selection.

This is a leaderboard-free growth experiment.  Four late temporal folds are
trained only on anchors whose 30-day labels finish before the fold anchor.
Checkpoint and direct/hurdle mix are chosen on the first three folds by mean
independent-200k RMSLE after calibration on a fixed 50k customer subset.  Fold
378 remains an untouched temporal holdout.  The final model is then trained on
every available historical anchor through 378 and predicts anchor 408.  No
public score, recovered target moment, or submission is read.
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
TAG = os.environ.get("ECUP_TAG", "multi_anchor_growth")
FINAL_ANCHOR = 408
ALL_ANCHORS = np.arange(186, 379, 12, dtype=int)
FOLDS = (270, 306, 342, 378)
CHECKPOINTS = (100, 200, 300, 400, 500)
MIXES = (0.0, 0.25, 0.50, 0.75, 1.0)
PUBLIC_USERS = int(os.environ.get("ECUP_PUBLIC_USERS", "50000"))
NUSERS = int(os.environ.get("ECUP_NUSERS", "250000"))
SELECTION_SEED = 20260825
NUM_THREADS = int(os.environ.get("ECUP_LGB_THREADS", "24"))
MAX_FOLDS = int(os.environ.get("ECUP_MAX_FOLDS", str(len(FOLDS))))
MAX_BOOST = int(os.environ.get("ECUP_MAX_BOOST", str(max(CHECKPOINTS))))


BASE_PARAMS = {
    "learning_rate": 0.04,
    "num_leaves": 127,
    "min_data_in_leaf": 250,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 15.0,
    "num_threads": NUM_THREADS,
    "verbosity": -1,
    "max_bin": 255,
    "seed": 2026,
    "bagging_seed": 2026,
    "feature_fraction_seed": 2026,
}


def calibrated_private_score(
    prediction: np.ndarray,
    truth: np.ndarray,
    public: np.ndarray,
    private: np.ndarray,
) -> tuple[float, float, float]:
    design = np.column_stack([prediction[public], np.ones(len(public))])
    coefficients = np.linalg.lstsq(design, truth[public], rcond=None)[0]
    calibrated = np.clip(
        prediction[private] * coefficients[0] + coefficients[1], 0, None
    )
    score = float(np.sqrt(np.mean((truth[private] - calibrated) ** 2)))
    return score, float(coefficients[0]), float(coefficients[1])


def train_heads(
    x: np.ndarray,
    target: np.ndarray,
    binary: np.ndarray,
    feature_names: list[str],
    seed: int,
) -> tuple[lgb.Booster, lgb.Booster, lgb.Booster]:
    params = BASE_PARAMS | {
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
    }
    direct = lgb.train(
        params | {"objective": "regression"},
        lgb.Dataset(x, target, feature_name=feature_names, free_raw_data=False),
        num_boost_round=MAX_BOOST,
    )
    classifier = lgb.train(
        params | {"objective": "binary"},
        lgb.Dataset(x, binary, feature_name=feature_names, free_raw_data=False),
        num_boost_round=MAX_BOOST,
    )
    positive = binary == 1
    magnitude = lgb.train(
        params | {"objective": "regression"},
        lgb.Dataset(
            x[positive], target[positive], feature_name=feature_names
        ),
        num_boost_round=MAX_BOOST,
    )
    return direct, classifier, magnitude


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    started = time.time()
    folds = FOLDS[:MAX_FOLDS]
    if len(folds) < 2:
        raise ValueError("at least two folds are required for a temporal holdout")
    selection_folds = folds[:-1]
    holdout_fold = folds[-1]
    required = sorted(set(ALL_ANCHORS.tolist()) | set(folds) | {FINAL_ANCHOR})
    print(f"building feats4 anchors={required}", flush=True)
    matrix, feature_names = feats4.build(required, verbose=True)
    matrix = matrix.reshape(len(required), NUSERS, -1)
    anchor_to_row = {anchor: index for index, anchor in enumerate(required)}
    fold_truth = {
        anchor: np.log1p(feats4.targets([anchor])).astype(np.float64)
        for anchor in folds
    }
    rng = np.random.default_rng(SELECTION_SEED)
    public_users = min(PUBLIC_USERS, max(1, NUSERS // 5))
    if public_users >= NUSERS:
        raise ValueError("public user split leaves no private users")
    public = np.sort(rng.choice(NUSERS, public_users, replace=False))
    private_mask = np.ones(NUSERS, dtype=bool)
    private_mask[public] = False
    private = np.flatnonzero(private_mask)

    fold_predictions: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    fold_rows = []
    for fold in folds:
        training_anchors = [
            int(anchor) for anchor in ALL_ANCHORS if anchor + 30 <= fold
        ]
        indices = [anchor_to_row[anchor] for anchor in training_anchors]
        x_train = np.ascontiguousarray(matrix[indices].reshape(-1, matrix.shape[-1]))
        raw_target = feats4.targets(training_anchors)
        target = np.log1p(raw_target).astype(np.float32)
        binary = (raw_target > 0).astype(np.int8)
        x_fold = np.ascontiguousarray(matrix[anchor_to_row[fold]])
        print(
            f"fold={fold} train_anchors={training_anchors} X={x_train.shape} "
            f"elapsed={time.time()-started:.0f}s",
            flush=True,
        )
        direct, classifier, magnitude = train_heads(
            x_train, target, binary, feature_names, seed=10_000 + fold
        )
        checkpoints = tuple(value for value in CHECKPOINTS if value <= MAX_BOOST)
        fold_predictions[fold] = {}
        for checkpoint in checkpoints:
            direct_prediction = direct.predict(x_fold, num_iteration=checkpoint)
            hurdle_prediction = (
                classifier.predict(x_fold, num_iteration=checkpoint)
                * magnitude.predict(x_fold, num_iteration=checkpoint)
            )
            fold_predictions[fold][checkpoint] = (
                direct_prediction.astype(np.float32),
                hurdle_prediction.astype(np.float32),
            )
        fold_rows.append({
            "fold": fold,
            "training_anchors": training_anchors,
            "training_rows": len(target),
        })
        del x_train, raw_target, target, binary, x_fold
        del direct, classifier, magnitude
        gc.collect()

    configs = []
    checkpoints = tuple(value for value in CHECKPOINTS if value <= MAX_BOOST)
    for direct_iteration in checkpoints:
        for hurdle_iteration in checkpoints:
            for mix in MIXES:
                rows = []
                for fold in selection_folds:
                    direct_prediction = fold_predictions[fold][direct_iteration][0]
                    hurdle_prediction = fold_predictions[fold][hurdle_iteration][1]
                    prediction = (
                        mix * direct_prediction
                        + (1.0 - mix) * hurdle_prediction
                    )
                    score, slope, intercept = calibrated_private_score(
                        prediction, fold_truth[fold], public, private
                    )
                    rows.append({
                        "fold": fold,
                        "private_score": score,
                        "slope": slope,
                        "intercept": intercept,
                    })
                scores = np.asarray([row["private_score"] for row in rows])
                configs.append({
                    "direct_iteration": direct_iteration,
                    "hurdle_iteration": hurdle_iteration,
                    "direct_mix": mix,
                    "mean_private_score": float(scores.mean()),
                    "worst_private_score": float(scores.max()),
                    "private_score_std": (
                        float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
                    ),
                    "folds": rows,
                })

    # Mean score is primary; a small worst-fold term prevents a configuration
    # from winning by sacrificing one temporal regime.
    for row in configs:
        row["selection_objective"] = (
            row["mean_private_score"]
            + 0.20 * (row["worst_private_score"] - row["mean_private_score"])
        )
    selected = min(configs, key=lambda row: row["selection_objective"])
    print(f"selected={selected}", flush=True)

    # Persist every honestly out-of-time fold, not only the latest one.  A
    # downstream residual learner may train corrections on earlier folds and
    # evaluate them on later folds without ever seeing the evaluation labels.
    selected_oof = []
    selected_direct_oof = []
    selected_hurdle_oof = []
    for fold in folds:
        direct_prediction = fold_predictions[fold][selected["direct_iteration"]][0]
        hurdle_prediction = fold_predictions[fold][selected["hurdle_iteration"]][1]
        selected_direct_oof.append(direct_prediction)
        selected_hurdle_oof.append(hurdle_prediction)
        selected_oof.append(
            selected["direct_mix"] * direct_prediction
            + (1.0 - selected["direct_mix"]) * hurdle_prediction
        )
    oof_file = WORK / f"{TAG}_oof.npz"
    np.savez(
        oof_file,
        folds=np.asarray(folds, dtype=np.int16),
        predictions=np.stack(selected_oof).astype(np.float32),
        direct_predictions=np.stack(selected_direct_oof).astype(np.float32),
        hurdle_predictions=np.stack(selected_hurdle_oof).astype(np.float32),
        public_users=public.astype(np.int32),
    )

    holdout_prediction = selected_oof[-1]
    holdout_score, holdout_slope, holdout_intercept = calibrated_private_score(
        holdout_prediction, fold_truth[holdout_fold], public, private
    )
    untouched_holdout = {
        "fold": holdout_fold,
        "private_score": holdout_score,
        "slope": holdout_slope,
        "intercept": holdout_intercept,
    }
    print(f"untouched_holdout={untouched_holdout}", flush=True)

    # Honest latest-fold vector for downstream joint/private audits.
    latest = folds[-1]
    latest_prediction = selected_oof[-1]
    np.save(
        WORK / f"{TAG}_val.npy", latest_prediction.astype(np.float64)
    )

    # Final training uses only labels fully observed by anchor 408.
    final_training_anchors = [
        int(anchor) for anchor in ALL_ANCHORS
        if anchor + 30 <= FINAL_ANCHOR
    ]
    indices = [anchor_to_row[anchor] for anchor in final_training_anchors]
    x_train = np.ascontiguousarray(matrix[indices].reshape(-1, matrix.shape[-1]))
    raw_target = feats4.targets(final_training_anchors)
    target = np.log1p(raw_target).astype(np.float32)
    binary = (raw_target > 0).astype(np.int8)
    x_final = np.ascontiguousarray(matrix[anchor_to_row[FINAL_ANCHOR]])
    final_seeds = []
    final_direct_seeds = []
    final_hurdle_seeds = []
    for seed in (20261, 20262):
        direct, classifier, magnitude = train_heads(
            x_train, target, binary, feature_names, seed
        )
        direct_prediction = direct.predict(
            x_final, num_iteration=selected["direct_iteration"]
        )
        hurdle_prediction = (
            classifier.predict(x_final, num_iteration=selected["hurdle_iteration"])
            * magnitude.predict(x_final, num_iteration=selected["hurdle_iteration"])
        )
        final_direct_seeds.append(direct_prediction)
        final_hurdle_seeds.append(hurdle_prediction)
        final_seeds.append(
            selected["direct_mix"] * direct_prediction
            + (1.0 - selected["direct_mix"]) * hurdle_prediction
        )
        del direct, classifier, magnitude
        gc.collect()
    final_prediction = np.mean(final_seeds, axis=0).astype(np.float64)
    np.save(WORK / f"{TAG}_final.npy", final_prediction)
    final_components_file = WORK / f"{TAG}_final_components.npz"
    np.savez(
        final_components_file,
        direct=np.mean(final_direct_seeds, axis=0).astype(np.float32),
        hurdle=np.mean(final_hurdle_seeds, axis=0).astype(np.float32),
    )

    report = {
        "tag": TAG,
        "uses_public_scores": False,
        "uses_recovered_moments": False,
        "uses_competition_target_mean": False,
        "all_anchors": ALL_ANCHORS.tolist(),
        "folds": list(folds),
        "selection_folds": list(selection_folds),
        "untouched_holdout": untouched_holdout,
        "fold_protocol": (
            "labels end before fold; calibrate "
            f"{public_users} users, score independent {NUSERS - public_users}"
        ),
        "users": NUSERS,
        "public_users": public_users,
        "private_users": NUSERS - public_users,
        "features": len(feature_names),
        "feature_names": feature_names,
        "selection_grid": {
            "checkpoints": list(checkpoints),
            "direct_mixes": list(MIXES),
            "objective": "mean_private + 0.20 * worst_private",
        },
        "selected": selected,
        "fold_training": fold_rows,
        "final_training_anchors": final_training_anchors,
        "final_seeds": [20261, 20262],
        "validation_file": str(WORK / f"{TAG}_val.npy"),
        "oof_file": str(oof_file),
        "final_file": str(WORK / f"{TAG}_final.npy"),
        "final_components_file": str(final_components_file),
        "elapsed_seconds": time.time() - started,
    }
    (WORK / f"{TAG}_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
