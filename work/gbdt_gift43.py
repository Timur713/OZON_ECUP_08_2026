#!/usr/bin/env python
"""Exact-calendar analogue: Jan 2-Feb 13 history -> Feb 14-Mar 15 GMV."""
import gc
import json
import os
import time

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

ROOT = os.environ.get("ECUP_ROOT", "/Users/timur/Desktop/dev/OZON_ECUP_2026_3")
MAT = os.environ.get("ECUP_MAT", os.path.join(ROOT, "work", "mat"))
OUT = os.environ.get("ECUP_OUT", os.path.join(ROOT, "work"))
TRAIN_ANCHOR = int(os.environ.get("ECUP_TRAIN_ANCHOR", "43"))
FINAL_ANCHOR = int(os.environ.get("ECUP_FINAL_ANCHOR", "408"))
HISTORY_WINDOW = int(os.environ.get("ECUP_HISTORY_WINDOW", "43"))
TAG = os.environ.get("ECUP_TAG", "gbdt_gift43")
WINDOWS = tuple(sorted({min(value, HISTORY_WINDOW) for value in (3, 7, 14, 30, 43)}))
DAY_WINDOWS = tuple(sorted({min(value, HISTORY_WINDOW) for value in (7, 14, 30, 43)}))
MID_WINDOW = min(30, HISTORY_WINDOW)
RANK_WINDOWS = {MID_WINDOW, HISTORY_WINDOW}
CHANNELS = (
    "gmv", "to_ord", "to_cart", "searches", "active", "search", "cat",
    "gmv_search", "gmv_cat", "search_to_ord", "cat_to_ord",
    "has_search_to_ord", "has_search_to_cart", "has_cat_to_ord",
    "has_cat_to_cart", "search_to_cart", "cat_to_cart",
)
RANK_CHANNELS = {
    "gmv", "to_ord", "to_cart", "searches", "active", "search", "cat",
    "gmv_search", "gmv_cat",
}
NUSERS = int(os.environ.get("ECUP_NUSERS", "0"))
MAX_ROUNDS = int(os.environ.get("ECUP_MAX_ROUNDS", "1200"))
EARLY_STOPPING = int(os.environ.get("ECUP_EARLY_STOPPING", "70"))


def recent_slice(matrix, anchor, window):
    return matrix[:, anchor - window + 1:anchor + 1]


def build_features(anchor):
    features = []
    names = []
    raw_sums = {}
    for channel in CHANNELS:
        matrix = np.load(os.path.join(MAT, channel + ".npy"), mmap_mode="r")
        if NUSERS:
            matrix = matrix[:NUSERS]
        for window in WINDOWS:
            values = recent_slice(matrix, anchor, window).sum(axis=1, dtype=np.float32)
            raw_sums[(channel, window)] = values
            features.append(np.log1p(values).astype(np.float32))
            names.append(f"{channel}_sum{window}")
            if channel in RANK_CHANNELS and window in RANK_WINDOWS:
                rank = rankdata(values, method="average").astype(np.float32) / len(values)
                features.append(rank)
                names.append(f"{channel}_rank{window}")
        nonzero = recent_slice(matrix, anchor, HISTORY_WINDOW) > 0
        for window in DAY_WINDOWS:
            features.append(nonzero[:, -window:].mean(axis=1, dtype=np.float32))
            names.append(f"{channel}_freq{window}")
        indices = np.where(
            nonzero,
            np.arange(HISTORY_WINDOW, dtype=np.int16)[None, :],
            -1,
        )
        last = indices.max(axis=1)
        rows = np.arange(len(last))
        indices[rows, np.maximum(last, 0)] = -1
        previous = indices.max(axis=1)
        recency = np.where(
            last >= 0, HISTORY_WINDOW - 1 - last, HISTORY_WINDOW
        ).astype(np.float32) / HISTORY_WINDOW
        gap = np.where(
            previous >= 0, last - previous, HISTORY_WINDOW
        ).astype(np.float32) / HISTORY_WINDOW
        features.extend([recency, gap])
        names.extend([f"{channel}_recency43", f"{channel}_lastgap43"])
        del matrix, nonzero, indices, last, previous
        gc.collect()

    def ratio(numerator, denominator, name, window=MID_WINDOW, cap=20.0):
        value = raw_sums[(numerator, window)] / (raw_sums[(denominator, window)] + 1.0)
        features.append(np.clip(value, 0, cap).astype(np.float32))
        names.append(name)

    for channel in ("gmv", "to_ord", "to_cart", "searches", "active"):
        value = raw_sums[(channel, min(7, HISTORY_WINDOW))] / (
            raw_sums[(channel, MID_WINDOW)] + 1.0
        )
        features.append(np.clip(value, 0, 10).astype(np.float32))
        names.append(f"{channel}_share7_30")
        value = raw_sums[(channel, min(14, HISTORY_WINDOW))] / (
            raw_sums[(channel, HISTORY_WINDOW)] + 1.0
        )
        features.append(np.clip(value, 0, 10).astype(np.float32))
        names.append(f"{channel}_share14_43")
    ratio("to_ord", "to_cart", "conversion30")
    ratio("to_ord", "active", "orders_per_active30")
    ratio("gmv", "to_ord", "aov30", cap=10000.0)
    ratio("gmv_search", "gmv", "search_gmv_share30", cap=2.0)
    ratio("gmv_cat", "gmv", "cat_gmv_share30", cap=2.0)
    ratio("search_to_ord", "search_to_cart", "search_conversion30")
    ratio("cat_to_ord", "cat_to_cart", "cat_conversion30")

    output = np.column_stack(features).astype(np.float32, copy=False)
    assert output.shape[1] == len(names) and np.isfinite(output).all()
    return output, names


def calibrated_score(prediction, truth):
    design = np.column_stack([prediction, np.ones_like(prediction)])
    coefficients = np.linalg.lstsq(design, truth, rcond=None)[0]
    calibrated = np.maximum(design @ coefficients, 0)
    return float(np.mean((truth - calibrated) ** 2) ** 0.5), coefficients


def main():
    os.makedirs(OUT, exist_ok=True)
    started = time.time()
    train, feature_names = build_features(TRAIN_ANCHOR)
    final, final_names = build_features(FINAL_ANCHOR)
    assert feature_names == final_names
    gmv = np.load(os.path.join(MAT, "gmv.npy"), mmap_mode="r")
    if NUSERS:
        gmv = gmv[:NUSERS]
    truth_raw = gmv[:, TRAIN_ANCHOR + 1:TRAIN_ANCHOR + 31].sum(axis=1, dtype=np.float64)
    truth = np.log1p(truth_raw).astype(np.float32)
    binary = (truth_raw > 0).astype(np.int8)
    del gmv, truth_raw

    rng = np.random.default_rng(430043)
    validation_size = min(50_000, max(50, len(truth) // 5))
    validation = rng.choice(len(truth), validation_size, replace=False)
    train_mask = np.ones(len(truth), dtype=bool)
    train_mask[validation] = False
    fit = np.flatnonzero(train_mask)
    positive_fit = fit[binary[fit] == 1]
    positive_validation = validation[binary[validation] == 1]
    base = {
        "learning_rate": 0.035,
        "num_leaves": 127,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 10.0,
        "num_threads": 8,
        "max_bin": 255,
        "verbosity": -1,
    }
    validation_components = []
    best_iterations = []
    for seed in (43, 44):
        params = base | {
            "seed": seed,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
        }
        direct = lgb.train(
            params | {"objective": "regression", "metric": "rmse"},
            lgb.Dataset(train[fit], truth[fit], feature_name=feature_names),
            num_boost_round=MAX_ROUNDS,
            valid_sets=[lgb.Dataset(train[validation], truth[validation], reference=None)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )
        classifier = lgb.train(
            params | {"objective": "binary", "metric": "binary_logloss"},
            lgb.Dataset(train[fit], binary[fit], feature_name=feature_names),
            num_boost_round=MAX_ROUNDS,
            valid_sets=[lgb.Dataset(train[validation], binary[validation], reference=None)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )
        magnitude = lgb.train(
            params | {"objective": "regression", "metric": "rmse"},
            lgb.Dataset(train[positive_fit], truth[positive_fit], feature_name=feature_names),
            num_boost_round=MAX_ROUNDS,
            valid_sets=[lgb.Dataset(
                train[positive_validation], truth[positive_validation], reference=None
            )],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )
        direct_val = direct.predict(train[validation])
        probability_val = classifier.predict(train[validation])
        magnitude_val = magnitude.predict(train[validation])
        validation_components.append((direct_val, probability_val, magnitude_val))
        best_iterations.append((direct.best_iteration, classifier.best_iteration, magnitude.best_iteration))
        print(
            f"seed={seed} iterations={best_iterations[-1]} elapsed={time.time()-started:.0f}s",
            flush=True,
        )
        del direct, classifier, magnitude
        gc.collect()

    direct_val = np.mean([row[0] for row in validation_components], axis=0)
    probability_val = np.mean([row[1] for row in validation_components], axis=0)
    magnitude_val = np.mean([row[2] for row in validation_components], axis=0)
    hurdle_val = probability_val * magnitude_val
    best_mix = None
    best_score = float("inf")
    for mix in np.linspace(0, 1, 21):
        combined = (1 - mix) * direct_val + mix * hurdle_val
        score, coefficients = calibrated_score(combined, truth[validation])
        if score < best_score:
            best_score = score
            best_mix = float(mix)
            best_coefficients = coefficients
    print(
        f"validation_cal={best_score:.6f} hurdle_mix={best_mix:.2f} "
        f"slope={best_coefficients[0]:.5f}",
        flush=True,
    )

    final_components = []
    all_positive = np.flatnonzero(binary == 1)
    for offset, seed in enumerate((43, 44)):
        params = base | {
            "seed": seed,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
        }
        iterations = best_iterations[offset]
        direct = lgb.train(
            params | {"objective": "regression"},
            lgb.Dataset(train, truth, feature_name=feature_names),
            num_boost_round=iterations[0],
        )
        classifier = lgb.train(
            params | {"objective": "binary"},
            lgb.Dataset(train, binary, feature_name=feature_names),
            num_boost_round=iterations[1],
        )
        magnitude = lgb.train(
            params | {"objective": "regression"},
            lgb.Dataset(train[all_positive], truth[all_positive], feature_name=feature_names),
            num_boost_round=iterations[2],
        )
        final_components.append((
            direct.predict(final),
            classifier.predict(final),
            magnitude.predict(final),
        ))
        del direct, classifier, magnitude
        gc.collect()

    direct_final = np.mean([row[0] for row in final_components], axis=0)
    probability_final = np.mean([row[1] for row in final_components], axis=0)
    magnitude_final = np.mean([row[2] for row in final_components], axis=0)
    hurdle_final = probability_final * magnitude_final
    combined_final = (1 - best_mix) * direct_final + best_mix * hurdle_final
    np.save(os.path.join(OUT, TAG + "_final.npy"), combined_final.astype(np.float32))
    np.savez(
        os.path.join(OUT, TAG + "_components.npz"),
        combined=combined_final.astype(np.float32),
        direct=direct_final.astype(np.float32),
        hurdle=hurdle_final.astype(np.float32),
        probability=probability_final.astype(np.float32),
        magnitude=magnitude_final.astype(np.float32),
    )
    report = {
        "train_anchor": TRAIN_ANCHOR,
        "final_anchor": FINAL_ANCHOR,
        "history_window": HISTORY_WINDOW,
        "features": len(feature_names),
        "seeds": [43, 44],
        "best_iterations": best_iterations,
        "validation_calibrated_score": best_score,
        "hurdle_mix": best_mix,
        "validation_coefficients": best_coefficients.tolist(),
        "elapsed_seconds": time.time() - started,
    }
    if FINAL_ANCHOR + 30 < np.load(os.path.join(MAT, "gmv.npy"), mmap_mode="r").shape[1]:
        final_gmv = np.load(os.path.join(MAT, "gmv.npy"), mmap_mode="r")
        if NUSERS:
            final_gmv = final_gmv[:NUSERS]
        final_truth = np.log1p(
            final_gmv[:, FINAL_ANCHOR + 1:FINAL_ANCHOR + 31].sum(
                axis=1, dtype=np.float64
            )
        )
        final_score, final_coefficients = calibrated_score(combined_final, final_truth)
        report["known_final_calibrated_score"] = final_score
        report["known_final_coefficients"] = final_coefficients.tolist()
    with open(os.path.join(OUT, TAG + "_report.json"), "w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
