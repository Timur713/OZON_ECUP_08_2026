#!/usr/bin/env python
"""Conditionally build a clean lagged-residual seventh-family candidate.

The trigger and gates are frozen in lagged_residual_followup_preregister.json.
No public score, recovered target moment, or competition label is read.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import nnls


ROOT = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get("ECUP_OUT", ROOT / "work"))
OUTPUT = ROOT / "submissions" / "candidate_lagged_residual_clean.csv"
META = WORK / "lagged_residual_clean_candidate_meta.json"
PREREG = WORK / "lagged_residual_followup_preregister.json"
PAIR_REPORT = WORK / "residcurve_pairsafe_report.json"
PAIR_ARRAY = WORK / "residcurve_pairsafe_residual_pairs.npy"
LAMBDA = 0.001
REPEATS = 96
SEED = 20260825


def load(name: str) -> np.ndarray:
    return np.load(WORK / name).astype(np.float64)


def fit(design: np.ndarray, target: np.ndarray, index: np.ndarray) -> np.ndarray:
    x = design[index, :-1]
    y = target[index]
    x_mean = x.mean(axis=0)
    y_mean = y.mean()
    scale = np.sqrt(len(index))
    augmented_x = np.vstack([
        (x - x_mean) / scale,
        np.sqrt(LAMBDA) * np.eye(x.shape[1]),
    ])
    augmented_y = np.r_[
        (y - y_mean) / scale,
        np.zeros(x.shape[1]),
    ]
    model_weights, _ = nnls(augmented_x, augmented_y, maxiter=10_000)
    intercept = y_mean - x_mean @ model_weights
    return np.r_[model_weights, intercept]


def shift_to_mean(values: np.ndarray, target_mean: float) -> np.ndarray:
    low, high = -8.0, 8.0
    for _ in range(100):
        middle = (low + high) / 2
        if np.clip(values + middle, 0, None).mean() < target_mean:
            low = middle
        else:
            high = middle
    return np.clip(values + (low + high) / 2, 0, None)


def stop(report: dict[str, object], reason: str) -> None:
    report["promoted"] = False
    report["decision"] = reason
    META.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    prereg = json.loads(PREREG.read_text())
    pair_report = json.loads(PAIR_REPORT.read_text())
    report: dict[str, object] = {
        "uses_public_scores": False,
        "uses_recovered_moments": False,
        "uses_competition_target_mean": False,
        "preregister": str(PREREG),
        "pair_report": str(PAIR_REPORT),
    }
    pairs = pair_report["pairs"]
    expected_last = [348, 378]
    if len(pairs) != 8 or pairs[-1]["anchors"] != expected_last:
        stop(report, "invalid pair-safe chronology")
        return
    preholdout = np.asarray(
        [float(row["correlation"]) for row in pairs[:-1]], dtype=np.float64
    )
    trigger = prereg["trigger"]
    trigger_checks = {
        "controls_valid": bool(pair_report["controls_valid"]),
        "full_diagnostic_support_diagnostic_only": bool(
            pair_report["supports_stable_user_residual_branch"]
        ),
        "preholdout_mean": float(preholdout.mean()),
        "preholdout_positive_pairs": int(np.count_nonzero(preholdout > 0)),
        "preholdout_mean_pass": bool(
            preholdout.mean()
            >= trigger["preholdout_minimum_mean_correlation"]
        ),
        "preholdout_sign_pass": bool(
            np.count_nonzero(preholdout > 0)
            >= trigger["preholdout_minimum_positive_pairs"]
        ),
    }
    report["trigger_checks"] = trigger_checks
    if not all([
        trigger_checks["controls_valid"],
        trigger_checks["preholdout_mean_pass"],
        trigger_checks["preholdout_sign_pass"],
    ]):
        stop(report, "pair-safe trigger failed; no candidate or CSV built")
        return

    residual_pairs = np.load(PAIR_ARRAY).astype(np.float64)
    if residual_pairs.shape != (8, 2, 250_000):
        stop(report, f"invalid residual array shape {residual_pairs.shape}")
        return
    validation_residual = residual_pairs[-1, 0]
    final_residual = residual_pairs[-1, 1]

    family_files = {
        "GBD262": ("v4_262_valpred.npy", "AVG_GBD.npy"),
        "W120_seed_average": (None, None),
        "TCN180_two_head": ("tcn180two_val.npy", "tcn180two_final.npy"),
        "TCN365_growing_anchor": (
            "tcn365v336_val.npy", "tcn365v336_final.npy"
        ),
        "TCN409": ("tcn409_val.npy", "tcn409_final.npy"),
        "TCN409_replication": ("w409c_val.npy", "w409c_final.npy"),
    }
    validation_columns = []
    final_columns = []
    for name, (validation_file, final_file) in family_files.items():
        if name == "W120_seed_average":
            validation_columns.append(np.mean([
                load(f"w120{seed}_val.npy") for seed in "abc"
            ], axis=0))
            final_columns.append(np.mean([
                load(f"w120{seed}_final.npy") for seed in "abc"
            ], axis=0))
        else:
            validation_columns.append(load(validation_file))
            final_columns.append(load(final_file))
    base_validation = np.column_stack([
        *validation_columns, np.ones(250_000, dtype=np.float64)
    ])
    base_final = np.column_stack([
        *final_columns, np.ones(250_000, dtype=np.float64)
    ])
    augmented_validation = np.column_stack([
        *validation_columns, validation_residual,
        np.ones(250_000, dtype=np.float64),
    ])
    augmented_final = np.column_stack([
        *final_columns, final_residual,
        np.ones(250_000, dtype=np.float64),
    ])

    gmv = np.load(WORK / "mat" / "gmv.npy", mmap_mode="r")[:250_000]
    truth = np.log1p(gmv[:, 379:409].sum(axis=1, dtype=np.float64))
    all_index = np.arange(250_000)
    rng = np.random.default_rng(SEED)
    gains = []
    residual_weights = []
    for _ in range(REPEATS):
        fit_index = rng.choice(250_000, 50_000, replace=False)
        score_mask = np.ones(250_000, dtype=bool)
        score_mask[fit_index] = False
        score_index = all_index[score_mask]
        base_weights = fit(base_validation, truth, fit_index)
        augmented_weights = fit(augmented_validation, truth, fit_index)
        base_prediction = np.clip(
            base_validation[score_index] @ base_weights, 0, None
        )
        augmented_prediction = np.clip(
            augmented_validation[score_index] @ augmented_weights, 0, None
        )
        base_score = np.sqrt(np.mean(
            (truth[score_index] - base_prediction) ** 2
        ))
        augmented_score = np.sqrt(np.mean(
            (truth[score_index] - augmented_prediction) ** 2
        ))
        gains.append(float(base_score - augmented_score))
        residual_weights.append(float(augmented_weights[-2]))

    base_full_weights = fit(base_validation, truth, all_index)
    augmented_full_weights = fit(augmented_validation, truth, all_index)
    base_full_score = float(np.sqrt(np.mean((
        truth - np.clip(base_validation @ base_full_weights, 0, None)
    ) ** 2)))
    augmented_full_score = float(np.sqrt(np.mean((
        truth - np.clip(augmented_validation @ augmented_full_weights, 0, None)
    ) ** 2)))
    clean_meta = json.loads(
        (WORK / "119_offline_rules_safe_6model_meta.json").read_text()
    )
    expected_base_score = float(clean_meta["validation_score_full_250k"])
    if abs(base_full_score - expected_base_score) > 1e-10:
        raise RuntimeError(
            "six-family baseline drift: "
            f"{base_full_score} != {expected_base_score}"
        )
    gains_array = np.asarray(gains)
    weights_array = np.asarray(residual_weights)
    gates = prereg["promotion_gates"]
    gate_results = {
        "mean_independent_private_gain": float(gains_array.mean()),
        "positive_gain_splits": int(np.count_nonzero(gains_array > 0)),
        "residual_weight_positive_splits": int(
            np.count_nonzero(weights_array > 1e-10)
        ),
        "full_validation_gain": base_full_score - augmented_full_score,
        "mean_gain_pass": bool(
            gains_array.mean() >= gates["mean_independent_private_gain_min"]
        ),
        "positive_splits_pass": bool(
            np.count_nonzero(gains_array > 0)
            >= gates["positive_gain_splits_min"]
        ),
        "weight_sign_pass": bool(
            np.count_nonzero(weights_array > 1e-10)
            >= gates["residual_weight_positive_splits_min"]
        ),
        "full_gain_pass": bool(
            base_full_score - augmented_full_score
            >= gates["full_validation_gain_min"]
        ),
    }
    report.update({
        "repeats": REPEATS,
        "lambda": LAMBDA,
        "base_full_validation_score": base_full_score,
        "expected_base_full_validation_score": expected_base_score,
        "augmented_full_validation_score": augmented_full_score,
        "residual_full_weight": float(augmented_full_weights[-2]),
        "gate_results": gate_results,
    })
    if not all([
        gate_results["mean_gain_pass"],
        gate_results["positive_splits_pass"],
        gate_results["weight_sign_pass"],
        gate_results["full_gain_pass"],
    ]):
        stop(report, "offline promotion gates failed; no CSV built")
        return

    raw_final = np.clip(augmented_final @ augmented_full_weights, 0, None)
    mean_meta = json.loads(
        (WORK / "120_offline_rules_safe_meanforecast_meta.json").read_text()
    )
    if mean_meta["uses_public_scores"] or mean_meta["uses_recovered_moments"]:
        raise ValueError("frozen mean forecast is not leaderboard-free")
    final_log = shift_to_mean(raw_final, mean_meta["forecast_mean_log1p"])
    baseline_log = np.log1p(
        np.maximum(
            0,
            np.genfromtxt(
                ROOT / "submissions" / "120_offline_rules_safe_meanforecast.csv",
                delimiter=",", skip_header=1, usecols=1,
            ),
        )
    )
    distance = float(np.sqrt(np.mean((final_log - baseline_log) ** 2)))
    report["final_rms_log_distance_from_120"] = distance
    report["distance_pass"] = bool(
        distance >= gates["final_rms_log_distance_from_120_min"]
    )
    if not report["distance_pass"]:
        stop(report, "final prediction distance gate failed; no CSV built")
        return

    prediction = np.expm1(final_log)
    uids = np.load(WORK / "mat" / "uids.npy")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["user_id", "predict"])
        writer.writerows(
            (int(user_id), float(value))
            for user_id, value in zip(uids, prediction)
        )
    report.update({
        "promoted": True,
        "decision": "all frozen offline gates passed; plain CSV built, not submitted",
        "output": str(OUTPUT),
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "rows": len(prediction),
        "output_mean_log1p": float(final_log.mean()),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
    })
    META.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
