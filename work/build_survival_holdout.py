#!/usr/bin/env python
"""Freeze fold-342 survival choices and score them once on fold 378."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get("ECUP_OUT", ROOT / "work"))
SELECT_TAG = os.environ.get("ECUP_SELECT_TAG", "surv409_select342")
HOLDOUT_TAG = os.environ.get("ECUP_HOLDOUT_TAG", "surv409_holdout378")
OUTPUT_TAG = os.environ.get("ECUP_TAG", "surv409_growth")
NUSERS = int(os.environ.get("ECUP_NUSERS", "250000"))


def calibrated_private_score(
    prediction: np.ndarray,
    truth: np.ndarray,
    public: np.ndarray,
    private: np.ndarray,
) -> tuple[float, float, float]:
    design = np.column_stack([prediction[public], np.ones(len(public))])
    slope, intercept = np.linalg.lstsq(design, truth[public], rcond=None)[0]
    calibrated = np.clip(prediction[private] * slope + intercept, 0, None)
    score = float(np.sqrt(np.mean((truth[private] - calibrated) ** 2)))
    return score, float(slope), float(intercept)


def main() -> None:
    history = json.loads((WORK / f"{SELECT_TAG}_history.json").read_text())
    selected = min(history, key=lambda row: row["score"])
    selected_epoch = int(selected["epoch"])
    selected_mix = float(selected["best_hurdle_weight"])

    components_file = WORK / (
        f"{HOLDOUT_TAG}_validate_epoch{selected_epoch}_components.npz"
    )
    with np.load(components_file) as components:
        direct = components["direct"].astype(np.float64)
        hurdle = components["hurdle"].astype(np.float64)
        horizon_probabilities = components["probability_horizons"].astype(np.float64)
    if direct.shape != (NUSERS,) or hurdle.shape != (NUSERS,):
        raise ValueError("unexpected holdout component shape")
    monotonic_violations = int(
        np.count_nonzero(np.diff(horizon_probabilities, axis=1) < -1e-7)
    )
    if monotonic_violations:
        raise ValueError(f"survival probabilities are not monotone: {monotonic_violations}")

    prediction = selected_mix * hurdle + (1.0 - selected_mix) * direct
    gmv = np.load(WORK / "mat" / "gmv.npy", mmap_mode="r")[:NUSERS]
    truth = np.log1p(gmv[:, 379:409].sum(axis=1, dtype=np.float64))
    rng = np.random.default_rng(20260825)
    public = np.sort(rng.choice(NUSERS, min(50_000, NUSERS // 5), replace=False))
    private_mask = np.ones(NUSERS, dtype=bool)
    private_mask[public] = False
    private = np.flatnonzero(private_mask)
    score, slope, intercept = calibrated_private_score(
        prediction, truth, public, private
    )

    output_file = WORK / f"{OUTPUT_TAG}_val.npy"
    np.save(output_file, prediction.astype(np.float64))
    report = {
        "tag": OUTPUT_TAG,
        "uses_public_scores": False,
        "selection_fold": 342,
        "untouched_holdout_fold": 378,
        "selected_epoch": selected_epoch,
        "selected_hurdle_weight": selected_mix,
        "selection_private_score": float(selected["score"]),
        "holdout_private_score": score,
        "holdout_slope": slope,
        "holdout_intercept": intercept,
        "monotonic_probability_violations": monotonic_violations,
        "validation_file": str(output_file),
    }
    w409c_file = WORK / "w409c_val.npy"
    if w409c_file.exists():
        w409c_score, _, _ = calibrated_private_score(
            np.load(w409c_file).astype(np.float64), truth, public, private
        )
        report["w409c_same_split_score"] = w409c_score
        report["gain_vs_w409c"] = w409c_score - score
    (WORK / f"{OUTPUT_TAG}_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
