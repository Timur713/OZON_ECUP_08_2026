#!/usr/bin/env python
"""Apply fold-342 event-profile choices once to untouched fold 378."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get("ECUP_OUT", ROOT / "work"))
SELECT_TAG = os.environ.get("ECUP_SELECT_TAG", "reg409_select342")
HOLDOUT_TAG = os.environ.get("ECUP_HOLDOUT_TAG", "reg409_holdout378")
OUTPUT_TAG = os.environ.get("ECUP_TAG", "reg409_growth")
NUSERS = int(os.environ.get("ECUP_NUSERS", "250000"))


def calibrated_private_score(prediction, truth, public, private):
    design = np.column_stack([prediction[public], np.ones(len(public))])
    slope, intercept = np.linalg.lstsq(design, truth[public], rcond=None)[0]
    calibrated = np.clip(prediction[private] * slope + intercept, 0, None)
    score = float(np.sqrt(np.mean((truth[private] - calibrated) ** 2)))
    return score, float(slope), float(intercept)


def main():
    history = json.loads((WORK / f"{SELECT_TAG}_history.json").read_text())
    if len(history) != 3:
        raise ValueError(f"expected three selection epochs, received {len(history)}")
    selected = min(history, key=lambda row: row["score"])
    selected_epoch = int(selected["epoch"])
    selected_mix = float(selected["best_hurdle_weight"])

    select_config = json.loads((WORK / f"{SELECT_TAG}_config.json").read_text())
    holdout_config = json.loads((WORK / f"{HOLDOUT_TAG}_config.json").read_text())
    frozen = {
        "window": 409,
        "width": 256,
        "blocks": 8,
        "epochs": 3,
        "seed": 1310,
        "stride": 4,
        "frac": 0.25,
        "channels": "all",
        "summary": True,
        "event_summary": False,
        "event_profile": True,
        "calendar": True,
        "market": False,
        "survival_head": False,
        "private_selection": True,
        "anchor_start": 43,
        "require_30_target": False,
        "head_selection": "auto",
    }
    for key, expected in frozen.items():
        if select_config.get(key) != expected or holdout_config.get(key) != expected:
            raise ValueError(f"configuration drift for {key}")

    components_file = WORK / (
        f"{HOLDOUT_TAG}_validate_epoch{selected_epoch}_components.npz"
    )
    with np.load(components_file) as components:
        direct = components["direct"].astype(np.float64)
        hurdle = components["hurdle"].astype(np.float64)
    if direct.shape != (NUSERS,) or hurdle.shape != (NUSERS,):
        raise ValueError("unexpected holdout component shape")
    if not np.isfinite(direct).all() or not np.isfinite(hurdle).all():
        raise ValueError("non-finite holdout component")

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
        "uses_recovered_moments": False,
        "uses_competition_target_mean": False,
        "selection_fold": 342,
        "untouched_holdout_fold": 378,
        "selected_epoch": selected_epoch,
        "selected_hurdle_weight": selected_mix,
        "selection_private_score": float(selected["score"]),
        "holdout_private_score": score,
        "holdout_slope": slope,
        "holdout_intercept": intercept,
        "validation_file": str(output_file),
        "frozen_config": frozen,
    }
    for baseline_tag in ("w409c", "event409_growth"):
        baseline_file = WORK / f"{baseline_tag}_val.npy"
        if baseline_file.exists():
            baseline_score, _, _ = calibrated_private_score(
                np.load(baseline_file).astype(np.float64), truth, public, private
            )
            report[f"{baseline_tag}_same_split_score"] = baseline_score
            report[f"gain_vs_{baseline_tag}"] = baseline_score - score
    (WORK / f"{OUTPUT_TAG}_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
