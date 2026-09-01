#!/usr/bin/env python
"""Apply fold-342 direct-event epoch once to untouched fold 378."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from build_event_summary_holdout import calibrated_private_score


ROOT = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get("ECUP_OUT", ROOT / "work"))
SELECT_TAG = os.environ.get("ECUP_SELECT_TAG", "directevent409_select342")
HOLDOUT_TAG = os.environ.get("ECUP_HOLDOUT_TAG", "directevent409_holdout378")
OUTPUT_TAG = os.environ.get("ECUP_TAG", "directevent409_growth")
NUSERS = int(os.environ.get("ECUP_NUSERS", "250000"))


def main():
    history = json.loads((WORK / f"{SELECT_TAG}_history.json").read_text())
    if len(history) != 3:
        raise ValueError(f"expected three selection epochs, received {len(history)}")
    if any(float(row["best_hurdle_weight"]) != 0.0 for row in history):
        raise ValueError("direct-only selection unexpectedly used the hurdle head")
    selected = min(history, key=lambda row: row["score"])
    selected_epoch = int(selected["epoch"])

    select_config = json.loads((WORK / f"{SELECT_TAG}_config.json").read_text())
    holdout_config = json.loads((WORK / f"{HOLDOUT_TAG}_config.json").read_text())
    frozen = {
        "window": 409,
        "width": 256,
        "blocks": 8,
        "epochs": 3,
        "seed": 1320,
        "stride": 4,
        "frac": 0.25,
        "channels": "all",
        "summary": True,
        "event_summary": True,
        "calendar": True,
        "market": False,
        "survival_head": False,
        "private_selection": True,
        "anchor_start": 43,
        "require_30_target": True,
        "head_selection": "direct",
        "class_weight": 0.0,
        "magnitude_weight": 0.0,
        "direct_weight": 1.0,
        "mix": 0.0,
    }
    for key, expected in frozen.items():
        if select_config.get(key) != expected or holdout_config.get(key) != expected:
            raise ValueError(f"configuration drift for {key}")

    components_file = WORK / (
        f"{HOLDOUT_TAG}_validate_epoch{selected_epoch}_components.npz"
    )
    with np.load(components_file) as components:
        prediction = components["direct"].astype(np.float64)
    if prediction.shape != (NUSERS,) or not np.isfinite(prediction).all():
        raise ValueError("invalid direct holdout component")

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
    np.save(output_file, prediction)
    report = {
        "tag": OUTPUT_TAG,
        "uses_public_scores": False,
        "uses_recovered_moments": False,
        "uses_competition_target_mean": False,
        "selection_fold": 342,
        "untouched_holdout_fold": 378,
        "selected_epoch": selected_epoch,
        "selected_hurdle_weight": 0.0,
        "selection_private_score": float(selected["score"]),
        "holdout_private_score": score,
        "holdout_slope": slope,
        "holdout_intercept": intercept,
        "validation_file": str(output_file),
        "frozen_config": frozen,
    }
    for comparison_tag in ("w409c", "event409_growth", "control409_growth"):
        comparison_file = WORK / f"{comparison_tag}_val.npy"
        if comparison_file.exists():
            comparison_score, _, _ = calibrated_private_score(
                np.load(comparison_file).astype(np.float64), truth, public, private
            )
            report[f"{comparison_tag}_same_split_score"] = comparison_score
            report[f"gain_vs_{comparison_tag}"] = comparison_score - score
    (WORK / f"{OUTPUT_TAG}_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
