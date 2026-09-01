#!/usr/bin/env python
"""Build an untouched report for a frozen event/control training pair."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get("ECUP_OUT", ROOT / "work"))

parser = argparse.ArgumentParser()
parser.add_argument("--select-tag", required=True)
parser.add_argument("--holdout-tag", required=True)
parser.add_argument("--output-tag", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--event-profile", action="store_true")
parser.add_argument("--role", required=True)
args = parser.parse_args()


def calibrated_private_score(prediction, truth, public, private):
    design = np.column_stack([prediction[public], np.ones(len(public))])
    slope, intercept = np.linalg.lstsq(design, truth[public], rcond=None)[0]
    calibrated = np.clip(prediction[private] * slope + intercept, 0, None)
    score = float(np.sqrt(np.mean((truth[private] - calibrated) ** 2)))
    return score, float(slope), float(intercept)


history = json.loads((WORK / f"{args.select_tag}_history.json").read_text())
if len(history) != 3:
    raise ValueError(f"expected three selection epochs, received {len(history)}")
selected = min(history, key=lambda row: row["score"])
selected_epoch = int(selected["epoch"])
selected_mix = float(selected["best_hurdle_weight"])

select_config = json.loads((WORK / f"{args.select_tag}_config.json").read_text())
holdout_config = json.loads((WORK / f"{args.holdout_tag}_config.json").read_text())
frozen = {
    "window": 409,
    "width": 256,
    "blocks": 8,
    "epochs": 3,
    "seed": args.seed,
    "stride": 4,
    "frac": 0.25,
    "channels": "all",
    "summary": True,
    "event_summary": False,
    "event_profile": args.event_profile,
    "calendar": True,
    "market": False,
    "survival_head": False,
    "private_selection": True,
    "anchor_start": 43,
}
for key, expected in frozen.items():
    # The exact-control trainer predates the optional event-profile flag, so
    # absence is its canonical False value. Profile trainers record it
    # explicitly. All other missing fields remain configuration drift.
    default = False if key == "event_profile" else None
    if (
        select_config.get(key, default) != expected
        or holdout_config.get(key, default) != expected
    ):
        raise ValueError(f"configuration drift for {key}")

components_file = WORK / (
    f"{args.holdout_tag}_validate_epoch{selected_epoch}_components.npz"
)
with np.load(components_file) as components:
    direct = components["direct"].astype(np.float64)
    hurdle = components["hurdle"].astype(np.float64)
if direct.shape != (250_000,) or hurdle.shape != (250_000,):
    raise ValueError("unexpected holdout component shape")
if not np.isfinite(direct).all() or not np.isfinite(hurdle).all():
    raise ValueError("non-finite holdout component")
prediction = selected_mix * hurdle + (1.0 - selected_mix) * direct

gmv = np.load(WORK / "mat" / "gmv.npy", mmap_mode="r")
truth = np.log1p(gmv[:, 379:409].sum(axis=1, dtype=np.float64))
rng = np.random.default_rng(20260825)
public = np.sort(rng.choice(250_000, 50_000, replace=False))
private_mask = np.ones(250_000, dtype=bool)
private_mask[public] = False
private = np.flatnonzero(private_mask)
score, slope, intercept = calibrated_private_score(
    prediction, truth, public, private
)

output_file = WORK / f"{args.output_tag}_val.npy"
np.save(output_file, prediction)
report = {
    "tag": args.output_tag,
    "role": args.role,
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
for comparison_tag in (
    "w409c", "event409_growth", "reg409_growth", "control409_growth",
    "mark409_growth",
):
    comparison_file = WORK / f"{comparison_tag}_val.npy"
    if comparison_file.exists():
        comparison_score, _, _ = calibrated_private_score(
            np.load(comparison_file).astype(np.float64), truth, public, private
        )
        report[f"{comparison_tag}_same_split_score"] = comparison_score
        report[f"gain_vs_{comparison_tag}"] = comparison_score - score
(WORK / f"{args.output_tag}_report.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
