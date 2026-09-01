#!/usr/bin/env python
"""Frozen incremental gate for combined exact-w409c position+decay pooling."""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get("ECUP_OUT", ROOT / "work"))
TAG = "w409_exact_position_decay_s93"
THRESHOLDS = {
    "single_gain": 0.00012,
    "incremental_beyond_separate_families": 0.00008,
    "positive_fraction": 0.90,
    "maximum_negative_weight_fraction": 0.10,
}


def one(name):
    rows = json.loads((WORK / name).read_text())
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(name)
    return rows[0]


config = json.loads((WORK / f"{TAG}_config.json").read_text())
if (
    config["seed"] != 93
    or config["only_change_from_w409c"] != "position_decay"
    or config["decays"] != [7.0, 30.0, 90.0, 180.0]
):
    raise ValueError("position-decay config drift")
single = one(f"{TAG}_ridge96.json")
separate = one("w409_exact_separate_families_joint96.json")
combined = one("w409_exact_position_decay_full_joint96.json")
single_gain = float(single["mean_independent_private_gain"])
incremental = float(combined["mean_independent_private_gain"]) - float(
    separate["mean_independent_private_gain"]
)
positive = float(single["private_gain_positive_fraction"])
single_negative = float(single["candidate_weight_negative_fraction"])
candidate_path = str(WORK / f"{TAG}_val.npy")
joint_negative = float(combined["candidate_weight_negative_fraction"][candidate_path])
checks = {
    "single_gain": single_gain >= THRESHOLDS["single_gain"],
    "incremental_beyond_separate_families": incremental
    >= THRESHOLDS["incremental_beyond_separate_families"],
    "positive_fraction": positive >= THRESHOLDS["positive_fraction"],
    "single_weight_sign": single_negative <= THRESHOLDS["maximum_negative_weight_fraction"],
    "joint_weight_sign": joint_negative <= THRESHOLDS["maximum_negative_weight_fraction"],
}
report = {
    "hypothesis": "relative-position inputs and fixed multiscale pooling are complementary on exact w409c",
    "mechanism_and_trainer_frozen_before_131_public": True,
    "formal_incremental_gate_file_created_after_131_public": True,
    "uses_public_scores": False,
    "seed": 93,
    "single_independent_gain": single_gain,
    "separate_families_joint_gain": float(separate["mean_independent_private_gain"]),
    "combined_full_joint_gain": float(combined["mean_independent_private_gain"]),
    "incremental_gain_beyond_separate_families": incremental,
    "positive_fraction": positive,
    "single_negative_weight_fraction": single_negative,
    "joint_negative_weight_fraction": joint_negative,
    "thresholds": THRESHOLDS,
    "checks": checks,
    "passed": all(checks.values()),
}
(WORK / "w409_exact_position_decay_decision.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
