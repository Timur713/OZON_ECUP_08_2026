#!/usr/bin/env python
"""Frozen strict gate for exact-w409c buyer auxiliary loss."""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get("ECUP_OUT", ROOT / "work"))
TAG = "w409_exact_buyer_s93"
THRESHOLDS = {
    "single_gain": 0.00012,
    "incremental_beyond_w409c": 0.00008,
    "positive_fraction": 0.90,
    "maximum_negative_weight_fraction": 0.10,
}


def one(name):
    rows = json.loads((WORK / name).read_text())
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(name)
    return rows[0]


config = json.loads((WORK / f"{TAG}_config.json").read_text())
if config["seed"] != 93 or config["buyer_auxiliary_loss_weight"] != 0.25:
    raise ValueError("buyer config drift")
baseline = one("w409c_ridge96_current.json")
single = one(f"{TAG}_ridge96.json")
joint = one(f"{TAG}_w409c_joint96.json")
single_gain = float(single["mean_independent_private_gain"])
incremental = float(joint["mean_independent_private_gain"]) - float(
    baseline["mean_independent_private_gain"]
)
positive = float(single["private_gain_positive_fraction"])
single_negative = float(single["candidate_weight_negative_fraction"])
joint_negative = float(
    joint["candidate_weight_negative_fraction"][str(WORK / f"{TAG}_val.npy")]
)
checks = {
    "single_gain": single_gain >= THRESHOLDS["single_gain"],
    "incremental_beyond_w409c": incremental >= THRESHOLDS["incremental_beyond_w409c"],
    "positive_fraction": positive >= THRESHOLDS["positive_fraction"],
    "single_weight_sign": single_negative <= THRESHOLDS["maximum_negative_weight_fraction"],
    "joint_weight_sign": joint_negative <= THRESHOLDS["maximum_negative_weight_fraction"],
}
report = {
    "hypothesis": "buyer auxiliary supervision improves the exact direct w409c representation",
    "uses_public_scores": False,
    "seed": 93,
    "single_independent_gain": single_gain,
    "incremental_gain_beyond_w409c": incremental,
    "positive_fraction": positive,
    "single_negative_weight_fraction": single_negative,
    "joint_negative_weight_fraction": joint_negative,
    "thresholds": THRESHOLDS,
    "checks": checks,
    "passed": all(checks.values()),
}
(WORK / "w409_exact_buyer_decision.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
