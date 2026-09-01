#!/usr/bin/env python
"""Frozen promotion decision for exact-w409c multiscale pooling."""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get("ECUP_OUT", ROOT / "work"))
STRICT = {
    "single_gain": 0.00012,
    "incremental_beyond_w409c": 0.00008,
    "positive_fraction": 0.90,
    "maximum_negative_weight_fraction": 0.10,
    "minimum_each_seed_gain": 0.0,
}
DIAGNOSTIC = {
    "single_gain": 0.00004,
    "incremental_beyond_w409c": 0.00002,
    "positive_fraction": 0.75,
    "maximum_negative_weight_fraction": 0.25,
    "minimum_each_seed_gain": -0.00002,
}


def one(name):
    rows = json.loads((WORK / name).read_text())
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(f"invalid audit {name}")
    return rows[0]


baseline = one("w409c_ridge96_current.json")
average = one("w409_exact_decay_seedavg_ridge96.json")
joint = one("w409_exact_decay_seedavg_w409c_joint96.json")
seed_rows = [
    one("w409_exact_decay_s93_ridge96.json"),
    one("w409_exact_decay_s1310_ridge96.json"),
]
single_gain = float(average["mean_independent_private_gain"])
incremental = float(joint["mean_independent_private_gain"]) - float(
    baseline["mean_independent_private_gain"]
)
positive = float(average["private_gain_positive_fraction"])
single_negative = float(average["candidate_weight_negative_fraction"])
joint_negative_map = joint["candidate_weight_negative_fraction"]
joint_negative = next(
    float(value) for path, value in joint_negative_map.items()
    if Path(path).name == "promote_w409_exact_decay_val_seedavg.npy"
)
individual_gains = [float(row["mean_independent_private_gain"]) for row in seed_rows]


def checks(thresholds):
    return {
        "single_gain": single_gain >= thresholds["single_gain"],
        "incremental_beyond_w409c": incremental >= thresholds["incremental_beyond_w409c"],
        "positive_fraction": positive >= thresholds["positive_fraction"],
        "single_weight_sign": single_negative <= thresholds["maximum_negative_weight_fraction"],
        "joint_weight_sign": joint_negative <= thresholds["maximum_negative_weight_fraction"],
        "both_seeds_noncatastrophic": min(individual_gains) >= thresholds["minimum_each_seed_gain"],
    }


strict_checks = checks(STRICT)
diagnostic_checks = checks(DIAGNOSTIC)
tier = (
    "strict" if all(strict_checks.values()) else
    "diagnostic" if all(diagnostic_checks.values()) else
    "reject"
)
report = {
    "hypothesis": "multiscale hidden-state time pooling on the exact successful w409c recipe",
    "only_change_from_w409c": "four normalized hidden-state decay pools",
    "uses_public_scores": False,
    "uses_recovered_moments": False,
    "seeds": [93, 1310],
    "single_independent_gain": single_gain,
    "incremental_gain_beyond_w409c": incremental,
    "positive_fraction": positive,
    "single_negative_weight_fraction": single_negative,
    "joint_negative_weight_fraction": joint_negative,
    "individual_seed_gains": individual_gains,
    "strict_thresholds": STRICT,
    "diagnostic_thresholds": DIAGNOSTIC,
    "strict_checks": strict_checks,
    "diagnostic_checks": diagnostic_checks,
    "tier": tier,
    "decision": "build one frozen public measurement CSV" if tier != "reject" else "reject without CSV",
}
(WORK / "w409_exact_decay_decision.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
