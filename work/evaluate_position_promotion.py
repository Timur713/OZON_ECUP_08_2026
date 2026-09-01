#!/usr/bin/env python
"""Frozen two-seed gate for explicit relative-position channels."""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get("ECUP_OUT", ROOT / "work"))
TAGS = {1310: "pos409_growth", 2718: "pos409s2718_growth"}
CONTROLS = {1310: "control409_growth", 2718: "control409s2718_growth"}
THRESHOLDS = {
    "minimum_single_independent_gain": 0.00012,
    "minimum_incremental_gain_beyond_w409c": 0.00008,
    "minimum_positive_fraction": 0.90,
    "maximum_negative_weight_fraction": 0.10,
    "minimum_holdout_gain_vs_same_seed_control": 0.00020,
}
DIAGNOSTIC_THRESHOLDS = {
    "minimum_single_independent_gain": 0.00004,
    "minimum_incremental_gain_beyond_w409c": 0.00002,
    "minimum_positive_fraction": 0.75,
    "maximum_negative_weight_fraction": 0.25,
    "minimum_holdout_gain_vs_same_seed_control": -0.00050,
}


def load(path: Path):
    return json.loads(path.read_text())


def audit(tag: str, suffix: str):
    rows = load(WORK / f"{tag}_{suffix}.json")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(f"invalid audit {tag}_{suffix}")
    return rows[0]


def negative_fraction(row: dict, tag: str):
    value = row["candidate_weight_negative_fraction"]
    if isinstance(value, (int, float)):
        return float(value)
    matches = [
        float(fraction) for path, fraction in value.items()
        if Path(path).name == f"{tag}_val.npy"
    ]
    if len(matches) != 1:
        raise ValueError(f"candidate weight missing for {tag}")
    return matches[0]


baseline_rows = load(WORK / "w409c_ridge96_current.json")
if not isinstance(baseline_rows, list) or len(baseline_rows) != 1:
    raise ValueError("invalid w409c baseline")
baseline_gain = float(baseline_rows[0]["mean_independent_private_gain"])
seed_rows = []
for seed, tag in TAGS.items():
    prefix = "pos409" if seed == 1310 else "pos409s2718"
    for split_tag in (f"{prefix}_select342", f"{prefix}_holdout378"):
        config = load(WORK / f"{split_tag}_config.json")
        if config.get("relative_position_channels") != 2:
            raise ValueError(f"relative-position config drift for {split_tag}")
        if config.get("seed") != seed:
            raise ValueError(f"seed drift for {split_tag}")
        expected_names = {"relative_age_linear", "relative_age_decay30"}
        if not expected_names.issubset(config.get("channel_names", [])):
            raise ValueError(f"relative-position channels missing for {split_tag}")
    single = audit(tag, "ridge96")
    joint = audit(tag, "w409c_joint96")
    model = load(WORK / f"{tag}_report.json")
    control = load(WORK / f"{CONTROLS[seed]}_report.json")
    single_gain = float(single["mean_independent_private_gain"])
    incremental = float(joint["mean_independent_private_gain"]) - baseline_gain
    positive = float(single["private_gain_positive_fraction"])
    single_negative = negative_fraction(single, tag)
    joint_negative = negative_fraction(joint, tag)
    holdout_gain = (
        float(control["holdout_private_score"])
        - float(model["holdout_private_score"])
    )
    checks = {
        "single_gain": single_gain >= THRESHOLDS["minimum_single_independent_gain"],
        "incremental_beyond_w409c": incremental
        >= THRESHOLDS["minimum_incremental_gain_beyond_w409c"],
        "positive_fraction": positive >= THRESHOLDS["minimum_positive_fraction"],
        "single_weight_sign": single_negative
        <= THRESHOLDS["maximum_negative_weight_fraction"],
        "joint_weight_sign": joint_negative
        <= THRESHOLDS["maximum_negative_weight_fraction"],
        "holdout": holdout_gain
        >= THRESHOLDS["minimum_holdout_gain_vs_same_seed_control"],
    }
    diagnostic_checks = {
        "single_gain": single_gain
        >= DIAGNOSTIC_THRESHOLDS["minimum_single_independent_gain"],
        "incremental_beyond_w409c": incremental
        >= DIAGNOSTIC_THRESHOLDS["minimum_incremental_gain_beyond_w409c"],
        "positive_fraction": positive
        >= DIAGNOSTIC_THRESHOLDS["minimum_positive_fraction"],
        "single_weight_sign": single_negative
        <= DIAGNOSTIC_THRESHOLDS["maximum_negative_weight_fraction"],
        "joint_weight_sign": joint_negative
        <= DIAGNOSTIC_THRESHOLDS["maximum_negative_weight_fraction"],
        "holdout": holdout_gain
        >= DIAGNOSTIC_THRESHOLDS["minimum_holdout_gain_vs_same_seed_control"],
    }
    seed_rows.append({
        "seed": seed,
        "tag": tag,
        "single_independent_gain": single_gain,
        "incremental_gain_beyond_w409c": incremental,
        "positive_fraction": positive,
        "single_negative_weight_fraction": single_negative,
        "joint_negative_weight_fraction": joint_negative,
        "holdout_private_score": float(model["holdout_private_score"]),
        "same_seed_control_private_score": float(control["holdout_private_score"]),
        "holdout_gain_vs_control": holdout_gain,
        "checks": checks,
        "passed": all(checks.values()),
        "diagnostic_checks": diagnostic_checks,
        "diagnostic_passed": all(diagnostic_checks.values()),
    })

passed = all(row["passed"] for row in seed_rows)
diagnostic_passed = all(row["diagnostic_passed"] for row in seed_rows)
tier = "strict" if passed else "diagnostic" if diagnostic_passed else "reject"
report = {
    "hypothesis": "explicit relative age lets the pooled TCN preserve where an event occurred in the 409-day history",
    "uses_public_scores": False,
    "uses_recovered_moments": False,
    "seeds": seed_rows,
    "thresholds": THRESHOLDS,
    "diagnostic_thresholds": DIAGNOSTIC_THRESHOLDS,
    "passed": passed,
    "diagnostic_passed": diagnostic_passed,
    "tier": tier,
    "decision": (
        "run frozen two-seed full refit and build one seed-average probe"
        if tier != "reject" else "reject without full refit or CSV"
    ),
}
(WORK / "position_promotion_decision.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
