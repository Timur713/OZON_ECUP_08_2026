#!/usr/bin/env python
"""Confirm two live representation hypotheses on the frozen third seed.

This is a leaderboard-blind confirmation layer.  It cannot rescue a rejected
two-seed family: it only upgrades an already strict/diagnostic family to a
three-seed-confirmed recommendation when seed 31415 clears the same tier.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get("ECUP_OUT", ROOT / "work"))
SEED = 31415
FAMILIES = {
    "control": {
        "tag": "control409s31415_growth",
        "two_seed_file": "profile_promotion_decision.json",
    },
    "position": {
        "tag": "pos409s31415_growth",
        "two_seed_file": "position_promotion_decision.json",
    },
    "marked": {
        "tag": "mark409s31415_growth",
        "two_seed_file": "profile_promotion_decision.json",
    },
}
CONTROL = "control409s31415_growth"
STRICT = {
    "minimum_single_independent_gain": 0.00012,
    "minimum_incremental_gain_beyond_w409c": 0.00008,
    "minimum_positive_fraction": 0.90,
    "maximum_negative_weight_fraction": 0.10,
    "minimum_holdout_gain_vs_same_seed_control": 0.00020,
    "maximum_control_holdout_loss_vs_w409c": 0.00050,
}
DIAGNOSTIC = {
    "minimum_single_independent_gain": 0.00004,
    "minimum_incremental_gain_beyond_w409c": 0.00002,
    "minimum_positive_fraction": 0.75,
    "maximum_negative_weight_fraction": 0.25,
    "minimum_holdout_gain_vs_same_seed_control": -0.00050,
    "maximum_control_holdout_loss_vs_w409c": 0.00150,
}


def load(name: str):
    return json.loads((WORK / name).read_text())


def audit(tag: str, suffix: str):
    rows = load(f"{tag}_{suffix}.json")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(f"invalid audit {tag}_{suffix}")
    return rows[0]


def negative_fraction(row: dict, tag: str) -> float:
    value = row["candidate_weight_negative_fraction"]
    if isinstance(value, (int, float)):
        return float(value)
    matches = [
        float(fraction)
        for path, fraction in value.items()
        if Path(path).name == f"{tag}_val.npy"
    ]
    if len(matches) != 1:
        raise ValueError(f"candidate weight missing for {tag}")
    return matches[0]


baseline_rows = load("w409c_ridge96_current.json")
if not isinstance(baseline_rows, list) or len(baseline_rows) != 1:
    raise ValueError("invalid w409c baseline audit")
baseline_gain = float(baseline_rows[0]["mean_independent_private_gain"])
control_report = load(f"{CONTROL}_report.json")
report = {
    "uses_public_scores": False,
    "uses_recovered_moments": False,
    "seed": SEED,
    "rule": "third seed confirms but can never rescue a rejected two-seed family",
    "strict_thresholds": STRICT,
    "diagnostic_thresholds": DIAGNOSTIC,
    "families": {},
}

for family, spec in FAMILIES.items():
    tag = spec["tag"]
    single = audit(tag, "ridge96")
    joint = audit(tag, "w409c_joint96")
    model_report = load(f"{tag}_report.json")
    single_gain = float(single["mean_independent_private_gain"])
    incremental = float(joint["mean_independent_private_gain"]) - baseline_gain
    positive = float(single["private_gain_positive_fraction"])
    single_negative = negative_fraction(single, tag)
    joint_negative = negative_fraction(joint, tag)
    if family == "control":
        holdout_gain = (
            float(model_report["w409c_same_split_score"])
            - float(model_report["holdout_private_score"])
        )
    else:
        holdout_gain = (
            float(control_report["holdout_private_score"])
            - float(model_report["holdout_private_score"])
        )

    def checks(thresholds: dict):
        return {
            "single_gain": single_gain
            >= thresholds["minimum_single_independent_gain"],
            "incremental_beyond_w409c": incremental
            >= thresholds["minimum_incremental_gain_beyond_w409c"],
            "positive_fraction": positive
            >= thresholds["minimum_positive_fraction"],
            "single_weight_sign": single_negative
            <= thresholds["maximum_negative_weight_fraction"],
            "joint_weight_sign": joint_negative
            <= thresholds["maximum_negative_weight_fraction"],
            "holdout": (
                holdout_gain >= -thresholds["maximum_control_holdout_loss_vs_w409c"]
                if family == "control"
                else holdout_gain
                >= thresholds["minimum_holdout_gain_vs_same_seed_control"]
            ),
        }

    strict_checks = checks(STRICT)
    diagnostic_checks = checks(DIAGNOSTIC)
    seed_tier = (
        "strict" if all(strict_checks.values())
        else "diagnostic" if all(diagnostic_checks.values())
        else "reject"
    )
    two_seed = load(spec["two_seed_file"])
    two_seed_tier = (
        two_seed["tier"] if family == "position"
        else two_seed["families"][family]["tier"]
    )
    confirmed_tier = (
        "strict" if two_seed_tier == seed_tier == "strict"
        else "diagnostic"
        if two_seed_tier in {"strict", "diagnostic"}
        and seed_tier in {"strict", "diagnostic"}
        else "reject"
    )
    report["families"][family] = {
        "tag": tag,
        "single_independent_gain": single_gain,
        "incremental_gain_beyond_w409c": incremental,
        "positive_fraction": positive,
        "single_negative_weight_fraction": single_negative,
        "joint_negative_weight_fraction": joint_negative,
        "holdout_private_score": float(model_report["holdout_private_score"]),
        "same_seed_control_private_score": float(
            control_report["holdout_private_score"]
        ),
        "holdout_gain_vs_control": holdout_gain,
        "strict_checks": strict_checks,
        "diagnostic_checks": diagnostic_checks,
        "seed31415_tier": seed_tier,
        "two_seed_tier": two_seed_tier,
        "three_seed_confirmed_tier": confirmed_tier,
        "submission_recommendation": confirmed_tier != "reject",
    }

(WORK / "seed31415_confirmation.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
