#!/usr/bin/env python
"""Frozen two-seed promotion decision for the profile experiment family.

This script is intentionally leaderboard-blind.  It is run only after every
seed-1310/2718 validation artifact has been written, and it promotes every
family that clears the same preregistered gates.  It never chooses the best
family among failures and never creates a submission itself.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get("ECUP_OUT", ROOT / "work"))
SEEDS = (1310, 2718)
FAMILIES = {
    "control": {
        1310: "control409_growth",
        2718: "control409s2718_growth",
    },
    "marked": {
        1310: "mark409_growth",
        2718: "mark409s2718_growth",
    },
}
THRESHOLDS = {
    "minimum_single_independent_gain": 0.00012,
    "minimum_incremental_gain_beyond_w409c": 0.00008,
    "minimum_positive_fraction": 0.90,
    "maximum_negative_weight_fraction": 0.10,
    "maximum_control_holdout_loss_vs_w409c": 0.00050,
    "minimum_profile_holdout_gain_vs_control": 0.00020,
}
DIAGNOSTIC_THRESHOLDS = {
    "minimum_single_independent_gain": 0.00004,
    "minimum_incremental_gain_beyond_w409c": 0.00002,
    "minimum_positive_fraction": 0.75,
    "maximum_negative_weight_fraction": 0.25,
    "maximum_control_holdout_loss_vs_w409c": 0.00150,
    "minimum_profile_holdout_gain_vs_control": -0.00050,
}


def load(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def audit(tag: str, suffix: str):
    value = load(WORK / f"{tag}_{suffix}.json")
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"expected one audit row for {tag}_{suffix}")
    return value[0]


def candidate_negative_fraction(row: dict, tag: str) -> float:
    value = row["candidate_weight_negative_fraction"]
    if isinstance(value, (int, float)):
        return float(value)
    matches = [
        float(fraction)
        for path, fraction in value.items()
        if Path(path).name == f"{tag}_val.npy"
    ]
    if len(matches) != 1:
        raise ValueError(f"cannot identify candidate weight for {tag}: {value}")
    return matches[0]


w409c = load(WORK / "w409c_ridge96_current.json")
if not isinstance(w409c, list) or len(w409c) != 1:
    raise ValueError("invalid w409c baseline audit")
w409c_gain = float(w409c[0]["mean_independent_private_gain"])

report = {
    "uses_public_scores": False,
    "uses_recovered_moments": False,
    "uses_competition_target_mean": False,
    "selection_rule": "promote every family passing identical frozen gates on both seeds",
    "seeds": list(SEEDS),
    "thresholds": THRESHOLDS,
    "diagnostic_thresholds": DIAGNOSTIC_THRESHOLDS,
    "w409c_mean_independent_private_gain": w409c_gain,
    "families": {},
    "eligible_families": [],
    "diagnostic_families": [],
    "full_refit_families": [],
}

for family, seed_tags in FAMILIES.items():
    family_rows = []
    for seed in SEEDS:
        tag = seed_tags[seed]
        single = audit(tag, "ridge96")
        joint = audit(tag, "w409c_joint96")
        model_report = load(WORK / f"{tag}_report.json")
        control_tag = FAMILIES["control"][seed]
        control_report = load(WORK / f"{control_tag}_report.json")

        single_gain = float(single["mean_independent_private_gain"])
        incremental_gain = (
            float(joint["mean_independent_private_gain"]) - w409c_gain
        )
        positive_fraction = float(single["private_gain_positive_fraction"])
        single_negative = candidate_negative_fraction(single, tag)
        joint_negative = candidate_negative_fraction(joint, tag)
        holdout_score = float(model_report["holdout_private_score"])
        control_score = float(control_report["holdout_private_score"])

        checks = {
            "single_gain": single_gain >= THRESHOLDS["minimum_single_independent_gain"],
            "incremental_beyond_w409c": incremental_gain
            >= THRESHOLDS["minimum_incremental_gain_beyond_w409c"],
            "positive_fraction": positive_fraction
            >= THRESHOLDS["minimum_positive_fraction"],
            "single_weight_sign": single_negative
            <= THRESHOLDS["maximum_negative_weight_fraction"],
            "joint_weight_sign": joint_negative
            <= THRESHOLDS["maximum_negative_weight_fraction"],
        }
        if family == "control":
            w409c_score = float(model_report["w409c_same_split_score"])
            holdout_margin = w409c_score - holdout_score
            checks["holdout"] = holdout_margin >= -THRESHOLDS[
                "maximum_control_holdout_loss_vs_w409c"
            ]
        else:
            holdout_margin = control_score - holdout_score
            checks["holdout"] = holdout_margin >= THRESHOLDS[
                "minimum_profile_holdout_gain_vs_control"
            ]

        diagnostic_checks = {
            "single_gain": single_gain
            >= DIAGNOSTIC_THRESHOLDS["minimum_single_independent_gain"],
            "incremental_beyond_w409c": incremental_gain
            >= DIAGNOSTIC_THRESHOLDS["minimum_incremental_gain_beyond_w409c"],
            "positive_fraction": positive_fraction
            >= DIAGNOSTIC_THRESHOLDS["minimum_positive_fraction"],
            "single_weight_sign": single_negative
            <= DIAGNOSTIC_THRESHOLDS["maximum_negative_weight_fraction"],
            "joint_weight_sign": joint_negative
            <= DIAGNOSTIC_THRESHOLDS["maximum_negative_weight_fraction"],
        }
        if family == "control":
            diagnostic_checks["holdout"] = holdout_margin >= -DIAGNOSTIC_THRESHOLDS[
                "maximum_control_holdout_loss_vs_w409c"
            ]
        else:
            diagnostic_checks["holdout"] = holdout_margin >= DIAGNOSTIC_THRESHOLDS[
                "minimum_profile_holdout_gain_vs_control"
            ]

        family_rows.append({
            "seed": seed,
            "tag": tag,
            "single_independent_gain": single_gain,
            "incremental_gain_beyond_w409c": incremental_gain,
            "positive_fraction": positive_fraction,
            "single_negative_weight_fraction": single_negative,
            "joint_negative_weight_fraction": joint_negative,
            "holdout_private_score": holdout_score,
            "same_seed_control_private_score": control_score,
            "holdout_margin": holdout_margin,
            "checks": checks,
            "passed": all(checks.values()),
            "diagnostic_checks": diagnostic_checks,
            "diagnostic_passed": all(diagnostic_checks.values()),
        })
    passed = all(row["passed"] for row in family_rows)
    diagnostic_passed = all(row["diagnostic_passed"] for row in family_rows)
    tier = "strict" if passed else "diagnostic" if diagnostic_passed else "reject"
    report["families"][family] = {
        "seeds": family_rows,
        "passed": passed,
        "diagnostic_passed": diagnostic_passed,
        "tier": tier,
    }
    if passed:
        report["eligible_families"].append(family)
    elif diagnostic_passed:
        report["diagnostic_families"].append(family)
    if tier != "reject":
        report["full_refit_families"].append(family)

report["decision"] = (
    "run two-seed full refits for every strict or diagnostic family"
    if report["full_refit_families"]
    else "no family cleared even the frozen diagnostic gates; no full refit"
)
(WORK / "profile_promotion_decision.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
