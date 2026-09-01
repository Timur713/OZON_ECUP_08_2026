#!/usr/bin/env python
"""Frozen gates for exact-w409c position/event ablations, seed 93."""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[1]))
WORK = Path(os.environ.get("ECUP_OUT", ROOT / "work"))
THRESHOLDS = {
    "single_gain": 0.00012,
    "incremental_beyond_w409c": 0.00008,
    "positive_fraction": 0.90,
    "maximum_negative_weight_fraction": 0.10,
}


def one(name):
    rows = json.loads((WORK / name).read_text())
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(f"invalid audit {name}")
    return rows[0]


baseline_gain = float(one("w409c_ridge96_current.json")["mean_independent_private_gain"])
rows = []
for variant in ("position", "event"):
    tag = f"w409_exact_{variant}_s93"
    config = json.loads((WORK / f"{tag}_config.json").read_text())
    if config["seed"] != 93 or config["only_change_from_w409c"] != variant:
        raise ValueError(f"configuration drift for {tag}")
    single = one(f"{tag}_ridge96.json")
    joint = one(f"{tag}_w409c_joint96.json")
    single_gain = float(single["mean_independent_private_gain"])
    incremental = float(joint["mean_independent_private_gain"]) - baseline_gain
    positive = float(single["private_gain_positive_fraction"])
    single_negative = float(single["candidate_weight_negative_fraction"])
    joint_negative = float(joint["candidate_weight_negative_fraction"][str(WORK / f"{tag}_val.npy")])
    checks = {
        "single_gain": single_gain >= THRESHOLDS["single_gain"],
        "incremental_beyond_w409c": incremental >= THRESHOLDS["incremental_beyond_w409c"],
        "positive_fraction": positive >= THRESHOLDS["positive_fraction"],
        "single_weight_sign": single_negative <= THRESHOLDS["maximum_negative_weight_fraction"],
        "joint_weight_sign": joint_negative <= THRESHOLDS["maximum_negative_weight_fraction"],
    }
    rows.append({
        "variant": variant,
        "tag": tag,
        "single_independent_gain": single_gain,
        "incremental_gain_beyond_w409c": incremental,
        "positive_fraction": positive,
        "single_negative_weight_fraction": single_negative,
        "joint_negative_weight_fraction": joint_negative,
        "checks": checks,
        "passed": all(checks.values()),
    })

report = {
    "hypothesis": "position/cadence features can help when attached to the exact strong w409c backbone",
    "uses_public_scores": False,
    "seed": 93,
    "thresholds": THRESHOLDS,
    "variants": rows,
    "passed_variants": [row["variant"] for row in rows if row["passed"]],
}
(WORK / "w409_exact_structures_decision.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
