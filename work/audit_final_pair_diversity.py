#!/usr/bin/env python
"""Reproducible label-free diversity audit for candidate final pairs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = ROOT / "submissions"
OUTPUT = ROOT / "work" / "final_pair_diversity_audit.json"
FILES = {
    "competition_quality": "200_shape_anchor_l003.csv",
    "competition_hedge": "147_private_safe_nonnegative_current_l001.csv",
    "120_clean_primary": "120_offline_rules_safe_meanforecast.csv",
    "122_clean_shift_reserve": "122_offline_diverse_no_replica.csv",
    "123_clean_capped_insurance": "123_offline_capped_w035.csv",
}
PAIRS = {
    "competition_selected": ("competition_quality", "competition_hedge"),
    "clean_preferred": ("120_clean_primary", "123_clean_capped_insurance"),
    "clean_severe_shift": ("120_clean_primary", "122_clean_shift_reserve"),
}


def load_submission(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.open().readline().strip() != "user_id,predict":
        raise ValueError(f"invalid submission header: {path}")
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if values.shape != (250_000, 2):
        raise ValueError(f"invalid submission shape/schema: {path}")
    user_id = values[:, 0].astype(np.int64)
    prediction = values[:, 1]
    if len(np.unique(user_id)) != len(user_id):
        raise ValueError(f"duplicate user_id: {path}")
    if not np.isfinite(prediction).all() or np.any(prediction < 0):
        raise ValueError(f"invalid predictions: {path}")
    return user_id, np.log1p(prediction)


def main() -> None:
    reference_ids = None
    predictions: dict[str, np.ndarray] = {}
    files: dict[str, dict[str, object]] = {}
    for role, filename in FILES.items():
        path = SUBMISSIONS / filename
        user_id, log_prediction = load_submission(path)
        if reference_ids is None:
            reference_ids = user_id
        elif not np.array_equal(reference_ids, user_id):
            raise ValueError(f"user order mismatch: {filename}")
        predictions[role] = log_prediction
        files[role] = {
            "filename": filename,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mean_log1p": float(log_prediction.mean()),
            "std_log1p": float(log_prediction.std()),
        }

    pairs = {}
    for scenario, (left_role, right_role) in PAIRS.items():
        left = predictions[left_role]
        right = predictions[right_role]
        delta = left - right
        pairs[scenario] = {
            "left": left_role,
            "right": right_role,
            "correlation": float(np.corrcoef(left, right)[0, 1]),
            "rms_log_distance": float(np.sqrt(np.mean(delta * delta))),
            "mean_log_delta": float(delta.mean()),
            "absolute_log_delta_quantiles": {
                key: float(value)
                for key, value in zip(
                    ("p50", "p90", "p99", "p999"),
                    np.quantile(np.abs(delta), (0.50, 0.90, 0.99, 0.999)),
                )
            },
        }

    report = {
        "uses_labels": False,
        "uses_public_scores": False,
        "rows": 250_000,
        "files": files,
        "pairs": pairs,
        "interpretation": {
            "competition_selected": (
                "selected quality/low-complexity pair; modest but larger separation "
                "than the previous competition pair"
            ),
            "clean_preferred": (
                "current quality-preserving clean pair; diversification is modest"
            ),
            "clean_severe_shift": (
                "more prediction diversity but historical quality cost makes it a reserve"
            ),
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
