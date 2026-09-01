#!/usr/bin/env python
"""Score saved classifier heads on a known rolling-origin target window."""
import argparse
import json
import os

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("components", help="NPZ with combined/hurdle/direct heads")
parser.add_argument("--anchor", type=int, default=378)
parser.add_argument("--horizon", type=int, default=30)
parser.add_argument(
    "--gmv",
    default=os.path.join(
        os.environ.get(
            "ECUP_MAT", "/Users/timur/Desktop/dev/OZON_ECUP_2026_3/work/mat"
        ),
        "gmv.npy",
    ),
)
parser.add_argument("--mix-step", type=float, default=0.05)
parser.add_argument(
    "--users",
    help="optional NPY of user-row indices for a fixed user holdout",
)
args = parser.parse_args()
if args.horizon < 1 or not 0 < args.mix_step <= 1:
    raise ValueError("invalid horizon or mix step")

components = np.load(args.components)
required = {"combined", "hurdle", "direct", "probability", "magnitude"}
if not required.issubset(components.files):
    raise ValueError(f"missing component keys: {sorted(required-set(components.files))}")
gmv = np.load(args.gmv, mmap_mode="r")
start = args.anchor + 1
stop = start + args.horizon
if stop > gmv.shape[1]:
    raise ValueError("target window falls outside the observed matrix")
truth = np.log1p(
    np.asarray(gmv[:, start:stop], dtype=np.float64).sum(axis=1)
)
if args.users:
    user_rows = np.load(args.users)
    if user_rows.ndim != 1 or not np.issubdtype(user_rows.dtype, np.integer):
        raise ValueError("--users must contain one-dimensional integer row indices")
    if len(user_rows) and (user_rows.min() < 0 or user_rows.max() >= len(truth)):
        raise ValueError("--users contains an out-of-range row index")
    truth = truth[user_rows]


def calibrated_score(prediction):
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError(f"invalid prediction: {prediction.shape}")
    design = np.column_stack([prediction, np.ones_like(prediction)])
    coefficients = np.linalg.lstsq(design, truth, rcond=None)[0]
    calibrated = np.clip(design @ coefficients, 0, None)
    score = float(np.sqrt(np.mean((truth - calibrated) ** 2)))
    return score, coefficients.tolist()


scores = {}
for key in sorted(required):
    score, coefficients = calibrated_score(components[key])
    scores[key] = {"score": score, "coefficients": coefficients}

hurdle = components["hurdle"].astype(np.float64)
direct = components["direct"].astype(np.float64)
mixes = []
for mix in np.arange(0, 1 + args.mix_step / 2, args.mix_step):
    mix = min(float(mix), 1.0)
    score, coefficients = calibrated_score(mix * hurdle + (1 - mix) * direct)
    mixes.append({"hurdle_weight": mix, "score": score, "coefficients": coefficients})
best_mix = min(mixes, key=lambda row: row["score"])

report = {
    "components": os.path.abspath(args.components),
    "gmv": os.path.abspath(args.gmv),
    "anchor": args.anchor,
    "horizon": args.horizon,
    "users": len(truth),
    "user_rows": os.path.abspath(args.users) if args.users else None,
    "component_scores": scores,
    "best_hurdle_direct_mix": best_mix,
    "mix_grid": mixes,
}
print(json.dumps(report, indent=2))
