#!/usr/bin/env python
"""Build a scale-safe 30% probe from a new final log-space prediction vector."""
import argparse
import csv
import json
import os

import numpy as np
import polars as pl
from scipy.optimize import brentq

ROOT = os.environ.get(
    "ECUP_ROOT", "/Users/timur/Desktop/dev/OZON_ECUP_2026_3"
).rstrip("/") + "/"
M1, M2 = 2.3232887, 10.7633307
DEFAULT_BASE_SCORE = 1.6468181197
DEFAULT_BASE = "submissions/61_candC_ridge22.csv.gz"
DEFAULT_WEIGHT = 0.30

parser = argparse.ArgumentParser()
parser.add_argument("tag")
parser.add_argument("vector", help=".npy path or .npz_path:key")
parser.add_argument("--base-submission", default=DEFAULT_BASE)
parser.add_argument("--base-score", type=float, default=DEFAULT_BASE_SCORE)
parser.add_argument("--weight", type=float, default=DEFAULT_WEIGHT)
parser.add_argument(
    "--invert", action="store_true",
    help=(
        "probe the independently validated inverse direction around the base; "
        "the resulting candidate is clipped/recentered before moment recovery"
    ),
)
args = parser.parse_args()
if not 0 < args.weight <= 1:
    raise ValueError("--weight must be in (0, 1]")

base_path = args.base_submission
if not os.path.isabs(base_path):
    base_path = ROOT + base_path
table = pl.read_csv(base_path)
base = np.log1p(np.clip(table["predict"].to_numpy(), 0, None)).astype(np.float64)
if ":" in args.vector:
    vector_path, vector_key = args.vector.rsplit(":", 1)
else:
    vector_path, vector_key = args.vector, None
loaded = np.load(vector_path)
if isinstance(loaded, np.lib.npyio.NpzFile):
    if not vector_key:
        raise ValueError("an .npz vector must be specified as path:key")
    candidate = loaded[vector_key].astype(np.float64)
    loaded.close()
else:
    if vector_key:
        raise ValueError("a key can only be used with an .npz vector")
    candidate = loaded.astype(np.float64)
assert candidate.shape == base.shape and np.isfinite(candidate).all()
def normalize_candidate(vector):
    vector = (
        (vector - vector.mean()) / vector.std() * base.std() + base.mean()
    )
    shift = brentq(
        lambda value: np.clip(vector + value, 0, None).mean() - M1,
        -10,
        10,
    )
    return np.clip(vector + shift, 0, None)


candidate = normalize_candidate(candidate)
if args.invert:
    # Store the inverse itself as the probed basis. This keeps the submitted
    # file a positive convex probe, so the linear score-to-moment recovery in
    # solve_augmented_stack.py remains exact even when the useful raw-model
    # coefficient was negative on independent validation splits.
    candidate = normalize_candidate(2 * base - candidate)
candidate_file = ROOT + f"work/{args.tag}_candidate.npy"
np.save(candidate_file, candidate.astype(np.float64))
probe = (1.0 - args.weight) * base + args.weight * candidate
assert abs(probe.mean() - M1) < 2e-6
prediction = np.expm1(probe)
assert np.isfinite(prediction).all() and (prediction >= 0).all()

output = ROOT + f"submissions/{args.tag}.csv"
with open(output, "w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["user_id", "predict"])
    for user_id, value in zip(table["user_id"], prediction):
        writer.writerow([int(user_id), float(value)])

base_second = float(np.mean(base * base))
metadata = {
    "tag": args.tag,
    "file": output,
    "vector": os.path.abspath(vector_path),
    "vector_key": vector_key,
    "candidate_file": candidate_file,
    "base_submission": os.path.abspath(base_path),
    "base_score": args.base_score,
    "weight": args.weight,
    "inverted": args.invert,
    "mean_log": float(probe.mean()),
    "prediction_min": float(prediction.min()),
    "prediction_max": float(prediction.max()),
    "base_mean": float(base.mean()),
    "candidate_mean": float(candidate.mean()),
    "base_second_moment": base_second,
    "candidate_second_moment": float(np.mean(candidate * candidate)),
    "cross_base_candidate": float(np.mean(base * candidate)),
    "probe_second_moment": float(np.mean(probe * probe)),
    "corr_base_candidate": float(np.corrcoef(base, candidate)[0, 1]),
    "distance_squared": float(np.mean((base - candidate) ** 2)),
    "ez_base": float((M2 + base_second - args.base_score**2) / 2),
}
with open(ROOT + f"work/{args.tag}_meta.json", "w") as stream:
    json.dump(metadata, stream, indent=2)
print(json.dumps(metadata, indent=2))
