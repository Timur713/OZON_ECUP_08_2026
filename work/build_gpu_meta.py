#!/usr/bin/env python
"""Combine several GPU prediction components into one probeable super-base."""
import argparse
import json
import os

import numpy as np


def load_vector(spec):
    if ":" in spec:
        path, key = spec.rsplit(":", 1)
    else:
        path, key = spec, None
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if not key:
            raise ValueError(f"missing npz key in {spec!r}")
        vector = loaded[key].astype(np.float64)
        loaded.close()
    else:
        if key:
            raise ValueError(f"unexpected key for npy vector in {spec!r}")
        vector = loaded.astype(np.float64)
    if vector.shape != (250000,) or not np.isfinite(vector).all():
        raise ValueError(f"invalid prediction vector {spec!r}: {vector.shape}")
    return vector


parser = argparse.ArgumentParser()
parser.add_argument("output", help="output .npy path")
parser.add_argument("vectors", nargs="+")
parser.add_argument("--weights", help="comma-separated weights; default is equal")
parser.add_argument(
    "--raw",
    action="store_true",
    help="blend heads in their native common scale (for validated component mixes)",
)
args = parser.parse_args()

vectors = [load_vector(spec) for spec in args.vectors]
if args.weights:
    weights = np.array([float(value) for value in args.weights.split(",")])
    if len(weights) != len(vectors):
        raise ValueError("number of weights does not match number of vectors")
else:
    weights = np.ones(len(vectors), dtype=np.float64)
if not np.isfinite(weights).all() or abs(weights.sum()) < 1e-12:
    raise ValueError("invalid weights")
weights /= weights.sum()

# Independent models are standardized so no model wins merely because its raw
# head has a larger numerical range. Heads from one checkpoint share a native
# scale; --raw preserves a hurdle/direct mix selected on validation.
standardized = np.stack([(value - value.mean()) / value.std() for value in vectors])
blend_matrix = np.stack(vectors) if args.raw else standardized
meta = weights @ blend_matrix
np.save(args.output, meta.astype(np.float32))

report = {
    "output": os.path.abspath(args.output),
    "vectors": args.vectors,
    "weights": weights.tolist(),
    "raw_blend": args.raw,
    "correlations": np.corrcoef(standardized).round(6).tolist(),
    "component_means": [float(value.mean()) for value in vectors],
    "component_stds": [float(value.std()) for value in vectors],
    "meta_mean": float(meta.mean()),
    "meta_std": float(meta.std()),
}
with open(os.path.splitext(args.output)[0] + "_meta.json", "w") as stream:
    json.dump(report, stream, indent=2)
print(json.dumps(report, indent=2))
