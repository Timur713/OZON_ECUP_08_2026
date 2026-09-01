#!/usr/bin/env python
"""Create a fixed equal-weight, per-seed-standardized final vector."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("family")
parser.add_argument("seed1310")
parser.add_argument("seed2718")
args = parser.parse_args()
root = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[1]))
work = Path(os.environ.get("ECUP_OUT", root / "work"))

vectors = []
inputs = []
for value in (args.seed1310, args.seed2718):
    path = Path(value)
    vector = np.load(path).astype(np.float64)
    if vector.shape != (250_000,) or not np.isfinite(vector).all():
        raise ValueError(f"invalid final vector {path}: {vector.shape}")
    standard_deviation = float(vector.std())
    if standard_deviation <= 0:
        raise ValueError(f"constant final vector {path}")
    vectors.append((vector - vector.mean()) / standard_deviation)
    inputs.append({
        "file": str(path),
        "mean": float(vector.mean()),
        "std": standard_deviation,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })

output = work / f"promote_{args.family}_seedavg.npy"
average = np.mean(vectors, axis=0, dtype=np.float64)
np.save(output, average)
report = {
    "family": args.family,
    "construction": "equal average of two independently trained standardized seed vectors",
    "uses_public_scores": False,
    "inputs": inputs,
    "rows": int(len(average)),
    "mean": float(average.mean()),
    "std": float(average.std()),
    "output": str(output),
    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
}
(work / f"promote_{args.family}_seedavg_report.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
