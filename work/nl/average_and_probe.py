#!/usr/bin/env python
"""177 - average a family of new bases into ONE column and screen it.

The pool already contains averaged bases: W45 is the mean of w45a..d, W60 of
w60a..c, and so on. Averaging a family is therefore the project's own
established construction, not a new liberty. It matters here because probes are
the scarce resource: five individually marginal siblings cost five submissions,
while their average costs one and one degree of freedom.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIV = ROOT / "work" / "div"
OUT = ROOT / "work" / "nl"

parser = argparse.ArgumentParser()
parser.add_argument("name", help="family name, e.g. divB")
parser.add_argument("members", nargs="+", help="tags to average")
args = parser.parse_args()

for suffix in ("val", "final"):
    vectors = []
    missing = []
    for tag in args.members:
        path = DIV / f"{tag}_{suffix}.npy"
        if path.exists():
            vectors.append(np.load(path).astype(np.float64))
        else:
            missing.append(tag)
    if missing:
        print(f"{suffix}: missing {missing}")
    if not vectors:
        raise SystemExit(f"no members found for {suffix}")
    # Each member is calibrated to its own scale before averaging, otherwise a
    # single badly-scaled sibling dominates the mean.
    standard = [(v - v.mean()) / (v.std() + 1e-12) for v in vectors]
    mean = np.mean(standard, axis=0)
    reference = vectors[0]
    mean = mean * reference.std() + reference.mean()
    np.save(DIV / f"{args.name}_avg_{suffix}.npy", mean.astype(np.float32))
    print(f"{suffix}: averaged {len(vectors)} members -> "
          f"{args.name}_avg_{suffix}.npy mean={mean.mean():.4f} sd={mean.std():.4f}")
    if suffix == "val":
        correlations = [
            float(np.corrcoef(standard[i], standard[j])[0, 1])
            for i in range(len(standard)) for j in range(i + 1, len(standard))
        ]
        if correlations:
            print(f"  pairwise member correlation: min={min(correlations):.5f} "
                  f"mean={np.mean(correlations):.5f} max={max(correlations):.5f}")
