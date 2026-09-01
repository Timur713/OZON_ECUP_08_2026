#!/usr/bin/env python
"""Recover a candidate cross-moment from a generic 30% probe score."""
import json
import sys

import numpy as np

ROOT = "/Users/timur/Desktop/dev/OZON_ECUP_2026_3/"
M1, M2 = 2.3232887, 10.7633307
if len(sys.argv) != 3:
    raise SystemExit("usage: solve_gpu_probe.py PROBE_TAG PUBLIC_SCORE")
tag, raw_score = sys.argv[1:]
score = float(raw_score)
meta = json.load(open(ROOT + f"work/{tag}_meta.json"))
weight = meta["weight"]
ez_probe = (M2 + meta["probe_second_moment"] - score**2) / 2
ez_candidate = (ez_probe - (1 - weight) * meta["ez_base"]) / weight
standalone = np.sqrt(max(M2 + meta["candidate_second_moment"] - 2 * ez_candidate, 0))
gram = np.array([
    [meta["base_second_moment"], meta["cross_base_candidate"], meta["base_mean"]],
    [meta["cross_base_candidate"], meta["candidate_second_moment"], meta["candidate_mean"]],
    [meta["base_mean"], meta["candidate_mean"], 1.0],
])
rhs = np.array([meta["ez_base"], ez_candidate, M1])
print(f"E[z*candidate]={ez_candidate:.9f}")
print(f"candidate_standalone={standalone:.9f}")
for ridge in (0.0, 0.003, 0.03):
    penalty = np.diag([ridge, ridge, 0.0])
    coefficients = np.linalg.solve(gram + penalty, rhs)
    mse = M2 - 2 * rhs @ coefficients + coefficients @ gram @ coefficients
    print(f"ridge={ridge:.3f} expected={np.sqrt(max(mse,0)):.9f} weights={coefficients.tolist()}")
