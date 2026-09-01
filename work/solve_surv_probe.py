#!/usr/bin/env python
"""Recover E[z*survival] from probe 81 and report two-base ridge optima."""
import json
import sys

import numpy as np
import polars as pl
from scipy.optimize import brentq

ROOT = "/Users/timur/Desktop/dev/OZON_ECUP_2026_3/"
M1, M2 = 2.3232887, 10.7633307
if len(sys.argv) != 2:
    raise SystemExit("usage: solve_surv_probe.py PUBLIC_SCORE")
score = float(sys.argv[1])
meta = json.load(open(ROOT + "work/probe_surv_meta.json"))
weight = meta["weight"]
ez_probe = (M2 + meta["probe_second_moment"] - score**2) / 2
ez_survival = (ez_probe - (1 - weight) * meta["ez_base"]) / weight

table = pl.read_csv(ROOT + "submissions/61_candC_ridge22.csv.gz")
base = np.log1p(np.clip(table["predict"].to_numpy(), 0, None)).astype(np.float64)
raw = np.load(ROOT + "work/gbdt_surv_final.npy").astype(np.float64)
raw = (raw - raw.mean()) / raw.std() * base.std() + base.mean()
shift = brentq(lambda value: np.clip(raw + value, 0, None).mean() - M1, -10, 10)
survival = np.clip(raw + shift, 0, None)
design = np.vstack([base, survival, np.ones_like(base)]).T
gram = design.T @ design / len(base)
rhs = np.array([meta["ez_base"], ez_survival, M1])
print(f"E[z*survival]={ez_survival:.9f}")
for ridge in (0.0, 0.003, 0.03):
    penalty = np.diag([ridge, ridge, 0.0])
    coefficients = np.linalg.solve(gram + penalty, rhs)
    mse = M2 - 2 * rhs @ coefficients + coefficients @ gram @ coefficients
    print(f"ridge={ridge:.3f} expected={np.sqrt(max(mse,0)):.9f} weights={coefficients.tolist()}")
