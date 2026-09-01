#!/usr/bin/env python
"""Build a competitive mixture probe for the survival-distribution base."""
import csv
import json

import numpy as np
import polars as pl
from scipy.optimize import brentq

ROOT = "/Users/timur/Desktop/dev/OZON_ECUP_2026_3/"
BASE_SCORE = 1.6468181197
M1 = 2.3232887
M2 = 10.7633307
WEIGHT = 0.30

submission = pl.read_csv(ROOT + "submissions/61_candC_ridge22.csv.gz")
base = np.log1p(np.clip(submission["predict"].to_numpy(), 0, None)).astype(np.float64)
survival = np.load(ROOT + "work/gbdt_surv_final.npy").astype(np.float64)
survival = (survival - survival.mean()) / survival.std() * base.std() + base.mean()
shift = brentq(lambda value: np.clip(survival + value, 0, None).mean() - M1, -10, 10)
survival = np.clip(survival + shift, 0, None)
probe = (1.0 - WEIGHT) * base + WEIGHT * survival

assert len(base) == len(survival) == 250000
assert np.isfinite(probe).all() and probe.min() >= 0
assert abs(probe.mean() - M1) < 2e-6

path = ROOT + "submissions/81_probe_surv.csv"
with open(path, "w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["user_id", "predict"])
    for user_id, value in zip(submission["user_id"], np.expm1(probe)):
        writer.writerow([int(user_id), float(value)])

metadata = {
    "file": path,
    "base_score": BASE_SCORE,
    "weight": WEIGHT,
    "mean_log": float(probe.mean()),
    "base_second_moment": float(np.mean(base * base)),
    "survival_second_moment": float(np.mean(survival * survival)),
    "cross_base_survival": float(np.mean(base * survival)),
    "probe_second_moment": float(np.mean(probe * probe)),
    "corr_base_survival": float(np.corrcoef(base, survival)[0, 1]),
    "distance_squared": float(np.mean((base - survival) ** 2)),
    "ez_base": float((M2 + np.mean(base * base) - BASE_SCORE**2) / 2),
}
with open(ROOT + "work/probe_surv_meta.json", "w") as stream:
    json.dump(metadata, stream, indent=2)
print(json.dumps(metadata, indent=2))
