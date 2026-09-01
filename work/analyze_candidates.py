#!/usr/bin/env python
"""Measure candidate diversity against the current prediction and ridge basis."""
import argparse
import json
import os

import numpy as np
import polars as pl
from scipy.optimize import brentq

ROOT = "/Users/timur/Desktop/dev/OZON_ECUP_2026_3/"
M1 = 2.3232887


def load_vector(spec):
    if ":" in spec:
        path, key = spec.rsplit(":", 1)
    else:
        path, key = spec, None
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if not key:
            raise ValueError(f"missing key for {spec}")
        vector = loaded[key].astype(np.float64)
        loaded.close()
        return vector
    if key:
        raise ValueError(f"unexpected key for {spec}")
    return loaded.astype(np.float64)


def calibrate(vector, reference):
    vector = (vector - vector.mean()) / vector.std() * reference.std() + reference.mean()
    shift = brentq(lambda value: np.clip(vector + value, 0, None).mean() - M1, -10, 10)
    return np.clip(vector + shift, 0, None)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--reference-submission",
    default="submissions/95_provisional_ridge27_l003.csv",
)
parser.add_argument("vectors", nargs="+")
args = parser.parse_args()

reference_path = args.reference_submission
if not reference_path.startswith("/"):
    reference_path = ROOT + reference_path
table = pl.read_csv(reference_path)
current = np.log1p(np.clip(table["predict"].to_numpy(), 0, None)).astype(np.float64)
load = lambda name: np.load(ROOT + f"work/{name}_final.npy").astype(np.float64)
pool = {
    "gb": (load("v4_zh") + load("cfg3")) / 2,
    "tcn45": load("tcn45"), "tcn90": load("tcn90"),
    "tcn180two": load("tcn180two"), "tcn270": load("tcn270"),
    "tcn409": load("tcn409"), "tcn365v336": load("tcn365v336"),
    "t3b": load("tcn365b"), "t1": load("seq"), "gru180": load("gru180"),
    "tcn365": load("tcn365"), "LY": np.load(ROOT + "work/basis_prior_year_gmv.npy"),
}
for name in ("GBD", "W120", "W150", "W365", "W409", "W90", "W45", "W60", "W180", "W270"):
    pool[name] = np.load(ROOT + f"work/AVG_{name}.npy").astype(np.float64)
keys = np.load(ROOT + "work/ridge22_keys.npy").tolist()
measured_path = ROOT + "work/measured_candidates.json"
measured = json.load(open(measured_path)) if os.path.exists(measured_path) else {}
for tag in measured:
    metadata = json.load(open(ROOT + f"work/{tag}_meta.json"))
    pool[tag] = np.load(metadata["candidate_file"]).astype(np.float64)
keys = keys + list(measured)
design = np.vstack([pool[key] for key in keys] + [np.ones_like(current)]).T
gram = design.T @ design / len(current)
penalty = np.eye(len(keys) + 1) * 0.001
penalty[-1, -1] = 0
system = gram + penalty

recent = {}
for name in ("gbdt_surv_final.npy", "gbdt_seasw_final.npy", "tcn409rep_server_final.npy"):
    try:
        recent[name] = calibrate(np.load(ROOT + "work/" + name), current)
    except FileNotFoundError:
        pass

report = []
for spec in args.vectors:
    vector = calibrate(load_vector(spec), current)
    cross = design.T @ vector / len(vector)
    coefficients = np.linalg.solve(system, cross)
    residual = vector - design @ coefficients
    row = {
        "vector": spec,
        "reference_submission": reference_path,
        "ridge_span_columns": len(keys),
        "corr_current": float(np.corrcoef(vector, current)[0, 1]),
        "distance2_current": float(np.mean((vector - current) ** 2)),
        "ridge_span_residual2": float(np.mean(residual**2)),
        "ridge_span_r2": float(1 - np.var(residual) / np.var(vector)),
        "mean": float(vector.mean()),
        "std": float(vector.std()),
        "corr_recent": {
            name: float(np.corrcoef(vector, other)[0, 1]) for name, other in recent.items()
        },
    }
    report.append(row)
print(json.dumps(report, indent=2))
