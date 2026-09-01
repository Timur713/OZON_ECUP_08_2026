#!/usr/bin/env python
"""Sweep ridge lambda against the frozen private-risk objective.

Uses only moments that were already measured for the current primary pool.  No
new public score is read and no subset search is performed: the single free
knob is regularisation strength, scored by the same
`expected_public + degrees * empirical_transfer_penalty_per_degree` objective
that solve_augmented_stack.py already reports.
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
M1, M2 = 2.3232887, 10.7633307
ONE_SIDED_PER_DEGREE = 0.0000165
THEORETICAL_PER_DEGREE = 2 * ONE_SIDED_PER_DEGREE
EMPIRICAL_PER_DEGREE = 0.0000395

parser = argparse.ArgumentParser()
parser.add_argument(
    "--extra", nargs=2, action="append", default=[], metavar=("TAG", "SCORE")
)
parser.add_argument("--output", default=ROOT + "work/private_optimal_lambda_sweep.json")
args = parser.parse_args()

candidates = {}
candidate_moments = {}
for tag, score in args.extra:
    score = float(score)
    metadata = json.load(open(ROOT + f"work/{tag}_meta.json"))
    weight = metadata["weight"]
    ez_probe = (M2 + metadata["probe_second_moment"] - score**2) / 2
    candidates[tag] = np.load(metadata["candidate_file"]).astype(np.float64)
    candidate_moments[tag] = (
        ez_probe - (1 - weight) * metadata["ez_base"]
    ) / weight

ez_pool = json.load(open(ROOT + "work/EZ_pool.json"))
load = lambda name: np.load(ROOT + f"work/{name}_final.npy").astype(np.float64)
pool = {
    "gb": (load("v4_zh") + load("cfg3")) / 2,
    "tcn45": load("tcn45"),
    "tcn90": load("tcn90"),
    "tcn180two": load("tcn180two"),
    "tcn270": load("tcn270"),
    "tcn409": load("tcn409"),
    "tcn365v336": load("tcn365v336"),
    "t3b": load("tcn365b"),
    "t1": load("seq"),
    "gru180": load("gru180"),
    "tcn365": load("tcn365"),
    "a409a": load("a409a"),
    "LY": np.load(ROOT + "work/basis_prior_year_gmv.npy").astype(np.float64),
}
for name in ("GBD", "W120", "W150", "W365", "W409", "W90", "W45", "W60", "W180", "W270"):
    pool[name] = np.load(ROOT + f"work/AVG_{name}.npy").astype(np.float64)
ridge_keys = set(np.load(ROOT + "work/ridge22_keys.npy").tolist())
pool = {key: value for key, value in pool.items() if key in ridge_keys}
pool.update(candidates)

keys = sorted(pool)
reference = next(iter(candidates.values()))
design = np.vstack([pool[key] for key in keys] + [np.ones_like(reference)]).T
gram = design.T @ design / len(reference)
rhs = np.array(
    [candidate_moments[key] if key in candidate_moments else ez_pool[key] for key in keys]
    + [M1]
)

rows = []
for lam in np.concatenate([
    np.array([0.0]),
    np.geomspace(1e-4, 3.0, 121),
]):
    penalty = np.eye(len(keys) + 1) * lam
    penalty[-1, -1] = 0
    system = gram + penalty
    coefficients = np.linalg.solve(system, rhs)
    mse = M2 - 2 * rhs @ coefficients + coefficients @ gram @ coefficients
    expected = float(np.sqrt(max(mse, 0)))
    degrees = float(np.trace(gram @ np.linalg.inv(system)))
    rows.append({
        "lambda": float(lam),
        "expected_public": expected,
        "degrees_of_freedom": degrees,
        "theoretical_private": expected + degrees * THEORETICAL_PER_DEGREE,
        "empirical_private": expected + degrees * EMPIRICAL_PER_DEGREE,
    })

best_empirical = min(rows, key=lambda row: row["empirical_private"])
best_theoretical = min(rows, key=lambda row: row["theoretical_private"])
report = {
    "pool_candidates": sorted(candidates),
    "model_count": len(keys),
    "empirical_transfer_penalty_per_degree": EMPIRICAL_PER_DEGREE,
    "theoretical_transfer_penalty_per_degree": THEORETICAL_PER_DEGREE,
    "private_optimal_lambda_empirical": best_empirical,
    "private_optimal_lambda_theoretical": best_theoretical,
    "sweep": rows,
}
json.dump(report, open(args.output, "w"), indent=2)
print(json.dumps({k: v for k, v in report.items() if k != "sweep"}, indent=2))
