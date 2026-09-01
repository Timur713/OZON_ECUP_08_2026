#!/usr/bin/env python
"""Compare unconstrained and nonnegative final-solver risk on 50k/200k splits.

This is a sampling-risk proxy, not a time-domain estimate of the competition
private score.  It mirrors the two frozen solvers on the validation vectors
that have exact local counterparts, including the admitted w409c branch.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"

parser = argparse.ArgumentParser()
parser.add_argument("--repeats", type=int, default=48)
parser.add_argument("--public-users", type=int, default=50_000)
parser.add_argument("--seed", type=int, default=20260825)
args = parser.parse_args()
if args.repeats < 2 or args.public_users < 1:
    raise ValueError("invalid repeats or public-users")


def load(name):
    return np.load(WORK / name).astype(np.float64)


columns = {
    "gbdt262": load("v4_262_valpred.npy"),
    "gbdt159": load("gbdt_v5_val.npy"),
    "seq180": load("seq_val.npy"),
    "tcn45": load("tcn45_val.npy"),
    "tcn90": load("tcn90_val.npy"),
    "tcn180two": load("tcn180two_val.npy"),
    "tcn270": load("tcn270_val.npy"),
    "tcn365": load("tcn365_val.npy"),
    "tcn365b": load("tcn365b_val.npy"),
    "tcn365v336": load("tcn365v336_val.npy"),
    "tcn409": load("tcn409_val.npy"),
    "gru180": load("gru180_val.npy"),
    "W45": np.mean([load(f"w45{s}_val.npy") for s in "abcd"], axis=0),
    "W60": np.mean([load(f"w60{s}_val.npy") for s in "abc"], axis=0),
    "W90": np.mean([load(f"w90{s}_val.npy") for s in "abc"], axis=0),
    "W120": np.mean([load(f"w120{s}_val.npy") for s in "abc"], axis=0),
    "W150": load("w150a_val.npy"),
    "W180": np.mean([load(f"w180{s}_val.npy") for s in "ab"], axis=0),
    "W210": load("w210a_val.npy"),
    "W270": load("w270a_val.npy"),
    "W300": load("w300a_val.npy"),
    "W365": np.mean([load(f"w365{s}_val.npy") for s in "ab"], axis=0),
    "W409": load("w409a_val.npy"),
    "cls300": load("cls300_val_server_val.npy"),
    "cls409": load("cls409_val_server_val.npy"),
    "w409c": load("w409c_val.npy"),
}
# The original cls300 probability validation vector was not retained.  Use the
# later calendar-equivalent head only as a constraint-risk proxy and name it
# explicitly so the report cannot be mistaken for an exact final reconstruction.
with np.load(WORK / "cls300cal_val_server_best_val_components.npz") as components:
    columns["cls300cal_probability_proxy"] = components["probability"].astype(np.float64)

gmv = np.load(WORK / "mat" / "gmv.npy", mmap_mode="r")
truth = np.log1p(gmv[:, 379:409].sum(axis=1, dtype=np.float64))
design = np.column_stack([*columns.values(), np.ones_like(truth)])
if design.shape[0] != len(truth) or not np.isfinite(design).all():
    raise ValueError("invalid design")
if args.public_users >= len(truth):
    raise ValueError("public-users must be smaller than the population")


def system(index, lam):
    x = design[index]
    y = truth[index]
    gram = x.T @ x / len(index)
    rhs = x.T @ y / len(index)
    penalty = np.eye(design.shape[1]) * lam
    penalty[-1, -1] = 0.0
    return gram, rhs, penalty


def solve(index, lam, nonnegative):
    gram, rhs, penalty = system(index, lam)
    matrix = gram + penalty
    unconstrained = np.linalg.solve(matrix, rhs)
    if not nonnegative:
        return unconstrained
    initial = unconstrained.copy()
    initial[:-1] = np.maximum(initial[:-1], 0.0)
    initial[-1] = (rhs[-1] - gram[-1, :-1] @ initial[:-1]) / gram[-1, -1]

    def objective(value):
        return 0.5 * value @ matrix @ value - rhs @ value

    def gradient(value):
        return matrix @ value - rhs

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * (len(initial) - 1) + [(None, None)],
        options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 10_000, "maxls": 50},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(result.message)
    return result.x


def score(index, weights):
    residual = truth[index] - design[index] @ weights
    return float(np.sqrt(np.mean(residual * residual)))


solvers = {
    "ridge_l003": (0.003, False),
    "nonnegative_l001": (0.001, True),
}
all_index = np.arange(len(truth))
oracle = {
    name: solve(all_index, lam, nonnegative)
    for name, (lam, nonnegative) in solvers.items()
}
rng = np.random.default_rng(args.seed)
rows = []
for _ in range(args.repeats):
    public = rng.choice(len(truth), args.public_users, replace=False)
    mask = np.ones(len(truth), dtype=bool)
    mask[public] = False
    private = all_index[mask]
    row = {}
    for name, (lam, nonnegative) in solvers.items():
        weights = solve(public, lam, nonnegative)
        row[name] = {
            "public": score(public, weights),
            "private": score(private, weights),
            "private_excess": score(private, weights) - score(private, oracle[name]),
            "negative_l1": float(np.maximum(-weights[:-1], 0).sum()),
            "active": int(np.sum(weights[:-1] > 1e-8)),
        }
    rows.append(row)

ridge_private = np.array([row["ridge_l003"]["private"] for row in rows])
nonnegative_private = np.array([
    row["nonnegative_l001"]["private"] for row in rows
])
delta = nonnegative_private - ridge_private
report = {
    "scope": "sampling-risk proxy on one historical target window; not a private-score forecast",
    "columns": list(columns),
    "model_columns": len(columns),
    "repeats": args.repeats,
    "public_users": args.public_users,
    "oracle_scores_250k": {
        name: score(all_index, weights) for name, weights in oracle.items()
    },
    "mean_scores": {
        name: {
            key: float(np.mean([row[name][key] for row in rows]))
            for key in ("public", "private", "private_excess", "negative_l1", "active")
        }
        for name in solvers
    },
    "nonnegative_minus_ridge_private": {
        "mean": float(delta.mean()),
        "se": float(delta.std(ddof=1) / np.sqrt(args.repeats)),
        "median": float(np.median(delta)),
        "p10": float(np.quantile(delta, 0.10)),
        "p90": float(np.quantile(delta, 0.90)),
        "nonnegative_win_fraction": float(np.mean(delta < 0)),
    },
}
print(json.dumps(report, indent=2))
