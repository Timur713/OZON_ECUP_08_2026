#!/usr/bin/env python
"""Stress-test fixed compact prefixes on 50k-fit -> independent-200k users.

This is a user-sampling/solver-risk audit, not an absolute forecast of the
competition private score: the available validation target is the January
anchor and does not reproduce the February/March seasonal shift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"


def load(name: str) -> np.ndarray:
    return np.load(WORK / name).astype(np.float64)


def validation_pool() -> dict[str, np.ndarray]:
    return {
        "102_probe_w409c": load("w409c_val.npy"),
        "tcn365": load("tcn365_val.npy"),
        # AVG_GBD is exactly gbdt_d262 on final.  The stored 262-feature
        # validation analogue is used here.
        "GBD": load("v4_262_valpred.npy"),
        "tcn180two": load("tcn180two_val.npy"),
        "tcn409": load("tcn409_val.npy"),
        "tcn365v336": load("tcn365v336_val.npy"),
        "92_probe_cls409_r26": load("cls409_val_server_val.npy"),
        "W120": np.mean([load(f"w120{seed}_val.npy") for seed in "abc"], axis=0),
        # The final `gb` averages v4 and cfg3.  No cfg3 validation vector was
        # persisted, so v4 is the declared conservative proxy.
        "gb": load("v4_262_valpred.npy"),
        "85_probe_w210a": load("w210a_val.npy"),
        "83_probe_cls300": load("cls300_val_server_val.npy"),
        "89_probe_w300a": load("w300a_val.npy"),
    }


def fit(design, truth, fit_index, score_index, lam):
    x = design[fit_index]
    y = truth[fit_index]
    gram = x.T @ x / len(fit_index)
    rhs = x.T @ y / len(fit_index)
    penalty = np.eye(design.shape[1]) * lam
    penalty[-1, -1] = 0
    system = gram + penalty
    weights = np.linalg.solve(system, rhs)
    residual = truth[score_index] - design[score_index] @ weights
    score = float(np.sqrt(np.mean(residual * residual)))
    degrees = float(np.trace(gram @ np.linalg.inv(system)))
    return score, degrees, weights


parser = argparse.ArgumentParser()
parser.add_argument("--lam", type=float, default=0.003)
parser.add_argument("--repeats", type=int, default=96)
parser.add_argument("--public-users", type=int, default=50_000)
parser.add_argument("--seed", type=int, default=20260825)
parser.add_argument("--max-models", type=int, default=10)
parser.add_argument("--output", default="work/compact_stack_risk96.json")
args = parser.parse_args()

pareto = json.loads((WORK / "compact_stack_pareto.json").read_text())
pool = validation_pool()
truth_raw = np.load(WORK / "mat" / "gmv.npy", mmap_mode="r")
truth = np.log1p(truth_raw[:, 379:409].sum(axis=1, dtype=np.float64))
all_index = np.arange(len(truth))
rng = np.random.default_rng(args.seed)
splits = []
for _ in range(args.repeats):
    public = rng.choice(len(truth), args.public_users, replace=False)
    private_mask = np.ones(len(truth), dtype=bool)
    private_mask[public] = False
    splits.append((public, all_index[private_mask]))

reports = []
for compact in pareto["rows"]:
    count = compact["model_count"]
    if count > args.max_models:
        continue
    keys = compact["models"]
    missing = [key for key in keys if key not in pool]
    if missing:
        reports.append({
            "model_count": count,
            "models": keys,
            "status": "missing_validation_vectors",
            "missing": missing,
        })
        continue
    design = np.column_stack([*[pool[key] for key in keys], np.ones(len(truth))])
    oracle, oracle_df, oracle_weights = fit(
        design, truth, all_index, all_index, args.lam
    )
    rows = []
    weights = []
    for public, private in splits:
        public_score, degrees, fitted_weights = fit(
            design, truth, public, public, args.lam
        )
        private_score, _, _ = fit(
            design, truth, public, private, args.lam
        )
        oracle_private_residual = truth[private] - design[private] @ oracle_weights
        oracle_private_score = float(np.sqrt(np.mean(
            oracle_private_residual * oracle_private_residual
        )))
        rows.append([
            public_score,
            private_score,
            degrees,
            oracle_private_score,
            private_score - oracle_private_score,
        ])
        weights.append(fitted_weights[:-1])
    rows = np.asarray(rows)
    weights = np.asarray(weights)
    reports.append({
        "model_count": count,
        "models": keys,
        "status": "ok",
        "lambda": args.lam,
        "repeats": args.repeats,
        "oracle_score_250k": oracle,
        "oracle_degrees_250k": oracle_df,
        "oracle_weights": dict(zip(keys, oracle_weights[:-1].tolist())),
        "mean_fitted_public": float(rows[:, 0].mean()),
        "mean_independent_private": float(rows[:, 1].mean()),
        "private_score_se": float(rows[:, 1].std(ddof=1) / np.sqrt(args.repeats)),
        "mean_public_private_gap": float((rows[:, 1] - rows[:, 0]).mean()),
        "mean_private_excess_vs_population_oracle": float(rows[:, 4].mean()),
        "private_excess_se": float(
            rows[:, 4].std(ddof=1) / np.sqrt(args.repeats)
        ),
        "mean_degrees": float(rows[:, 2].mean()),
        "weight_negative_fraction": dict(zip(keys, np.mean(weights < 0, axis=0).tolist())),
        "mean_weights": dict(zip(keys, weights.mean(axis=0).tolist())),
    })

result = {
    "purpose": "fixed-prefix user-sampling risk; not absolute competition-private forecast",
    "selection_source": "compact_stack_pareto frozen teacher ranking",
    "validation_proxies": {
        "GBD": "v4_262_valpred",
        "gb": "v4_262_valpred because cfg3 validation was not persisted",
    },
    "reports": reports,
}
output = Path(args.output)
if not output.is_absolute():
    output = ROOT / output
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
