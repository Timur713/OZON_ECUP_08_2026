#!/usr/bin/env python
"""Measure a new validation vector's marginal ridge value on held-out users."""
import argparse
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"


def load(name):
    return np.load(WORK / name).astype(np.float64)


def load_spec(spec):
    path, key = spec.rsplit(":", 1) if ":" in spec else (spec, None)
    value = np.load(path)
    if isinstance(value, np.lib.npyio.NpzFile):
        if key is None:
            raise ValueError(f"NPZ key required: {spec}")
        result = value[key].astype(np.float64)
        value.close()
        return result
    if key is not None:
        raise ValueError(f"unexpected key for NPY: {spec}")
    return value.astype(np.float64)


parser = argparse.ArgumentParser()
parser.add_argument("candidates", nargs="+")
parser.add_argument("--lam", type=float, default=0.003)
parser.add_argument("--repeats", type=int, default=12)
parser.add_argument("--public-users", type=int, default=50_000)
parser.add_argument("--seed", type=int, default=20260825)
parser.add_argument(
    "--joint", action="store_true",
    help="evaluate all supplied candidates as one augmented ridge block",
)
parser.add_argument(
    "--select-one", action="store_true",
    help=(
        "on each 50k split select one candidate by fitted-public score and "
        "evaluate that fixed choice on the independent 200k"
    ),
)
args = parser.parse_args()
if args.joint and args.select_one:
    raise ValueError("--joint and --select-one are mutually exclusive")

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
    "W45": np.mean([load(f"w45{seed}_val.npy") for seed in "abcd"], axis=0),
    "W60": np.mean([load(f"w60{seed}_val.npy") for seed in "abc"], axis=0),
    "W90": np.mean([load(f"w90{seed}_val.npy") for seed in "abc"], axis=0),
    "W120": np.mean([load(f"w120{seed}_val.npy") for seed in "abc"], axis=0),
    "W150": load("w150a_val.npy"),
    "W180": np.mean([load(f"w180{seed}_val.npy") for seed in "ab"], axis=0),
    "W210": load("w210a_val.npy"),
    "W270": load("w270a_val.npy"),
    "W300": load("w300a_val.npy"),
    "W365": np.mean([load(f"w365{seed}_val.npy") for seed in "ab"], axis=0),
    "W409": load("w409a_val.npy"),
    "cls300": load("cls300_val_server_val.npy"),
    "cls409": load("cls409_val_server_val.npy"),
}
gmv = np.load(WORK / "mat" / "gmv.npy", mmap_mode="r")
truth = np.log1p(gmv[:, 379:409].sum(axis=1, dtype=np.float64))
base = np.column_stack([*columns.values(), np.ones_like(truth)])
all_index = np.arange(len(truth))


def fit_model(design, fit_index, score_index):
    x = design[fit_index]
    y = truth[fit_index]
    gram = x.T @ x / len(fit_index)
    rhs = x.T @ y / len(fit_index)
    penalty = np.eye(design.shape[1]) * args.lam
    penalty[-1, -1] = 0
    weights = np.linalg.solve(gram + penalty, rhs)
    residual = truth[score_index] - design[score_index] @ weights
    score = float(np.sqrt(np.mean(residual * residual)))
    degrees = float(np.trace(gram @ np.linalg.inv(gram + penalty)))
    return score, degrees, weights


def fit_score(design, fit_index, score_index):
    score, degrees, weights = fit_model(design, fit_index, score_index)
    return score, degrees, float(weights[-2])


base_oracle, base_df, _ = fit_score(base, all_index, all_index)
rng = np.random.default_rng(args.seed)
splits = []
for _ in range(args.repeats):
    public = rng.choice(len(truth), args.public_users, replace=False)
    private_mask = np.ones(len(truth), dtype=bool)
    private_mask[public] = False
    splits.append((public, all_index[private_mask]))

if args.select_one:
    candidate_values = [load_spec(spec) for spec in args.candidates]
    augmented_designs = []
    for spec, candidate in zip(args.candidates, candidate_values):
        if candidate.shape != truth.shape or not np.isfinite(candidate).all():
            raise ValueError(f"invalid candidate {spec}: {candidate.shape}")
        augmented_designs.append(np.column_stack([
            base[:, :-1], candidate, base[:, -1],
        ]))
    selected_rows = []
    selected_indices = []
    all_private_gains = []
    for public, private in splits:
        base_public, base_split_df, _ = fit_model(base, public, public)
        base_private, _, _ = fit_model(base, public, private)
        candidate_rows = []
        for augmented in augmented_designs:
            aug_public, aug_split_df, aug_weights = fit_model(
                augmented, public, public
            )
            aug_private, _, _ = fit_model(augmented, public, private)
            candidate_rows.append([
                base_public - aug_public,
                base_private - aug_private,
                aug_split_df - base_split_df,
                aug_weights[-2],
            ])
        candidate_rows = np.asarray(candidate_rows)
        chosen = int(np.argmax(candidate_rows[:, 0]))
        selected_indices.append(chosen)
        selected_rows.append(candidate_rows[chosen])
        all_private_gains.append(candidate_rows[:, 1])
    selected_rows = np.asarray(selected_rows)
    all_private_gains = np.asarray(all_private_gains)
    counts = np.bincount(selected_indices, minlength=len(args.candidates))
    report = {
        "selection_candidates": args.candidates,
        "lambda": args.lam,
        "base_columns": len(columns),
        "selection_rule": "maximum fitted-public gain on each 50k split",
        "repeats": args.repeats,
        "mean_selected_fitted_public_gain": float(selected_rows[:, 0].mean()),
        "mean_selected_independent_private_gain": float(
            selected_rows[:, 1].mean()
        ),
        "selected_private_gain_se": float(
            selected_rows[:, 1].std(ddof=1) / np.sqrt(args.repeats)
        ),
        "selected_private_gain_positive_fraction": float(
            np.mean(selected_rows[:, 1] > 0)
        ),
        "selected_private_gain_min": float(selected_rows[:, 1].min()),
        "selected_private_gain_max": float(selected_rows[:, 1].max()),
        "mean_selected_degrees_added": float(selected_rows[:, 2].mean()),
        "selection_counts": dict(zip(args.candidates, counts.tolist())),
        "mean_private_gain_by_candidate": dict(zip(
            args.candidates, all_private_gains.mean(axis=0).tolist()
        )),
        "oracle_mean_best_private_gain_per_split": float(
            all_private_gains.max(axis=1).mean()
        ),
    }
    print(json.dumps([report], indent=2))
    raise SystemExit(0)

if args.joint:
    candidate_values = [load_spec(spec) for spec in args.candidates]
    for spec, candidate in zip(args.candidates, candidate_values):
        if candidate.shape != truth.shape or not np.isfinite(candidate).all():
            raise ValueError(f"invalid candidate {spec}: {candidate.shape}")
    augmented = np.column_stack([
        base[:, :-1], *candidate_values, base[:, -1],
    ])
    base_oracle_score, base_oracle_df, _ = fit_model(
        base, all_index, all_index
    )
    aug_oracle_score, aug_oracle_df, aug_oracle_weights = fit_model(
        augmented, all_index, all_index
    )
    split_rows = []
    split_weights = []
    candidate_slice = slice(-(len(candidate_values) + 1), -1)
    for public, private in splits:
        base_public, base_split_df, _ = fit_model(base, public, public)
        base_private, _, _ = fit_model(base, public, private)
        aug_public, aug_split_df, aug_weights = fit_model(
            augmented, public, public
        )
        aug_private, _, _ = fit_model(augmented, public, private)
        split_rows.append([
            base_public - aug_public,
            base_private - aug_private,
            aug_split_df - base_split_df,
        ])
        split_weights.append(aug_weights[candidate_slice])
    split_rows = np.asarray(split_rows)
    split_weights = np.asarray(split_weights)
    report = {
        "candidates": args.candidates,
        "lambda": args.lam,
        "base_columns": len(columns),
        "oracle_base_score_250k": base_oracle_score,
        "oracle_augmented_score_250k": aug_oracle_score,
        "oracle_gain_250k": base_oracle_score - aug_oracle_score,
        "oracle_degrees_added": aug_oracle_df - base_oracle_df,
        "oracle_candidate_weights": dict(zip(
            args.candidates,
            aug_oracle_weights[candidate_slice].tolist(),
        )),
        "repeats": args.repeats,
        "mean_fitted_public_gain": float(split_rows[:, 0].mean()),
        "mean_independent_private_gain": float(split_rows[:, 1].mean()),
        "private_gain_se": float(
            split_rows[:, 1].std(ddof=1) / np.sqrt(args.repeats)
        ),
        "private_gain_positive_fraction": float(
            np.mean(split_rows[:, 1] > 0)
        ),
        "private_gain_min": float(split_rows[:, 1].min()),
        "private_gain_max": float(split_rows[:, 1].max()),
        "mean_degrees_added": float(split_rows[:, 2].mean()),
        "mean_candidate_weights": dict(zip(
            args.candidates,
            split_weights.mean(axis=0).tolist(),
        )),
        "candidate_weight_negative_fraction": dict(zip(
            args.candidates,
            np.mean(split_weights < 0, axis=0).tolist(),
        )),
    }
    print(json.dumps([report], indent=2))
    raise SystemExit(0)

reports = []
for spec in args.candidates:
    candidate = load_spec(spec)
    if candidate.shape != truth.shape or not np.isfinite(candidate).all():
        raise ValueError(f"invalid candidate {spec}: {candidate.shape}")
    augmented = np.column_stack([base[:, :-1], candidate, base[:, -1]])
    augmented_oracle, augmented_df, weight = fit_score(
        augmented, all_index, all_index
    )
    split_rows = []
    for public, private in splits:
        base_public, base_split_df, _ = fit_score(base, public, public)
        base_private, _, _ = fit_score(base, public, private)
        aug_public, aug_split_df, aug_weight = fit_score(augmented, public, public)
        aug_private, _, _ = fit_score(augmented, public, private)
        split_rows.append([
            base_public - aug_public,
            base_private - aug_private,
            aug_split_df - base_split_df,
            aug_weight,
        ])
    split_rows = np.asarray(split_rows)
    reports.append({
        "candidate": spec,
        "lambda": args.lam,
        "base_columns": len(columns),
        "oracle_base_score_250k": base_oracle,
        "oracle_augmented_score_250k": augmented_oracle,
        "oracle_gain_250k": base_oracle - augmented_oracle,
        "oracle_degrees_added": augmented_df - base_df,
        "oracle_candidate_weight": weight,
        "repeats": args.repeats,
        "mean_fitted_public_gain": float(split_rows[:, 0].mean()),
        "mean_independent_private_gain": float(split_rows[:, 1].mean()),
        "private_gain_se": float(split_rows[:, 1].std(ddof=1) / np.sqrt(args.repeats)),
        "private_gain_positive_fraction": float(np.mean(split_rows[:, 1] > 0)),
        "private_gain_min": float(split_rows[:, 1].min()),
        "private_gain_max": float(split_rows[:, 1].max()),
        "mean_degrees_added": float(split_rows[:, 2].mean()),
        "mean_candidate_weight": float(split_rows[:, 3].mean()),
        "candidate_weight_negative_fraction": float(np.mean(split_rows[:, 3] < 0)),
    })
print(json.dumps(reports, indent=2))
