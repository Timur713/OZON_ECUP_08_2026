#!/usr/bin/env python
"""152, third check — does the gain survive the transform the probe builder
actually applies?

work/build_gpu_probe.py rescales a candidate to the base vector's mean and
standard deviation and then shifts and clips it at zero.  The clip is a real
nonlinearity, so the audit has to be run on the clipped columns rather than on
the clean standardised ones.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

OUT = Path(__file__).resolve().parent
LAM = 0.003
REPEATS = 48
PUBLIC_USERS = 50_000
SEED = 20260828
K_GRID = [6, 10, 12, 16, 20, 24, 30, 40, 60]
RANK_FOLDS = [288, 318, 348]

base = np.load(OUT / "base378.npy").astype(np.float64)
truth = np.load(OUT / "truth378.npy").astype(np.float64)
hist = np.load(OUT / "hist378.npy").astype(np.float64)
names = json.loads((OUT / "hist_keys.json").read_text())
stack = np.load(OUT / "oof378_ridge.npy").astype(np.float64)
unique = json.loads((OUT / "hist_unique.json").read_text())["kept_indices"]
ranking = np.mean(
    [np.abs(np.load(OUT / f"state_{f}_beta.npy")) for f in RANK_FOLDS], axis=0
)
masked = np.full(len(ranking), -np.inf)
masked[unique] = ranking[unique]
order = np.argsort(-masked)[:len(unique)]

target_mean, target_sd = stack.mean(), stack.std()


def probe_transform(column):
    scaled = (column - column.mean()) / (column.std() + 1e-12) * target_sd + target_mean
    shift = brentq(
        lambda value: np.clip(scaled + value, 0, None).mean() - target_mean, -50, 50
    )
    return np.clip(scaled + shift, 0, None)


clipped = np.column_stack([probe_transform(hist[:, c]) for c in order])
clip_fraction = [float((clipped[:, i] == 0).mean()) for i in range(clipped.shape[1])]

n = len(truth)
all_index = np.arange(n)
rng = np.random.default_rng(SEED)
splits = []
for _ in range(REPEATS):
    public = rng.choice(n, PUBLIC_USERS, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[public] = False
    splits.append((public, all_index[mask]))


def fit(design, fit_index, score_index):
    x = design[fit_index]
    gram = x.T @ x / len(fit_index)
    rhs = x.T @ truth[fit_index] / len(fit_index)
    penalty = np.eye(design.shape[1]) * LAM
    penalty[-1, -1] = 0
    weights = np.linalg.solve(gram + penalty, rhs)
    residual = truth[score_index] - design[score_index] @ weights
    degrees = float(np.trace(gram @ np.linalg.inv(gram + penalty)))
    return float(np.sqrt(np.mean(residual * residual))), degrees


ones = np.ones((n, 1))
base_design = np.hstack([base, ones])
base_public = np.empty(REPEATS)
base_private = np.empty(REPEATS)
base_df = np.empty(REPEATS)
for i, (public, private) in enumerate(splits):
    base_public[i], base_df[i] = fit(base_design, public, public)
    base_private[i], _ = fit(base_design, public, private)

report = {
    "tag": "152_probe_faithful_unique",
    "note": "columns passed through the exact build_gpu_probe transform",
    "base_mean_public": float(base_public.mean()),
    "base_mean_private": float(base_private.mean()),
    "column_order": [names[c] for c in order[:max(K_GRID)]],
    "clipped_fraction": dict(
        zip([names[c] for c in order[:max(K_GRID)]], clip_fraction[:max(K_GRID)])
    ),
    "by_k": {},
}
for k in K_GRID:
    design = np.hstack([base, clipped[:, :k], ones])
    public_scores = np.empty(REPEATS)
    private_scores = np.empty(REPEATS)
    degrees = np.empty(REPEATS)
    for i, (public, private) in enumerate(splits):
        public_scores[i], degrees[i] = fit(design, public, public)
        private_scores[i], _ = fit(design, public, private)
    gain = base_private - private_scores
    report["by_k"][str(k)] = {
        "mean_private_gain": float(gain.mean()),
        "standard_error": float(gain.std(ddof=1) / np.sqrt(REPEATS)),
        "positive_fraction": float((gain > 0).mean()),
        "mean_public_gain": float((base_public - public_scores).mean()),
        "added_df": float((degrees - base_df).mean()),
    }
    row = report["by_k"][str(k)]
    print(f"K={k:3d} private={row['mean_private_gain']:+.7f} "
          f"se={row['standard_error']:.7f} pos={row['positive_fraction']:.2f} "
          f"public={row['mean_public_gain']:+.7f} df+={row['added_df']:.2f}",
          flush=True)

(OUT / "152_probe_faithful_unique_fold378.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print("\ntop columns:", report["column_order"][:12])
