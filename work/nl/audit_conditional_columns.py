#!/usr/bin/env python
"""152 — do a few historical columns pay for their degrees of freedom?

Runs the project's own fit-50k / score-200k protocol at fold 378 with the 25
admitted validation columns as the base design, so the public-to-private
optimism of the extra parameters is paid inside the measurement.

Frozen reading key: work/152_public_conditional_calibration_preregister.json.
The ranking of the columns comes only from folds 288, 318 and 348.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
LAM = 0.003
REPEATS = 48
PUBLIC_USERS = 50_000
SEED = 20260828
K_GRID = [2, 4, 6, 8, 12, 16, 24, 76]
RANK_FOLDS = [288, 318, 348]

base = np.load(OUT / "base378.npy").astype(np.float64)
truth = np.load(OUT / "truth378.npy").astype(np.float64)
hist = np.load(OUT / "hist378.npy").astype(np.float64)
names = json.loads((OUT / "hist_keys.json").read_text())
hist = (hist - hist.mean(0)) / (hist.std(0) + 1e-9)

ranking_score = np.mean(
    [np.abs(np.load(OUT / f"state_{f}_beta.npy")) for f in RANK_FOLDS], axis=0
)
order = np.argsort(-ranking_score)
n = len(truth)
all_index = np.arange(n)
rng = np.random.default_rng(SEED)
splits = []
for _ in range(REPEATS):
    public = rng.choice(n, PUBLIC_USERS, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[public] = False
    splits.append((public, all_index[mask]))


def fit_model(design, fit_index, score_index):
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
    base_public[i], base_df[i] = fit_model(base_design, public, public)
    base_private[i], _ = fit_model(base_design, public, private)

report = {
    "tag": "152_public_conditional_calibration",
    "fold": 378,
    "repeats": REPEATS,
    "lambda": LAM,
    "base_mean_public": float(base_public.mean()),
    "base_mean_private": float(base_private.mean()),
    "base_mean_df": float(base_df.mean()),
    "ranking_from_folds": RANK_FOLDS,
    "by_k": {},
}
for k in K_GRID:
    columns = order[:k]
    design = np.hstack([base, hist[:, columns], ones])
    public_scores = np.empty(REPEATS)
    private_scores = np.empty(REPEATS)
    degrees = np.empty(REPEATS)
    for i, (public, private) in enumerate(splits):
        public_scores[i], degrees[i] = fit_model(design, public, public)
        private_scores[i], _ = fit_model(design, public, private)
    private_gain = base_private - private_scores
    report["by_k"][str(k)] = {
        "columns": [names[c] for c in columns],
        "mean_public_gain": float((base_public - public_scores).mean()),
        "mean_private_gain": float(private_gain.mean()),
        "private_gain_standard_error": float(
            private_gain.std(ddof=1) / np.sqrt(REPEATS)
        ),
        "positive_fraction": float((private_gain > 0).mean()),
        "added_df": float((degrees - base_df).mean()),
    }
    row = report["by_k"][str(k)]
    print(f"K={k:3d} private_gain={row['mean_private_gain']:+.7f} "
          f"se={row['private_gain_standard_error']:.7f} "
          f"pos={row['positive_fraction']:.2f} df+={row['added_df']:.2f} "
          f"public_gain={row['mean_public_gain']:+.7f}", flush=True)

best = max(report["by_k"], key=lambda k: report["by_k"][k]["mean_private_gain"])
best_row = report["by_k"][best]
report["best_k"] = int(best)
report["verdict"] = (
    "promote_candidate" if best_row["mean_private_gain"] >= 0.00040
    and best_row["positive_fraction"] >= 0.90
    else "diagnostic" if best_row["mean_private_gain"] >= 0.00015
    else "closed"
)
(OUT / "152_conditional_columns_fold378.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print("\nbest K", best, "verdict", report["verdict"])
