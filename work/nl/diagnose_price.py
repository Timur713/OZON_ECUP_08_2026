#!/usr/bin/env python
"""Diagnostic for 158: print the raw public and private scores so the measured
price per degree of freedom can be checked against the textbook 2*df*sigma^2/n.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
REPEATS = 48
PUBLIC_USERS = 50_000
SEED = 20260828

base = np.load(OUT / "base378.npy").astype(np.float64)
truth = np.load(OUT / "truth378.npy").astype(np.float64)
n = len(truth)
design = np.hstack([base, np.ones((n, 1))])
all_index = np.arange(n)
rng = np.random.default_rng(SEED)
splits = []
for _ in range(REPEATS):
    public = rng.choice(n, PUBLIC_USERS, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[public] = False
    splits.append((public, all_index[mask]))

for lam in (0.003, 0.0):
    penalty = np.eye(design.shape[1]) * lam
    penalty[-1, -1] = 0
    pub = np.empty(REPEATS)
    pri = np.empty(REPEATS)
    dfs = np.empty(REPEATS)
    for i, (public, private) in enumerate(splits):
        x = design[public]
        gram = x.T @ x / len(public)
        weights = np.linalg.solve(
            gram + penalty, x.T @ truth[public] / len(public)
        )
        r = truth[public] - design[public] @ weights
        pub[i] = np.sqrt(np.mean(r * r))
        r = truth[private] - design[private] @ weights
        pri[i] = np.sqrt(np.mean(r * r))
        dfs[i] = float(np.trace(gram @ np.linalg.inv(gram + penalty)))
    sigma2 = float(pub.mean() ** 2)
    predicted_gap = 2 * dfs.mean() * sigma2 / PUBLIC_USERS / (2 * pub.mean())
    print(f"lam={lam:<6g} df={dfs.mean():6.2f} public={pub.mean():.7f} "
          f"private={pri.mean():.7f} gap={pri.mean() - pub.mean():+.7f} "
          f"textbook_gap={predicted_gap:.7f} "
          f"per_df={(pri.mean() - pub.mean()) / dfs.mean():.3e}")
