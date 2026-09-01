#!/usr/bin/env python
"""164 — repackage a candidate as stack + delta so the moment recovery is clean.

The stack at fold 378 is the pool's own out-of-fold ridge prediction; at anchor
408 it is the log-prediction of the current best submission.  delta is the
candidate expressed as a deviation from the stack, rescaled to a fixed fraction
of the stack's spread.  Frozen key: work/164_stack_anchored_construction_preregister.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "nl"
CAND = ROOT / "work" / "cand"
LAM = 0.003
DELTA_SHARE = 0.30      # frozen: delta carries 30 percent of the stack's spread

parser = argparse.ArgumentParser()
parser.add_argument("tags", nargs="+", help="candidate stems without _val/_final")
args = parser.parse_args()

pool = np.load(OUT / "pool27_378.npy").astype(np.float64)
truth = np.load(OUT / "truth378.npy").astype(np.float64)
n = len(truth)
design = np.hstack([pool, np.ones((n, 1))])
penalty = np.eye(design.shape[1]) * LAM
penalty[-1, -1] = 0.0
rng = np.random.default_rng(20260828)
folds = rng.permutation(n) % 5
stack_val = np.zeros(n)
for k in range(5):
    score_index = np.flatnonzero(folds == k)
    fit_index = np.flatnonzero(folds != k)
    x = design[fit_index]
    stack_val[score_index] = design[score_index] @ np.linalg.solve(
        x.T @ x / len(fit_index) + penalty,
        x.T @ truth[fit_index] / len(fit_index),
    )
np.save(OUT / "stack378.npy", stack_val.astype(np.float32))

table = pl.read_csv(ROOT / "submissions" / "130_private_safe_exact_decay_l003.csv")
stack_final = np.log1p(np.clip(table["predict"].to_numpy(), 0, None)).astype(np.float64)

report = {}
for tag in args.tags:
    for suffix, stack in (("val", stack_val), ("final", stack_final)):
        path = CAND / f"{tag}_{suffix}.npy"
        if not path.exists():
            report.setdefault(tag, {})[suffix] = "missing"
            continue
        candidate = np.load(path).astype(np.float64)
        delta = candidate - candidate.mean()
        residual = delta - stack * (delta @ (stack - stack.mean()) /
                                    ((stack - stack.mean()) ** 2).sum())
        scale = DELTA_SHARE * stack.std() / (residual.std() + 1e-12)
        column = stack + scale * residual
        np.save(CAND / f"{tag}_anchored_{suffix}.npy", column.astype(np.float32))
        report.setdefault(tag, {})[suffix] = {
            "correlation_with_stack": float(np.corrcoef(column, stack)[0, 1]),
            "delta_sd": float((scale * residual).std()),
            "stack_sd": float(stack.std()),
        }
        print(f"{tag:20s} {suffix:5s} corr={report[tag][suffix]['correlation_with_stack']:.4f}",
              flush=True)

(CAND / "164_anchored_report.json").write_text(json.dumps(report, indent=2) + "\n")
