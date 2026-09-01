#!/usr/bin/env python
"""Predict each 153 probe's public score before it is sent.

The prediction is deliberately rough: it rescales the fold-378 inner product
between the target and the same transformed column onto the anchor-408 level
using the ratio of the two stacks' own inner products.  It is a transcription
guard and a falsifiable record, not a gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "nl"
M1, M2 = 2.3232887, 10.7633307

truth = np.load(OUT / "truth378.npy").astype(np.float64)
stack = np.load(OUT / "oof378_ridge.npy").astype(np.float64)
hist378 = np.load(OUT / "hist378.npy").astype(np.float64)
names = json.loads((OUT / "hist_keys.json").read_text())
manifest = json.loads((ROOT / "work" / "153_probe_manifest.json").read_text())
ez_base = manifest["ez_base"]
scale = ez_base / float(np.mean(truth * stack))
target_mean, target_sd = stack.mean(), stack.std()


def transform(column):
    scaled = (column - column.mean()) / (column.std() + 1e-12) * target_sd + target_mean
    shift = brentq(
        lambda v: np.clip(scaled + v, 0, None).mean() - target_mean, -50, 50
    )
    return np.clip(scaled + shift, 0, None)


rows = []
for row in manifest["probes"]:
    meta = json.loads((ROOT / "work" / f"{row['tag']}_meta.json").read_text())
    column = transform(hist378[:, names.index(row["column"])])
    ez_candidate = float(np.mean(truth * column)) * scale
    ez_probe = 0.70 * ez_base + 0.30 * ez_candidate
    predicted = float(np.sqrt(max(M2 + meta["probe_second_moment"] - 2 * ez_probe, 0)))
    rows.append({
        "rank": row["rank"], "tag": row["tag"], "column": row["column"],
        "estimated_ez_candidate": ez_candidate,
        "predicted_public": predicted,
    })
    print(f"{row['rank']:2d} {row['column']:16s} predicted public ~ {predicted:.4f}")

(ROOT / "work" / "153_probe_score_predictions.json").write_text(
    json.dumps({"method": "fold-378 inner product rescaled to the anchor-408 level",
                "accuracy": "order-of-magnitude; a returned score more than 0.02 away "
                            "from the prediction should be re-checked for transcription",
                "rows": rows}, indent=2) + "\n"
)
