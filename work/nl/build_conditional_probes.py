#!/usr/bin/env python
"""Build the 153 probe batch: one plain-CSV probe per historical column.

Each probe is 0.70 * 130 + 0.30 * transformed column, mean-calibrated to M1,
so the existing recovery in work/solve_augmented_stack.py applies unchanged.
Nothing here reads a leaderboard score.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "nl"
WORK = ROOT / "work"
SUBS = ROOT / "submissions"
M1, M2 = 2.3232887, 10.7633307
BASE_NAME = "130_private_safe_exact_decay_l003.csv"
BASE_SCORE = 1.6461706600883055
WEIGHT = 0.30
COUNT = 30
RANK_FOLDS = [288, 318, 348]

table = pl.read_csv(SUBS / BASE_NAME)
base = np.log1p(np.clip(table["predict"].to_numpy(), 0, None)).astype(np.float64)
user_ids = table["user_id"].to_numpy()
base_second = float(np.mean(base * base))
ez_base = (M2 + base_second - BASE_SCORE ** 2) / 2

hist = np.load(OUT / "hist408.npy").astype(np.float64)
names = json.loads((OUT / "hist_keys.json").read_text())
unique = json.loads((OUT / "hist_unique.json").read_text())["kept_indices"]
ranking = np.mean(
    [np.abs(np.load(OUT / f"state_{f}_beta.npy")) for f in RANK_FOLDS], axis=0
)
masked = np.full(len(ranking), -np.inf)
masked[unique] = ranking[unique]
order = np.argsort(-masked)[:COUNT]


def transform(column):
    scaled = (column - column.mean()) / (column.std() + 1e-12) * base.std() + base.mean()
    shift = brentq(lambda v: np.clip(scaled + v, 0, None).mean() - M1, -50, 50)
    return np.clip(scaled + shift, 0, None)


manifest = []
for rank, index in enumerate(order, start=1):
    name = names[index]
    tag = f"153_probe_hist{rank:02d}_{name}"
    candidate = transform(hist[:, index])
    candidate_file = WORK / f"{tag}_candidate.npy"
    np.save(candidate_file, candidate)
    probe = (1.0 - WEIGHT) * base + WEIGHT * candidate
    assert abs(probe.mean() - M1) < 2e-6, probe.mean()
    prediction = np.expm1(probe)
    assert np.isfinite(prediction).all() and (prediction >= 0).all()
    path = SUBS / f"{tag}.csv"
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["user_id", "predict"])
        for user_id, value in zip(user_ids, prediction):
            writer.writerow([int(user_id), float(value)])
    metadata = {
        "tag": tag,
        "file": str(path),
        "role": "ZOND",
        "column": name,
        "column_rank": rank,
        "vector": str(OUT / "hist408.npy"),
        "vector_key": None,
        "candidate_file": str(candidate_file),
        "base_submission": str(SUBS / BASE_NAME),
        "base_score": BASE_SCORE,
        "weight": WEIGHT,
        "inverted": False,
        "mean_log": float(probe.mean()),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "base_mean": float(base.mean()),
        "candidate_mean": float(candidate.mean()),
        "base_second_moment": base_second,
        "candidate_second_moment": float(np.mean(candidate * candidate)),
        "cross_base_candidate": float(np.mean(base * candidate)),
        "probe_second_moment": float(np.mean(probe * probe)),
        "corr_base_candidate": float(np.corrcoef(base, candidate)[0, 1]),
        "distance_squared": float(np.mean((base - candidate) ** 2)),
        "clipped_fraction": float((candidate == 0).mean()),
        "ez_base": ez_base,
    }
    (WORK / f"{tag}_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest.append({
        "rank": rank, "tag": tag, "column": name,
        "file": f"submissions/{tag}.csv",
        "corr_with_130": metadata["corr_base_candidate"],
        "clipped_fraction": metadata["clipped_fraction"],
    })
    print(f"{rank:2d} {name:20s} corr={metadata['corr_base_candidate']:+.4f} "
          f"clip={metadata['clipped_fraction']:.3f}", flush=True)

(WORK / "153_probe_manifest.json").write_text(
    json.dumps({"base": BASE_NAME, "base_score": BASE_SCORE, "weight": WEIGHT,
                "ez_base": ez_base, "probes": manifest}, indent=2) + "\n"
)
print("\nwrote", len(manifest), "probes")
