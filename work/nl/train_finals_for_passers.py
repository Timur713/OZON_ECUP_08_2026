#!/usr/bin/env python
"""Train ONLY the candidates that passed screening at the competition anchor.

The first pass trains every variant at both anchors, which wastes roughly an
hour on variants that will be rejected.  This trains the anchor-408 vector for a
named subset only.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[2]))
OUT = Path(os.environ.get("ECUP_OUT", ROOT / "work" / "cand"))
sys.path.insert(0, str(ROOT / "work"))
import feats4  # noqa: E402

MAT = Path(os.environ.get("ECUP_MAT", ROOT / "work" / "mat"))
GMV = np.load(MAT / "gmv.npy", mmap_mode="r")
ORD = np.load(MAT / "to_ord.npy", mmap_mode="r")
HORIZON = 30
BASE = dict(
    learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
    feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=10.0, num_threads=16, verbose=-1, max_bin=255, seed=161,
    bagging_seed=161, feature_fraction_seed=161,
)
ROUNDS = 300
SPEC = {
    "h1_head7": (dict(objective="regression"), "head7"),
    "h1_tail7": (dict(objective="regression"), "tail7"),
    "h2_expectile60": (dict(objective="quantile", alpha=0.60), "full"),
    "h2_expectile80": (dict(objective="quantile", alpha=0.80), "full"),
    "h3_orderdays": (dict(objective="poisson"), "orderdays"),
    "h6_rank": (dict(objective="regression"), "rank"),
}


def window_sum(matrix, anchor, start, stop):
    return matrix[:, anchor + start:anchor + stop].sum(axis=1, dtype=np.float64)


def targets(anchors, kind):
    blocks = []
    for anchor in anchors:
        if kind == "full":
            blocks.append(np.log1p(window_sum(GMV, anchor, 1, 1 + HORIZON)))
        elif kind == "head7":
            blocks.append(np.log1p(window_sum(GMV, anchor, 1, 8)))
        elif kind == "tail7":
            blocks.append(np.log1p(window_sum(GMV, anchor, 24, 31)))
        elif kind == "orderdays":
            blocks.append((np.asarray(
                ORD[:, anchor + 1:anchor + 1 + HORIZON], dtype=np.float32) > 0
            ).sum(axis=1, dtype=np.float64))
        elif kind == "rank":
            z = np.log1p(window_sum(GMV, anchor, 1, 1 + HORIZON))
            blocks.append(np.argsort(np.argsort(z)) / (len(z) - 1.0))
        else:
            raise ValueError(kind)
    return np.concatenate(blocks)


wanted = sys.argv[1:]
if not wanted:
    raise SystemExit("name at least one variant")
anchors = [t for t in range(186, 379, 12)]
start = time.time()
matrix, names = feats4.build(anchors)
predict_matrix, _ = feats4.build([408])
print(f"final: anchors={len(anchors)} X={matrix.shape} [{time.time() - start:.0f}s]",
      flush=True)
report = {}
for tag in wanted:
    params, kind = SPEC[tag]
    model = lgb.train(
        {**BASE, **params},
        lgb.Dataset(matrix, targets(anchors, kind), feature_name=names),
        num_boost_round=ROUNDS,
    )
    prediction = model.predict(predict_matrix).astype(np.float32)
    np.save(OUT / f"{tag}_final.npy", prediction)
    report[tag] = {"mean": float(prediction.mean()), "sd": float(prediction.std())}
    print(f"  {tag} mean={prediction.mean():.4f} sd={prediction.std():.4f} "
          f"[{time.time() - start:.0f}s]", flush=True)
    del model
    gc.collect()
(OUT / "161_final_report.json").write_text(json.dumps(report, indent=2) + "\n")
print("DONE", flush=True)
