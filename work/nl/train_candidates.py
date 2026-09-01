#!/usr/bin/env python
"""161 — train the cheap candidate bases and emit their two prediction vectors.

For every variant this writes <tag>_val.npy, the fold-378 prediction from a
model that never saw a label past day 372, and <tag>_final.npy, the anchor-408
prediction from a model trained the way the competition bases are.
Frozen register: work/161_aggressive_round_register.json.
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
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "work"))
import feats4  # noqa: E402

MAT = Path(os.environ.get("ECUP_MAT", ROOT / "work" / "mat"))
GMV = np.load(MAT / "gmv.npy", mmap_mode="r")
ORD = np.load(MAT / "to_ord.npy", mmap_mode="r")
HORIZON = 30
BASE = dict(
    learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
    feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=10.0, num_threads=10, verbose=-1, max_bin=255, seed=161,
    bagging_seed=161, feature_fraction_seed=161,
)
ROUNDS = 300


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
            days = (np.asarray(
                ORD[:, anchor + 1:anchor + 1 + HORIZON], dtype=np.float32) > 0
            ).sum(axis=1, dtype=np.float64)
            blocks.append(days)
        elif kind == "rank":
            z = np.log1p(window_sum(GMV, anchor, 1, 1 + HORIZON))
            order = np.argsort(np.argsort(z))
            blocks.append(order / (len(z) - 1.0))
        else:
            raise ValueError(kind)
    return np.concatenate(blocks)


VARIANTS = [
    ("h1_head7", dict(objective="regression"), "head7"),
    ("h1_tail7", dict(objective="regression"), "tail7"),
    ("h2_expectile60", dict(objective="quantile", alpha=0.60), "full"),
    ("h2_expectile80", dict(objective="quantile", alpha=0.80), "full"),
    ("h3_orderdays", dict(objective="poisson"), "orderdays"),
    ("h6_rank", dict(objective="regression"), "rank"),
]


def anchors_for(last):
    return [t for t in range(186, last + 1, 12)]


start = time.time()
report = {}
for label, last_anchor, predict_at, suffix in (
    ("val", 342, 378, "val"), ("final", 378, 408, "final"),
):
    train_anchors = anchors_for(last_anchor)
    matrix, names = feats4.build(train_anchors)
    predict_matrix, _ = feats4.build([predict_at])
    print(f"{label}: anchors={len(train_anchors)} X={matrix.shape} "
          f"[{time.time() - start:.0f}s]", flush=True)
    for tag, params, kind in VARIANTS:
        y = targets(train_anchors, kind)
        model = lgb.train(
            {**BASE, **params},
            lgb.Dataset(matrix, y, feature_name=names),
            num_boost_round=ROUNDS,
        )
        prediction = model.predict(predict_matrix).astype(np.float32)
        np.save(OUT / f"{tag}_{suffix}.npy", prediction)
        report.setdefault(tag, {})[suffix] = {
            "mean": float(prediction.mean()), "sd": float(prediction.std()),
        }
        print(f"  {tag:18s} {suffix} mean={prediction.mean():.4f} "
              f"sd={prediction.std():.4f} [{time.time() - start:.0f}s]", flush=True)
        del y, model
        gc.collect()

    # H7 needs the network residual, which only exists at the validation fold
    if label == "val":
        network = ROOT / "work" / "w409c_val.npy"
        if network.exists():
            base_target = targets(train_anchors, "full")
            # the network prediction exists only at 378, so H7 is trained to
            # predict the FULL target and its residual role is realised by the
            # ridge; a true residual target would need per-anchor network output
            report["h7_note"] = "skipped: w409c has no per-anchor predictions"
    del matrix, predict_matrix
    gc.collect()

(OUT / "161_training_report.json").write_text(json.dumps(report, indent=2) + "\n")
print("DONE", flush=True)
