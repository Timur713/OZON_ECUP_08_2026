#!/usr/bin/env python
"""176b - a properly long boosting run, on CPU so it does not contend with the
network queue on the GPU.

Every boosting base in the pool was trained for 250 to 300 rounds at learning
rate 0.05. That is a short run by any standard. FINDINGS records that
hyperparameter changes produced clones worth nothing in the stack, but a
ten-times longer run at a fifth of the learning rate is a different object from
a hyperparameter tweak.
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
OUT = Path(os.environ.get("ECUP_OUT", ROOT / "work" / "div"))
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "work"))
import feats4  # noqa: E402

MAT = Path(os.environ.get("ECUP_MAT", ROOT / "work" / "mat"))
GMV = np.load(MAT / "gmv.npy", mmap_mode="r")
HORIZON = 30
ROUNDS = int(os.environ.get("GBDT_ROUNDS", "2500"))
# Only ONE thing changes from the pool's boosting recipe: the schedule. The
# earlier attempt changed leaves, minimum leaf size, regularisation and the
# learning rate all at once and scored 1.73197, which measured overfitting from
# bigger trees rather than anything about longer training.
PARAMS = dict(
    objective="regression", metric="l2",
    learning_rate=float(os.environ.get("GBDT_LR", "0.02")),
    num_leaves=127, min_data_in_leaf=200, feature_fraction=0.6,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
    num_threads=8, verbose=-1, max_bin=255,
    seed=int(os.environ.get("GBDT_SEED", "176")),
    bagging_seed=int(os.environ.get("GBDT_SEED", "176")),
    feature_fraction_seed=int(os.environ.get("GBDT_SEED", "176")),
)


def target(anchors):
    return np.concatenate([
        np.log1p(GMV[:, a + 1:a + 1 + HORIZON].sum(axis=1, dtype=np.float64))
        for a in anchors
    ])


start = time.time()
for last, predict_anchor, suffix in ((342, 378, "val"), (378, 408, "final")):
    anchors = [t for t in range(186, last + 1, 12)]
    matrix, names = feats4.build(anchors)
    predict_matrix, _ = feats4.build([predict_anchor])
    print(f"{suffix}: anchors={len(anchors)} X={matrix.shape} "
          f"[{time.time() - start:.0f}s]", flush=True)
    model = lgb.train(
        PARAMS, lgb.Dataset(matrix, target(anchors), feature_name=names),
        num_boost_round=ROUNDS,
    )
    prediction = model.predict(predict_matrix).astype(np.float32)
    np.save(OUT / f"{os.environ.get('GBDT_TAG', 'gbdtfair')}_{suffix}.npy", prediction)
    print(f"  {os.environ.get('GBDT_TAG', 'gbdtfair')} {suffix} mean={prediction.mean():.4f} "
          f"sd={prediction.std():.4f} [{time.time() - start:.0f}s]", flush=True)
    del matrix, predict_matrix, model
    gc.collect()
print("DONE", flush=True)
