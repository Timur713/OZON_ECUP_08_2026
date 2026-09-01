#!/usr/bin/env python
"""Assemble the fold-378 audit frame: 25 admitted base columns, cheap
historical user features and the true 30-day log-GMV target.

Nothing here touches the leaderboard.  Every column is either an existing
validation prediction for anchor 378 or an aggregate over days < 379.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
OUT = WORK / "nl"
ANCHOR = 378
TARGET = slice(379, 409)


def load(name):
    return np.load(WORK / name).astype(np.float32)


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
    "W45": np.mean([load(f"w45{s}_val.npy") for s in "abcd"], axis=0),
    "W60": np.mean([load(f"w60{s}_val.npy") for s in "abc"], axis=0),
    "W90": np.mean([load(f"w90{s}_val.npy") for s in "abc"], axis=0),
    "W120": np.mean([load(f"w120{s}_val.npy") for s in "abc"], axis=0),
    "W150": load("w150a_val.npy"),
    "W180": np.mean([load(f"w180{s}_val.npy") for s in "ab"], axis=0),
    "W210": load("w210a_val.npy"),
    "W270": load("w270a_val.npy"),
    "W300": load("w300a_val.npy"),
    "W365": np.mean([load(f"w365{s}_val.npy") for s in "ab"], axis=0),
    "W409": load("w409a_val.npy"),
    "cls300": load("cls300_val_server_val.npy"),
    "cls409": load("cls409_val_server_val.npy"),
}
base = np.column_stack(list(columns.values()))
np.save(OUT / "base378.npy", base)
(OUT / "base378_keys.json").write_text(
    "[\n" + ",\n".join(f'  "{k}"' for k in columns) + "\n]\n"
)

gmv = np.load(WORK / "mat" / "gmv.npy", mmap_mode="r")
truth = np.log1p(gmv[:, TARGET].sum(axis=1, dtype=np.float64)).astype(np.float32)
np.save(OUT / "truth378.npy", truth)

WINDOWS = (7, 14, 30, 60, 90, 180, 365)
feature_names = []
feature_blocks = []


def window_block(matrix, tag, log_scale):
    hist = matrix[:, :ANCHOR + 1]
    for w in WINDOWS:
        value = hist[:, -w:].sum(axis=1, dtype=np.float64)
        feature_blocks.append(np.log1p(value) if log_scale else value)
        feature_names.append(f"{tag}_d{w}")


def recency_and_span(matrix, tag):
    hist = np.asarray(matrix[:, :ANCHOR + 1], dtype=np.float32) > 0
    idx = np.arange(hist.shape[1], dtype=np.float32)
    last = np.where(hist.any(axis=1), (hist * idx).max(axis=1), -1.0)
    first = np.where(hist.any(axis=1), np.where(hist, idx, np.inf).min(axis=1), -1.0)
    feature_blocks.append(np.where(last < 0, 999.0, ANCHOR - last))
    feature_names.append(f"{tag}_recency")
    feature_blocks.append(np.where(first < 0, 999.0, ANCHOR - first))
    feature_names.append(f"{tag}_age")
    feature_blocks.append(hist.sum(axis=1, dtype=np.float64))
    feature_names.append(f"{tag}_active_days")


for name, tag, log_scale in (
    ("gmv", "gmv", True),
    ("to_ord", "ord", False),
    ("searches", "srch", False),
    ("to_cart", "cart", False),
    ("active", "act", False),
):
    matrix = np.load(WORK / "mat" / f"{name}.npy", mmap_mode="r")
    window_block(matrix, tag, log_scale)
    if name in ("gmv", "to_ord", "active"):
        recency_and_span(matrix, tag)
    del matrix

features = np.column_stack(feature_blocks).astype(np.float32)
np.save(OUT / "feat378.npy", features)
(OUT / "feat378_keys.json").write_text(
    "[\n" + ",\n".join(f'  "{k}"' for k in feature_names) + "\n]\n"
)
print("base", base.shape, "features", features.shape, "truth", truth.shape)
print("truth mean", float(truth.mean()), "sd", float(truth.std()))
