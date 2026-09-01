"""One historical-feature definition, usable at any anchor.

Identical column set and order at every anchor, so a ranking learned on one
fold can be applied to another without any re-derivation.  Uses only days
<= anchor.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[2]))
MAT = ROOT / "work" / "mat"
WINDOWS = (7, 14, 30, 60, 90, 180, 365)
CHANNELS = (("gmv", True), ("to_ord", False), ("searches", False),
            ("to_cart", False), ("active", False))
RECENCY_CHANNELS = ("gmv", "to_ord", "active")

NAMES = []
for _name, _log in CHANNELS:
    NAMES += [f"{_name}_s{w}" for w in WINDOWS]
    NAMES += [f"{_name}_d{w}" for w in WINDOWS]
    if _name in RECENCY_CHANNELS:
        NAMES += [f"{_name}_recency", f"{_name}_age"]


class Frame:
    def __init__(self):
        self.cumulative = {}
        self.presence = {}
        self.last = {}
        self.first = {}
        for name, _ in CHANNELS:
            matrix = np.load(MAT / f"{name}.npy", mmap_mode="r")
            users, days = matrix.shape
            block = np.zeros((users, days + 1), dtype=np.float32)
            np.cumsum(matrix, axis=1, dtype=np.float32, out=block[:, 1:])
            self.cumulative[name] = block
            positive = np.asarray(matrix, dtype=np.float32) > 0
            block = np.zeros((users, days + 1), dtype=np.float32)
            np.cumsum(positive, axis=1, dtype=np.float32, out=block[:, 1:])
            self.presence[name] = block
            if name in RECENCY_CHANNELS:
                index = np.arange(days, dtype=np.float32)
                self.last[name] = np.maximum.accumulate(
                    np.where(positive, index, -1.0), axis=1
                ).astype(np.float32)
                self.first[name] = np.minimum.accumulate(
                    np.where(positive, index, np.inf), axis=1
                ).astype(np.float32)
            del matrix, positive
        self.users = users

    def at(self, anchor):
        blocks = []
        for name, log_scale in CHANNELS:
            block = self.cumulative[name]
            presence = self.presence[name]
            for w in WINDOWS:
                value = block[:, anchor + 1] - block[:, max(0, anchor - w + 1)]
                blocks.append(np.log1p(value) if log_scale else value)
            for w in WINDOWS:
                blocks.append(
                    presence[:, anchor + 1] - presence[:, max(0, anchor - w + 1)]
                )
            if name in RECENCY_CHANNELS:
                last = self.last[name][:, anchor]
                first = self.first[name][:, anchor]
                blocks.append(np.where(last < 0, 999.0, anchor - last))
                blocks.append(np.where(np.isfinite(first), anchor - first, 999.0))
        return np.column_stack(blocks).astype(np.float32)


def standardise(matrix):
    return ((matrix - matrix.mean(0)) / (matrix.std(0) + 1e-9)).astype(np.float64)
