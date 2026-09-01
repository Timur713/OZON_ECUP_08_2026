"""Extended historical-feature definition, streamed one channel at a time.

Adds the funnel-split and conversion channels that 153 never saw, plus ratio
families that are provably outside the linear span of raw window aggregates.
Channels are loaded and released one at a time, so the peak footprint is one
cumulative array rather than sixteen.  Uses only days <= anchor.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[2]))
MAT = ROOT / "work" / "mat"
WINDOWS = (7, 14, 30, 60, 90, 180, 365)
TREND_WINDOWS = (7, 30, 90)
# log-scaled channels are money; the rest are counts
CHANNELS = (
    ("gmv", True), ("gmv_search", True), ("gmv_cat", True),
    ("to_ord", False), ("to_cart", False), ("searches", False),
    ("search", False), ("cat", False),
    ("search_to_ord", False), ("cat_to_ord", False), ("active", False),
)
RECENCY_CHANNELS = ("gmv", "to_ord", "active", "to_cart")
FLOOR = 1.0


def _windows(cumulative, anchor):
    return {w: cumulative[:, anchor + 1] - cumulative[:, max(0, anchor - w + 1)]
            for w in WINDOWS}


def build(anchors):
    """Return {anchor: (matrix, names)} for the requested anchors."""
    anchors = list(anchors)
    sums = {a: {} for a in anchors}
    days = {a: {} for a in anchors}
    extra = {a: [] for a in anchors}
    extra_names = []
    for name, log_scale in CHANNELS:
        matrix = np.load(MAT / f"{name}.npy", mmap_mode="r")
        users, span = matrix.shape
        block = np.zeros((users, span + 1), dtype=np.float32)
        np.cumsum(matrix, axis=1, dtype=np.float32, out=block[:, 1:])
        positive = np.asarray(matrix, dtype=np.float32) > 0
        presence = np.zeros((users, span + 1), dtype=np.float32)
        np.cumsum(positive, axis=1, dtype=np.float32, out=presence[:, 1:])
        last = first = None
        if name in RECENCY_CHANNELS:
            index = np.arange(span, dtype=np.float32)
            last = np.maximum.accumulate(np.where(positive, index, -1.0), axis=1)
            first = np.minimum.accumulate(np.where(positive, index, np.inf), axis=1)
        for anchor in anchors:
            sums[anchor][name] = {
                w: (np.log1p(v) if log_scale else v)
                for w, v in _windows(block, anchor).items()
            }
            sums[anchor][name]["_raw"] = _windows(block, anchor)
            days[anchor][name] = _windows(presence, anchor)
            if last is not None:
                value = last[:, anchor]
                extra[anchor].append(np.where(value < 0, 999.0, anchor - value))
                value = first[:, anchor]
                extra[anchor].append(
                    np.where(np.isfinite(value), anchor - value, 999.0)
                )
        if name in RECENCY_CHANNELS:
            extra_names += [f"{name}_recency", f"{name}_age"]
        del matrix, positive, block, presence, last, first

    result = {}
    for anchor in anchors:
        blocks, names = [], []
        for name, _ in CHANNELS:
            for w in WINDOWS:
                blocks.append(sums[anchor][name][w]); names.append(f"{name}_s{w}")
            for w in WINDOWS:
                blocks.append(days[anchor][name][w]); names.append(f"{name}_d{w}")
        blocks += extra[anchor]
        names += extra_names
        raw = {name: sums[anchor][name]["_raw"] for name, _ in CHANNELS}
        for w in WINDOWS:
            blocks.append(raw["to_ord"][w] / (raw["searches"][w] + FLOOR))
            names.append(f"conv_ord_per_search_{w}")
            blocks.append(raw["to_cart"][w] / (raw["searches"][w] + FLOOR))
            names.append(f"conv_cart_per_search_{w}")
            blocks.append(raw["to_ord"][w] / (raw["to_cart"][w] + FLOOR))
            names.append(f"conv_ord_per_cart_{w}")
            blocks.append(np.log1p(raw["gmv"][w] / (raw["to_ord"][w] + FLOOR)))
            names.append(f"check_{w}")
            blocks.append(raw["gmv_search"][w] / (raw["gmv"][w] + FLOOR))
            names.append(f"search_share_gmv_{w}")
            blocks.append(raw["to_ord"][w] / (days[anchor]["active"][w] + FLOOR))
            names.append(f"orders_per_active_day_{w}")
        for w in TREND_WINDOWS:
            scale = 365.0 / w
            for channel in ("gmv", "to_ord", "searches"):
                blocks.append(
                    raw[channel][w] * scale / (raw[channel][365] + FLOOR)
                )
                names.append(f"trend_{channel}_{w}_over_365")
        result[anchor] = (
            np.column_stack(blocks).astype(np.float32), names,
        )
        del blocks
    return result


def names():
    return build([0])[0][1]
