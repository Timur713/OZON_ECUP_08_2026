#!/usr/bin/env python
"""Fail fast on malformed plain-CSV competition submissions."""
import argparse
import json
import os
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+")
parser.add_argument("--uids", default="work/mat/uids.npy")
args = parser.parse_args()
expected_uids = np.load(args.uids)
reports = []
for path in args.files:
    if path.endswith(".gz"):
        raise ValueError(f"compressed submissions are forbidden: {path}")
    file_path = Path(path)
    with file_path.open() as stream:
        if stream.readline().strip() != "user_id,predict":
            raise ValueError(f"invalid columns in {path}")
    values = np.loadtxt(file_path, delimiter=",", skiprows=1, dtype=np.float64)
    if values.shape != (len(expected_uids), 2):
        raise ValueError(f"invalid shape in {path}: {values.shape}")
    user_ids = values[:, 0].astype(expected_uids.dtype)
    predictions = values[:, 1]
    if not np.array_equal(user_ids, expected_uids):
        raise ValueError(f"user order/content differs from matrix uids: {path}")
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError(f"non-finite or negative predictions: {path}")
    reports.append({
        "file": os.path.abspath(path),
        "rows": len(values),
        "unique_users": int(np.unique(user_ids).size),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "mean_log1p": float(np.log1p(predictions).mean()),
        "bytes": os.path.getsize(path),
    })
print(json.dumps(reports, indent=2))
