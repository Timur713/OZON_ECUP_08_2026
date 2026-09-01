#!/usr/bin/env python
"""H4 — the one cross-user base: cohort seasonal lift from the 2025 gift window.

Every base in the pool looks at one user's own history.  The 2025 gift window,
days 44 to 73, is the only stretch of labelled data that shares the target
window's calendar position.  Its per-user signal was measured at +0.00001 and
is useless, but a COHORT statistic over it is estimable: cluster users on their
normalised behaviour profile, measure each cluster's realised lift from its
pre-window level into that window, and give a user its cluster's lift applied to
its own current level.

The vector is produced at both anchors with the same code; the clustering and
the lift table are computed only from days <= 73, so nothing downstream of the
2025 gift window enters the anchor-378 version.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("ECUP_ROOT", Path(__file__).resolve().parents[2]))
MAT = Path(os.environ.get("ECUP_MAT", ROOT / "work" / "mat"))
OUT = Path(os.environ.get("ECUP_OUT", ROOT / "work" / "cand"))
OUT.mkdir(parents=True, exist_ok=True)
GIFT = (44, 74)          # 2025-02-14 .. 2025-03-15 inclusive
PRE = (14, 44)           # the 30 days before it
CLUSTERS = 64
SEED = 161

gmv = np.load(MAT / "gmv.npy", mmap_mode="r")
orders = np.load(MAT / "to_ord.npy", mmap_mode="r")
searches = np.load(MAT / "searches.npy", mmap_mode="r")
users = gmv.shape[0]

pre_level = np.log1p(gmv[:, PRE[0]:PRE[1]].sum(axis=1, dtype=np.float64))
gift_level = np.log1p(gmv[:, GIFT[0]:GIFT[1]].sum(axis=1, dtype=np.float64))

# Profile for clustering: shape only, from the pre-window period, so the
# clustering itself never reads the gift window.
profile = np.column_stack([
    pre_level,
    np.log1p(orders[:, PRE[0]:PRE[1]].sum(axis=1, dtype=np.float64)),
    np.log1p(searches[:, PRE[0]:PRE[1]].sum(axis=1, dtype=np.float64)),
    (np.asarray(gmv[:, PRE[0]:PRE[1]], dtype=np.float32) > 0).sum(axis=1),
    (np.asarray(searches[:, PRE[0]:PRE[1]], dtype=np.float32) > 0).sum(axis=1),
])
profile = (profile - profile.mean(0)) / (profile.std(0) + 1e-9)

rng = np.random.default_rng(SEED)
centroids = profile[rng.choice(users, CLUSTERS, replace=False)]
for _ in range(25):
    distance = (
        (profile ** 2).sum(1)[:, None]
        - 2 * profile @ centroids.T
        + (centroids ** 2).sum(1)[None, :]
    )
    assignment = distance.argmin(1)
    for c in range(CLUSTERS):
        members = assignment == c
        if members.any():
            centroids[c] = profile[members].mean(0)

# Most users are inactive, so the profile space is degenerate and some
# centroids can end up with no members.  An empty cluster gets a zero lift and
# is never assigned to anyone, so it only has to be kept out of the statistics.
counts = np.bincount(assignment, minlength=CLUSTERS)
lift = np.zeros(CLUSTERS)
for c in range(CLUSTERS):
    if counts[c] == 0:
        continue
    members = assignment == c
    lift[c] = float(gift_level[members].mean() - pre_level[members].mean())
occupied = counts > 0

report = {
    "clusters": CLUSTERS,
    "occupied_clusters": int(occupied.sum()),
    "lift_min": float(lift[occupied].min()), "lift_max": float(lift[occupied].max()),
    "lift_mean": float(lift[occupied].mean()), "lift_sd": float(lift[occupied].std()),
    "smallest_occupied_cluster": int(counts[occupied].min()),
}
np.save(OUT / "h4_cluster_assignment.npy", assignment.astype(np.int16))
np.save(OUT / "h4_cluster_lift.npy", lift)

for anchor, suffix in ((378, "val"), (408, "final")):
    level = np.log1p(
        gmv[:, anchor - 29:anchor + 1].sum(axis=1, dtype=np.float64)
    )
    prediction = np.clip(level + lift[assignment], 0.0, None).astype(np.float32)
    np.save(OUT / f"h4_cohortlift_{suffix}.npy", prediction)
    report[f"{suffix}_mean"] = float(prediction.mean())
    report[f"{suffix}_sd"] = float(prediction.std())

(OUT / "h4_cohort_lift_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
