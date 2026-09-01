#!/usr/bin/env python
"""Drop exactly-duplicated historical columns before spending any probe.

gmv_d{w} counts days with positive GMV and to_ord_d{w} counts days with a
positive order; a day has one exactly when it has the other, so the two are the
same vector.  The same happens for active_s{w} against active_d{w}, because the
active channel is already binary, and for the gmv and order recencies.  Probing
a duplicate would buy a moment we already own.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
names = json.loads((OUT / "hist_keys.json").read_text())
hist = np.load(OUT / "hist408.npy").astype(np.float64)
standard = (hist - hist.mean(0)) / (hist.std(0) + 1e-9)

keep = []
duplicates = {}
for index in range(standard.shape[1]):
    match = None
    for kept in keep:
        if abs(float(np.corrcoef(standard[:, index], standard[:, kept])[0, 1])) > 0.99995:
            match = kept
            break
    if match is None:
        keep.append(index)
    else:
        duplicates.setdefault(names[match], []).append(names[index])

report = {
    "original_columns": len(names),
    "unique_columns": len(keep),
    "kept_indices": keep,
    "kept_names": [names[i] for i in keep],
    "duplicate_groups": duplicates,
}
(OUT / "hist_unique.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"{len(names)} columns -> {len(keep)} unique")
for head, rest in duplicates.items():
    print(f"  {head} == {', '.join(rest)}")
