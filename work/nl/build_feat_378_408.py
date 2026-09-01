#!/usr/bin/env python
"""Materialise the shared historical features at the audit fold and at the
competition anchor, using one definition for both."""
from pathlib import Path

import numpy as np

import histfeat

OUT = Path(__file__).resolve().parent
frame = histfeat.Frame()
for anchor in (378, 408):
    values = frame.at(anchor)
    np.save(OUT / f"hist{anchor}.npy", values)
    print(anchor, values.shape, "finite", bool(np.isfinite(values).all()))
(OUT / "hist_keys.json").write_text(
    "[\n" + ",\n".join(f'  "{k}"' for k in histfeat.NAMES) + "\n]\n"
)
print("columns", len(histfeat.NAMES))
