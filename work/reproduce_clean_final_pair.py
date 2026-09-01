#!/usr/bin/env python
"""Rebuild the offline-primary and capped offline-insurance candidates."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    ROOT / "submissions" / "120_offline_rules_safe_meanforecast.csv":
        "a17623edb7de20da05cb2de682c6ed78a9b891de112645ef3e99d07c8b6a88ae",
    ROOT / "submissions" / "123_offline_capped_w035.csv":
        "4a698bb23242ad19fd0edac4cab5c318c63544ea8012b97895f6fc86f36ac599",
}


for script in (
    "build_offline_rules_safe_final.py",
    "calibrate_offline_mean_forecast.py",
    "build_offline_capped_challenger.py",
):
    subprocess.run(
        [sys.executable, str(ROOT / "work" / script)],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )

subprocess.run(
    [
        sys.executable,
        str(ROOT / "work" / "validate_submissions.py"),
        *[str(path) for path in EXPECTED],
    ],
    cwd=ROOT,
    check=True,
)
for path, expected in EXPECTED.items():
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {path.name}: {actual} != {expected}"
        )
    print(f"OK {path.name} sha256={actual}")
