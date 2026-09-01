#!/usr/bin/env python
"""Rebuild and verify the two leaderboard-free provisional finals."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    ROOT / "submissions" / "120_offline_rules_safe_meanforecast.csv":
        "a17623edb7de20da05cb2de682c6ed78a9b891de112645ef3e99d07c8b6a88ae",
    ROOT / "submissions" / "122_offline_diverse_no_replica.csv":
        "218e141ec943e634d5547b16a85e0e2acbfb678b17108c440a5eca3f84299b48",
}


def run(script: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "work" / script)],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )


run("build_offline_rules_safe_final.py")
run("calibrate_offline_mean_forecast.py")
run("build_offline_diverse_challenger.py")
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
