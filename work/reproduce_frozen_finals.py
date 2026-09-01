#!/usr/bin/env python
"""Verify canonical finals and numerically rebuild the selected pair.

The canonical CSV files are hash-pinned. Solver output is compared in log1p
space because LAPACK/BLAS and the bounded optimizer are not byte-stable across
CPU architectures. Rebuilds are written only to a temporary directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "work" / "solve_augmented_stack.py"
VALIDATOR = ROOT / "work" / "validate_submissions.py"
COMMON = [
    "--extra", "83_probe_cls300", "1.6488394251718939",
    "--extra", "86_probe_cls300_probability", "1.6558577069",
    "--extra", "85_probe_w210a", "1.6482434279349687",
    "--extra", "89_probe_w300a", "1.6472946857056134",
    "--extra", "92_probe_cls409_r26", "1.647041762499095",
]
W409C = ["--extra", "102_probe_w409c", "1.646720938726788"]
TARGETS = [
    {
        "file": "147_private_safe_nonnegative_current_l001.csv",
        "primary": ["127_probe_w409_exact_decay_s93", "1.6464824096735666"],
        "args": [*COMMON, *W409C, "--lam", "0.001", "--nonnegative"],
        "sha256": "0c61bd73bfd0f699a25d893d3ba3cb762016021e446eb153fcbeef6d0bba51c2",
        "rms_log1p_max": 5e-6,
        "max_abs_log1p_max": 5e-5,
        "expected_public_delta_max": 1e-8,
        "weight_delta_max": 5e-5,
    },
    {
        "file": "200_shape_anchor_l003.csv",
        "primary": ["191_probe_shape_e4", "1.646181"],
        "args": [
            "--extra", "178_probe_divB_st24a", "1.6469745",
            "--extra", "146_probe_sw28_meanreversion", "1.6603005407",
            "--extra", "131_probe_w409_exact_position_s93", "1.6465450851",
            "--extra", "127_probe_w409_exact_decay_s93", "1.6464824096735666",
            *W409C,
            *COMMON,
            "--lam", "0.003",
        ],
        "sha256": "77b428ca6af9e74ffbdf22749cacfc87e76048d5dceef80146d557dabb21c598",
        "rms_log1p_max": 5e-11,
        "max_abs_log1p_max": 5e-10,
        "expected_public_delta_max": 1e-10,
        "weight_delta_max": 1e-9,
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(paths: list[Path]) -> None:
    subprocess.run(
        [sys.executable, str(VALIDATOR), *map(str, paths)],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def predictions(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, usecols=1)


def compare_weights(reference: dict, rebuilt: dict) -> float:
    keys = set(reference) | set(rebuilt)
    return max(
        abs(float(reference.get(key, 0.0)) - float(rebuilt.get(key, 0.0)))
        for key in keys
    )


def rebuild_one(target: dict, directory: Path) -> dict:
    canonical = ROOT / "submissions" / target["file"]
    rebuilt = directory / target["file"]
    command = [
        sys.executable,
        str(SOLVER),
        *target["primary"],
        *target["args"],
        "--output",
        str(rebuilt),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"solver failed for {target['file']}\n"
            f"stdout:\n{error.stdout}\nstderr:\n{error.stderr}"
        ) from error
    solver_report = json.loads(result.stdout)
    canonical_meta = json.loads(
        (ROOT / "work" / f"{canonical.stem}_meta.json").read_text()
    )
    canonical_values = predictions(canonical)
    rebuilt_values = predictions(rebuilt)
    log_delta = np.log1p(rebuilt_values) - np.log1p(canonical_values)
    comparison = {
        "file": target["file"],
        "canonical_sha256": sha256(canonical),
        "rebuilt_sha256": sha256(rebuilt),
        "byte_exact": sha256(canonical) == sha256(rebuilt),
        "rms_log1p_delta": float(np.sqrt(np.mean(log_delta * log_delta))),
        "max_abs_log1p_delta": float(np.max(np.abs(log_delta))),
        "expected_public_delta": float(
            solver_report["expected_public"] - canonical_meta["expected_public"]
        ),
        "max_weight_delta": compare_weights(
            canonical_meta["weights"], solver_report["weights"]
        ),
    }
    checks = {
        "rms_log1p_delta": comparison["rms_log1p_delta"]
        <= target["rms_log1p_max"],
        "max_abs_log1p_delta": comparison["max_abs_log1p_delta"]
        <= target["max_abs_log1p_max"],
        "expected_public_delta": abs(comparison["expected_public_delta"])
        <= target["expected_public_delta_max"],
        "max_weight_delta": comparison["max_weight_delta"]
        <= target["weight_delta_max"],
    }
    comparison["numerically_reproduced"] = all(checks.values())
    comparison["checks"] = checks
    if not comparison["numerically_reproduced"]:
        raise RuntimeError(f"numerical reproduction failed: {comparison}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help="validate canonical CSVs and hashes without running the solver",
    )
    args = parser.parse_args()

    canonical_paths = [ROOT / "submissions" / target["file"] for target in TARGETS]
    validate(canonical_paths)
    canonical = []
    for target, path in zip(TARGETS, canonical_paths):
        actual = sha256(path)
        if actual != target["sha256"]:
            raise RuntimeError(
                f"canonical hash mismatch for {path.name}: {actual} != {target['sha256']}"
            )
        canonical.append({"file": path.name, "sha256": actual})

    report = {
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "canonical_artifacts": canonical,
        "rebuilds": [],
    }
    if not args.artifacts_only:
        with tempfile.TemporaryDirectory(prefix="ecup_final_rebuild_") as temp:
            directory = Path(temp)
            report["rebuilds"] = [rebuild_one(target, directory) for target in TARGETS]
            validate([directory / target["file"] for target in TARGETS])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
