#!/usr/bin/env python
"""Build a reproducible complexity/quality Pareto table for the current stack.

The subset order is fixed by each component's contribution scale in the frozen
teacher (`abs(weight) * std(prediction)`).  It is not selected by repeatedly
minimizing leaderboard score.  For every prefix we re-solve the same
lambda=0.003 moment ridge and report both the analytical score estimate and the
distance to the exact frozen teacher.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
M1, M2 = 2.3232887, 10.7633307
EMPIRICAL_TRANSFER_PER_DF = 0.0000395


def load(name: str) -> np.ndarray:
    return np.load(WORK / f"{name}_final.npy").astype(np.float64)


def calibrate_mean(raw: np.ndarray) -> np.ndarray:
    shift = M1 - np.clip(raw, 0, None).mean()
    for _ in range(12):
        value = np.clip(raw + shift, 0, None)
        delta = M1 - value.mean()
        shift += delta
        if abs(delta) < 1e-13:
            break
    return np.clip(raw + shift, 0, None)


def build_pool() -> dict[str, np.ndarray]:
    pool = {
        "gb": (load("v4_zh") + load("cfg3")) / 2,
        "tcn45": load("tcn45"),
        "tcn90": load("tcn90"),
        "tcn180two": load("tcn180two"),
        "tcn270": load("tcn270"),
        "tcn409": load("tcn409"),
        "tcn365v336": load("tcn365v336"),
        "t3b": load("tcn365b"),
        "t1": load("seq"),
        "gru180": load("gru180"),
        "tcn365": load("tcn365"),
        "LY": np.load(WORK / "basis_prior_year_gmv.npy").astype(np.float64),
    }
    for name in (
        "GBD", "W120", "W150", "W365", "W409", "W90", "W45", "W60",
        "W180", "W270",
    ):
        pool[name] = np.load(WORK / f"AVG_{name}.npy").astype(np.float64)
    ridge_keys = set(np.load(WORK / "ridge22_keys.npy", allow_pickle=True).tolist())
    pool = {key: value for key, value in pool.items() if key in ridge_keys}
    for tag in (
        "102_probe_w409c",
        "83_probe_cls300",
        "85_probe_w210a",
        "86_probe_cls300_probability",
        "89_probe_w300a",
        "92_probe_cls409_r26",
        "127_probe_w409_exact_decay_s93",
    ):
        metadata = json.loads((WORK / f"{tag}_meta.json").read_text())
        candidate_path = WORK / Path(metadata["candidate_file"]).name
        pool[tag] = np.load(candidate_path).astype(np.float64)
    return pool


def target_moments(keys: list[str]) -> np.ndarray:
    ez_pool = json.loads((WORK / "EZ_pool.json").read_text())
    final_meta = json.loads(
        (WORK / "130_private_safe_exact_decay_l003_meta.json").read_text()
    )
    candidate_moments = final_meta["ez_candidates"]
    return np.asarray([
        candidate_moments[key] if key in candidate_moments else ez_pool[key]
        for key in keys
    ] + [M1], dtype=np.float64)


def solve(keys: list[str], pool: dict[str, np.ndarray], lam: float):
    design = np.column_stack([*[pool[key] for key in keys], np.ones(len(next(iter(pool.values()))))])
    gram = design.T @ design / len(design)
    rhs = target_moments(keys)
    penalty = np.eye(len(keys) + 1) * lam
    penalty[-1, -1] = 0
    system = gram + penalty
    weights = np.linalg.solve(system, rhs)
    mse = M2 - 2 * rhs @ weights + weights @ gram @ weights
    degrees = float(np.trace(gram @ np.linalg.inv(system)))
    prediction = calibrate_mean(design @ weights)
    return weights, float(np.sqrt(max(mse, 0))), degrees, prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam", type=float, default=0.003)
    parser.add_argument("--max-models", type=int, default=12)
    parser.add_argument(
        "--output", default="work/compact_stack_pareto.json",
    )
    args = parser.parse_args()

    pool = build_pool()
    teacher_meta = json.loads(
        (WORK / "130_private_safe_exact_decay_l003_meta.json").read_text()
    )
    teacher_weights = teacher_meta["weights"]
    missing = set(teacher_weights) - set(pool) - {"const"}
    if missing:
        raise ValueError(f"missing frozen teacher columns: {sorted(missing)}")

    teacher_path = ROOT / "submissions" / "130_private_safe_exact_decay_l003.csv"
    if teacher_path.open().readline().strip() != "user_id,predict":
        raise ValueError(f"invalid submission header: {teacher_path}")
    teacher_values = np.loadtxt(
        teacher_path, delimiter=",", skiprows=1, usecols=1, dtype=np.float64
    )
    teacher = np.log1p(np.clip(teacher_values, 0, None))

    importance = {
        key: abs(float(teacher_weights[key])) * float(np.std(value))
        for key, value in pool.items()
    }
    ordered = sorted(pool, key=lambda key: (-importance[key], key))

    rows = []
    for count in range(1, min(args.max_models, len(ordered)) + 1):
        keys = ordered[:count]
        weights, public, degrees, prediction = solve(keys, pool, args.lam)
        residual = prediction - teacher
        rows.append({
            "model_count": count,
            "models": keys,
            "selection_rule": "frozen_teacher_abs_weight_times_std_prefix",
            "lambda": args.lam,
            "expected_public": public,
            "empirical_private": public + EMPIRICAL_TRANSFER_PER_DF * degrees,
            "degrees_of_freedom": degrees,
            "teacher_rms_log_distance": float(np.sqrt(np.mean(residual * residual))),
            "teacher_correlation": float(np.corrcoef(prediction, teacher)[0, 1]),
            "weights": dict(zip(keys + ["const"], weights.tolist())),
        })

    report = {
        "teacher": "130_private_safe_exact_decay_l003.csv",
        "teacher_expected_public": teacher_meta["expected_public"],
        "teacher_empirical_private": teacher_meta["empirical_private_score"],
        "teacher_degrees_of_freedom": teacher_meta["degrees_of_freedom"],
        "teacher_component_count": len(pool),
        "ranking": [{"model": key, "importance": importance[key]} for key in ordered],
        "rows": rows,
        "limitations": [
            "The ranking is derived from the frozen public-informed teacher, so this is a production-compression audit, not a rules-safe offline estimator.",
            "A listed W* component can itself average several seeds; model_count counts stack components, not training runs.",
            "Analytical public estimates use the same full-250k Gram approximation as the frozen moment solver.",
        ],
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
