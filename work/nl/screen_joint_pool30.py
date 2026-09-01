#!/usr/bin/env python
"""179 - how much does a SET of new bases add together?

Five siblings that each add 0.00012 on their own do not add 0.0006 together if
they are contrasts of the same kind. Probes are the scarce resource, so the
question that decides how many to spend is the joint gain and the incremental
gain of each member given the others.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
LAM = 0.003
REPEATS = 96
PUBLIC_USERS = 50_000
SEED = 20260828

parser = argparse.ArgumentParser()
parser.add_argument("candidates", nargs="+")
parser.add_argument("--out", default="179_joint.json")
args = parser.parse_args()

pool = np.load(OUT / "pool30_378.npy").astype(np.float64)
truth = np.load(OUT / "truth378.npy").astype(np.float64)
n = len(truth)
ones = np.ones((n, 1))
all_index = np.arange(n)
rng = np.random.default_rng(SEED)
splits = []
for _ in range(REPEATS):
    public = rng.choice(n, PUBLIC_USERS, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[public] = False
    splits.append((public, all_index[mask]))

names = [Path(p).stem for p in args.candidates]
vectors = [np.load(p).astype(np.float64) for p in args.candidates]


def evaluate(columns):
    design = np.hstack([pool] + [v[:, None] for v in columns] + [ones])
    size = design.shape[1]
    penalty = np.eye(size) * LAM
    penalty[-1, -1] = 0.0
    private = np.empty(REPEATS)
    degrees = np.empty(REPEATS)
    for i, (public, hidden) in enumerate(splits):
        x = design[public]
        gram = x.T @ x / len(public)
        weights = np.linalg.solve(
            gram + penalty, x.T @ truth[public] / len(public)
        )
        residual = truth[hidden] - design[hidden] @ weights
        private[i] = np.sqrt(np.mean(residual * residual))
        degrees[i] = float(np.trace(gram @ np.linalg.inv(gram + penalty)))
    return private, degrees.mean()


base_private, base_df = evaluate([])
report = {"base_private": float(base_private.mean()), "base_df": base_df,
          "singles": {}, "joint": {}, "leave_one_out": {}}
for name, vector in zip(names, vectors):
    private, df = evaluate([vector])
    gain = base_private - private
    report["singles"][name] = {
        "gain": float(gain.mean()), "df_added": df - base_df,
        "positive_splits": int((gain > 0).sum()),
    }
    print(f"single {name:26s} gain={gain.mean():+.7f} df+={df - base_df:.3f}",
          flush=True)

joint_private, joint_df = evaluate(vectors)
joint_gain = base_private - joint_private
report["joint"] = {
    "members": names, "gain": float(joint_gain.mean()),
    "df_added": joint_df - base_df,
    "positive_splits": int((joint_gain > 0).sum()),
    "sum_of_singles": sum(v["gain"] for v in report["singles"].values()),
}
print(f"\njoint of {len(names)}: gain={joint_gain.mean():+.7f} "
      f"df+={joint_df - base_df:.3f} "
      f"(sum of singles {report['joint']['sum_of_singles']:+.7f})", flush=True)

if len(vectors) > 1:
    for index, name in enumerate(names):
        rest = [v for j, v in enumerate(vectors) if j != index]
        private, df = evaluate(rest)
        incremental = float((private - joint_private).mean())
        report["leave_one_out"][name] = {
            "incremental_gain_given_the_others": incremental,
            "df_it_costs": joint_df - df,
        }
        print(f"  {name:26s} adds {incremental:+.7f} beyond the others "
              f"for {joint_df - df:.3f} df", flush=True)

(OUT / args.out).write_text(json.dumps(report, indent=2) + "\n")
