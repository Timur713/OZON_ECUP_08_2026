#!/usr/bin/env python
"""Turn returned 153 probe scores into a decision, using the frozen solver.

Usage:
    python work/nl/apply_153_scores.py work/153_probe_scores.json [--lam 0.003]

The scores file is a flat mapping of probe tag to public RMSLE, for example
    {"153_probe_hist01_gmv_d365": 1.66xxxx, "153_probe_hist02_active_recency": ...}
Only the probes present in the file are admitted, and the block is evaluated at
exactly that K.  This wrapper does not pick a subset: it uses every score it is
given, in rank order, which is what the frozen key requires.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADMITTED = {
    "127_probe_w409_exact_decay_s93": 1.6464824096735666,
    "102_probe_w409c": 1.646720938726788,
    "83_probe_cls300": 1.6488394251718939,
    "86_probe_cls300_probability": 1.6558577069,
    "85_probe_w210a": 1.6482434279349687,
    "89_probe_w300a": 1.6472946857056134,
    "92_probe_cls409_r26": 1.647041762499095,
}
BASELINE_EXPECTED_PUBLIC = 1.6461788499434364
EMPIRICAL_PER_DF = 0.0000395
GATE_PRIVATE_REQUIREMENT = 0.00016

parser = argparse.ArgumentParser()
parser.add_argument("scores")
parser.add_argument("--lam", type=float, default=0.003)
parser.add_argument("--output", default=None)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

scores = json.loads(Path(args.scores).read_text())
manifest = json.loads((ROOT / "work" / "153_probe_manifest.json").read_text())
ordered = [row["tag"] for row in manifest["probes"] if row["tag"] in scores]
missing = sorted(set(scores) - set(ordered))
if missing:
    raise SystemExit(f"unknown probe tags in scores file: {missing}")
if not ordered:
    raise SystemExit("no 153 probe scores supplied")
k = len(ordered)
expected = [row["tag"] for row in manifest["probes"][:k]]
if ordered != expected:
    raise SystemExit(
        "scores must be a rank-ordered prefix of the manifest; the frozen key "
        f"forbids choosing a subset.\n  given: {ordered}\n  needed: {expected}"
    )

command = [
    sys.executable, str(ROOT / "work" / "solve_augmented_stack.py"),
    ordered[0], str(scores[ordered[0]]), "--lam", str(args.lam),
]
for tag, score in ADMITTED.items():
    command += ["--extra", tag, str(score)]
for tag in ordered[1:]:
    command += ["--extra", tag, str(scores[tag])]
output_name = args.output or f"154_conditional_block_k{k}_l{args.lam:g}.csv"
if args.dry_run:
    command.append("--dry-run")
else:
    command += ["--output", output_name]

print("running:", " ".join(command), file=sys.stderr)
result = subprocess.run(command, capture_output=True, text=True)
if result.returncode != 0:
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    raise SystemExit(result.returncode)
report = json.loads(result.stdout)

expected_public = report["expected_public"]
public_gain = BASELINE_EXPECTED_PUBLIC - expected_public
degrees_added = report["degrees_of_freedom"] - 20.530728291708222
private_estimate_gain = public_gain - degrees_added * EMPIRICAL_PER_DF
verdict = "ADMIT" if private_estimate_gain >= GATE_PRIVATE_REQUIREMENT else "REJECT"
decision = {
    "tag": "153_conditional_block",
    "k": k,
    "columns": [row["column"] for row in manifest["probes"][:k]],
    "lambda": args.lam,
    "baseline_expected_public_of_130": BASELINE_EXPECTED_PUBLIC,
    "expected_public_of_block": expected_public,
    "public_gain": public_gain,
    "degrees_of_freedom": report["degrees_of_freedom"],
    "degrees_added_over_130": degrees_added,
    "empirical_private_penalty_of_added_df": degrees_added * EMPIRICAL_PER_DF,
    "private_gain_estimate": private_estimate_gain,
    "gate_requirement": GATE_PRIVATE_REQUIREMENT,
    "gate_equivalent_public_gain": GATE_PRIVATE_REQUIREMENT + degrees_added * EMPIRICAL_PER_DF,
    "verdict": verdict,
    "empirical_private_score": report["empirical_private_score"],
    "empirical_private_score_of_130": 1.6469898137109589,
    "solver_report": report,
}
path = ROOT / "work" / f"153_block_decision_k{k}.json"
path.write_text(json.dumps(decision, indent=2) + "\n")
print(json.dumps({key: value for key, value in decision.items()
                  if key != "solver_report"}, indent=2))
print(f"\nwritten {path}")
