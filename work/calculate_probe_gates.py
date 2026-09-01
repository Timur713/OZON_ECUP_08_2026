#!/usr/bin/env python
"""Calculate pre-public positive and strict gates for a measured probe.

The design matrix and all extra measured moments are frozen. Only the primary
probe score varies, and roots are accepted solely on its positive-weight
branch. No submission or metadata is written.
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import brentq


ROOT = os.environ.get(
    "ECUP_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
).rstrip("/") + os.sep
M1, M2 = 2.3232887, 10.7633307


def resolve_artifact(path):
    """Locate an artifact recorded with an absolute path from another checkout.

    Probe metadata written before the repository was cloned stores absolute
    paths.  Those paths are part of the frozen record and are not rewritten;
    instead a missing one is looked up by name under this checkout, so the
    frozen solves reproduce on any machine.
    """
    if os.path.exists(path):
        return path
    local = ROOT + "work/" + os.path.basename(path)
    if os.path.exists(local):
        return local
    raise FileNotFoundError(f"artifact not found here or at its recorded path: {path}")



parser = argparse.ArgumentParser()
parser.add_argument("probe_tag")
parser.add_argument(
    "--extra", nargs=2, action="append", default=[], metavar=("TAG", "SCORE")
)
parser.add_argument("--lam", type=float, default=0.003)
parser.add_argument("--empirical-per-degree", type=float, default=0.0000395)
parser.add_argument(
    "--adaptive-cost",
    type=float,
    default=0.0,
    help="fixed extra marginal-gain cost for pre-measurement selection bias",
)
parser.add_argument("--score-min", type=float, default=1.40)
parser.add_argument("--score-max", type=float, default=1.90)
# Some useful positive-weight branches are narrower than 5e-4 RMSLE.  A coarse
# grid can step over both roots and incorrectly report a missing gate, so keep
# the default spacing at 2.5e-5 over the standard 1.40..1.90 interval.
parser.add_argument("--grid", type=int, default=20001)
# A gate is only meaningful against the pool that the CURRENT primary submission
# actually uses.  Omitting an already admitted base silently restores an older,
# weaker baseline and makes the gate too lenient, so the caller must state the
# baseline it expects and the run aborts on any mismatch.
parser.add_argument(
    "--expect-baseline",
    type=float,
    default=None,
    help="required marginal_baseline_expected_public of the current primary",
)
parser.add_argument("--baseline-tolerance", type=float, default=1e-9)
# A base that enters as a CONTRAST against bases already in the pool carries a
# stable negative coefficient.  The sign is a property of the parameterisation,
# not of the signal, so the gate must be allowed to sit on that branch when the
# offline audit measured a consistently negative weight before any probe.
parser.add_argument(
    "--allow-negative-weight",
    action="store_true",
    help="accept gate roots whose primary ridge weight is negative",
)
args = parser.parse_args()
if args.adaptive_cost < 0:
    raise ValueError("--adaptive-cost must be nonnegative")

probe_inputs = [(args.probe_tag, None)] + [
    (tag, float(score)) for tag, score in args.extra
]
candidate_vectors = {}
candidate_metadata = {}
fixed_moments = {}
for tag, score in probe_inputs:
    metadata = json.load(open(ROOT + f"work/{tag}_meta.json"))
    candidate_metadata[tag] = metadata
    candidate_vectors[tag] = np.load(resolve_artifact(metadata["candidate_file"])).astype(np.float64)
    if score is not None:
        ez_probe = (
            M2 + metadata["probe_second_moment"] - score * score
        ) / 2
        fixed_moments[tag] = (
            ez_probe - (1 - metadata["weight"]) * metadata["ez_base"]
        ) / metadata["weight"]

ez_pool = json.load(open(ROOT + "work/EZ_pool.json"))
load = lambda name: np.load(ROOT + f"work/{name}_final.npy").astype(np.float64)
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
    "a409a": load("a409a"),
    "LY": np.load(ROOT + "work/basis_prior_year_gmv.npy").astype(np.float64),
}
for name in (
    "GBD", "W120", "W150", "W365", "W409", "W90", "W45", "W60",
    "W180", "W270",
):
    pool[name] = np.load(ROOT + f"work/AVG_{name}.npy").astype(np.float64)
ridge_keys = set(np.load(ROOT + "work/ridge22_keys.npy").tolist())
pool = {key: value for key, value in pool.items() if key in ridge_keys}
pool.update(candidate_vectors)

keys = sorted(pool)
reference = candidate_vectors[args.probe_tag]
design = np.vstack([pool[key] for key in keys] + [np.ones_like(reference)]).T
gram = design.T @ design / len(reference)
penalty = np.eye(len(keys) + 1) * args.lam
penalty[-1, -1] = 0
system = gram + penalty
degrees = float(np.trace(gram @ np.linalg.inv(system)))
primary_index = keys.index(args.probe_tag)

marginal_keys = [key for key in keys if key != args.probe_tag]
marginal_design = np.vstack([
    *[pool[key] for key in marginal_keys], np.ones_like(reference),
]).T
marginal_gram = marginal_design.T @ marginal_design / len(reference)
marginal_penalty = np.eye(len(marginal_keys) + 1) * args.lam
marginal_penalty[-1, -1] = 0
marginal_rhs = np.array([
    fixed_moments[key] if key in fixed_moments else ez_pool[key]
    for key in marginal_keys
] + [M1])
marginal_coefficients = np.linalg.solve(
    marginal_gram + marginal_penalty, marginal_rhs
)
marginal_mse = (
    M2
    - 2 * marginal_rhs @ marginal_coefficients
    + marginal_coefficients @ marginal_gram @ marginal_coefficients
)
marginal_expected = float(np.sqrt(max(marginal_mse, 0)))
marginal_degrees = float(np.trace(
    marginal_gram @ np.linalg.inv(marginal_gram + marginal_penalty)
))
degrees_added = degrees - marginal_degrees

if args.expect_baseline is not None:
    drift = abs(marginal_expected - args.expect_baseline)
    if drift > args.baseline_tolerance:
        raise SystemExit(
            f"marginal pool drift: baseline {marginal_expected!r} differs from "
            f"expected {args.expect_baseline!r} by {drift:.3e}; an admitted "
            "base is probably missing from --extra"
        )


def primary_moment(score):
    metadata = candidate_metadata[args.probe_tag]
    ez_probe = (M2 + metadata["probe_second_moment"] - score * score) / 2
    return (
        ez_probe - (1 - metadata["weight"]) * metadata["ez_base"]
    ) / metadata["weight"]


def evaluate(score):
    rhs = np.array([
        primary_moment(score) if key == args.probe_tag
        else fixed_moments[key] if key in fixed_moments
        else ez_pool[key]
        for key in keys
    ] + [M1])
    coefficients = np.linalg.solve(system, rhs)
    mse = M2 - 2 * rhs @ coefficients + coefficients @ gram @ coefficients
    expected = float(np.sqrt(max(mse, 0)))
    return {
        "score": float(score),
        "expected_public": expected,
        "marginal_public_gain": marginal_expected - expected,
        "primary_weight": float(coefficients[primary_index]),
    }


grid = np.linspace(args.score_min, args.score_max, args.grid)


def roots_for(function, require_positive_weight=True):
    values = [function(value) for value in grid]
    roots = []
    for left, right, f_left, f_right in zip(
        grid[:-1], grid[1:], values[:-1], values[1:]
    ):
        if f_left == 0:
            root = float(left)
        elif f_left * f_right > 0:
            continue
        else:
            root = float(brentq(function, left, right))
        if not require_positive_weight or evaluate(root)["primary_weight"] > 0:
            roots.append(root)
    return sorted(set(round(root, 12) for root in roots))


neutral_roots = roots_for(
    lambda score: evaluate(score)["primary_weight"],
    require_positive_weight=False,
)
positive_cost = degrees_added * args.empirical_per_degree
require_sign = not args.allow_negative_weight
positive_roots = roots_for(
    lambda score: evaluate(score)["marginal_public_gain"] - positive_cost,
    require_positive_weight=require_sign,
)
strict_roots = roots_for(
    lambda score: evaluate(score)["marginal_public_gain"] - 2 * positive_cost,
    require_positive_weight=require_sign,
)
strict_adaptive_cost = 2 * positive_cost + args.adaptive_cost
strict_adaptive_roots = roots_for(
    lambda score: evaluate(score)["marginal_public_gain"] - strict_adaptive_cost,
    require_positive_weight=require_sign,
)


def upper_gate(roots):
    return max(roots) if roots else None


report = {
    "probe_tag": args.probe_tag,
    "extra_scores": dict(args.extra),
    "lambda": args.lam,
    "marginal_baseline_expected_public": marginal_expected,
    "marginal_degrees_added": degrees_added,
    "empirical_penalty_per_degree": args.empirical_per_degree,
    "positive_empirical_cost": positive_cost,
    "strict_double_empirical_cost": 2 * positive_cost,
    "adaptive_selection_cost": args.adaptive_cost,
    "strict_with_adaptive_cost": strict_adaptive_cost,
    "positive_empirical_net_gate": upper_gate(positive_roots),
    "strict_gate": upper_gate(strict_roots),
    "strict_with_adaptive_gate": upper_gate(strict_adaptive_roots),
    "positive_weight_neutral_score": upper_gate(neutral_roots),
    "positive_gate_state": (
        evaluate(upper_gate(positive_roots)) if positive_roots else None
    ),
    "strict_gate_state": (
        evaluate(upper_gate(strict_roots)) if strict_roots else None
    ),
    "strict_with_adaptive_gate_state": (
        evaluate(upper_gate(strict_adaptive_roots))
        if strict_adaptive_roots else None
    ),
    "adaptive_selection_bias_included": args.adaptive_cost > 0,
    "negative_weight_branch_allowed": args.allow_negative_weight,
}
print(json.dumps(report, indent=2))
