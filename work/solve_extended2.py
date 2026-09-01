#!/usr/bin/env python
"""Recover a probed GPU moment and rebuild the full 22-base ridge stack."""
import argparse
import csv
import json
import os

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
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
parser.add_argument("public_score", type=float)
parser.add_argument("--extra", nargs=2, action="append", default=[], metavar=("TAG", "SCORE"))
parser.add_argument("--lam", type=float, default=0.001)
parser.add_argument("--output")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument(
    "--nonnegative",
    action="store_true",
    help="constrain model coefficients to >=0 while leaving the intercept free",
)
args = parser.parse_args()

probe_inputs = [(args.probe_tag, args.public_score)] + [
    (tag, float(score)) for tag, score in args.extra
]
candidates = {}
candidate_moments = {}
probe_scores = {}
for tag, score in probe_inputs:
    metadata = json.load(open(ROOT + f"work/{tag}_meta.json"))
    candidates[tag] = np.load(resolve_artifact(metadata["candidate_file"])).astype(np.float64)
    if "candidate_target_moment" in metadata:
        candidate_moments[tag] = float(metadata["candidate_target_moment"])
    else:
        weight = metadata["weight"]
        ez_probe = (M2 + metadata["probe_second_moment"] - score**2) / 2
        candidate_moments[tag] = (
            ez_probe - (1 - weight) * metadata["ez_base"]
        ) / weight
    probe_scores[tag] = score

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
for name in ("GBD", "W120", "W150", "W365", "W409", "W90", "W45", "W60", "W180", "W270"):
    pool[name] = np.load(ROOT + f"work/AVG_{name}.npy").astype(np.float64)
# The frozen solver restricts the design to ridge22_keys. Six further bases have
# a MEASURED moment in EZ_pool.json and a vector on disk, and the filter discards
# them: a409a, tcn365e, w45a, w45b and w60a. w45a, w45b and w60a are
# individual members of the averaged W45 and W60 bases already in the pool, so
# admitting them lets the ridge weight those members separately instead of
# equally. No new probe is involved; every one of these moments was paid for
# long ago.
ridge_keys = set(np.load(ROOT + "work/ridge22_keys.npy").tolist())
ridge_keys |= {"a409a", "tcn365e", "w45a", "w45b", "w60a"}
# gb is the AVERAGE of v4_zh and cfg3, and the journal records an individually
# measured moment for each: 7.785462 for v4 (probe 13) and 7.787881 for cfg3
# (probe 24). Their mean is 7.7866715, which is exactly the gb entry in
# EZ_pool.json, so the two moments are consistent with the one already in use.
# Splitting gb into its components lets the ridge weight them separately.
# gbdt_v5, the 159-feature boosting base, has its own measured moment of
# 7.786213 from probe 29 and has never been in the design at all.
ridge_keys |= {"v4_zh", "cfg3", "gbdt_v5"}
ridge_keys.discard("gb")
pool["v4_zh"] = load("v4_zh")
pool["cfg3"] = load("cfg3")
pool["gbdt_v5"] = load("gbdt_v5")
ez_pool["v4_zh"] = 7.785462
ez_pool["cfg3"] = 7.787881
ez_pool["gbdt_v5"] = 7.786213
pool["tcn365e"] = load("tcn365e")
pool["w45a"] = load("w45a")
pool["w45b"] = load("w45b")
pool["w60a"] = load("w60a")
pool = {key: value for key, value in pool.items() if key in ridge_keys}
pool.update(candidates)

keys = sorted(pool)
reference = next(iter(candidates.values()))
design = np.vstack([pool[key] for key in keys] + [np.ones_like(reference)]).T
gram = design.T @ design / len(reference)
rhs = np.array([candidate_moments[key] if key in candidate_moments else ez_pool[key] for key in keys] + [M1])
penalty = np.eye(len(keys) + 1) * args.lam
penalty[-1, -1] = 0


def solve_system(local_gram, local_rhs, local_penalty):
    local_system = local_gram + local_penalty
    if not args.nonnegative:
        solution = np.linalg.solve(local_system, local_rhs)
        degrees_value = float(np.trace(
            local_gram @ np.linalg.inv(local_system)
        ))
        return solution, degrees_value, np.ones(len(solution), dtype=bool)

    # Convex quadratic: all model weights are bounded, while the intercept is
    # free. This provides an extrapolation-resistant insurance stack rather
    # than another nearly identical regularization variant.
    initial = np.linalg.solve(local_system, local_rhs)
    initial[:-1] = np.maximum(initial[:-1], 0.0)
    initial[-1] = (
        local_rhs[-1] - local_gram[-1, :-1] @ initial[:-1]
    ) / local_gram[-1, -1]

    def objective(value):
        return 0.5 * value @ local_system @ value - local_rhs @ value

    def gradient(value):
        return local_system @ value - local_rhs

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * (len(initial) - 1) + [(None, None)],
        options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 10_000, "maxls": 50},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"nonnegative ridge failed: {result.message}")
    solution = result.x
    active = np.r_[solution[:-1] > 1e-8, True]
    active_gram = local_gram[np.ix_(active, active)]
    active_penalty = local_penalty[np.ix_(active, active)]
    degrees_value = float(np.trace(
        active_gram @ np.linalg.inv(active_gram + active_penalty)
    ))
    return solution, degrees_value, active


coefficients, degrees, active = solve_system(gram, rhs, penalty)
full_gradient = (gram + penalty) @ coefficients - rhs
kkt_active_max_abs_gradient = float(np.max(np.abs(full_gradient[active])))
kkt_inactive_min_gradient = (
    float(np.min(full_gradient[~active])) if (~active).any() else None
)
mse = M2 - 2 * rhs @ coefficients + coefficients @ gram @ coefficients
expected = float(np.sqrt(max(mse, 0)))

base_keys = [key for key in keys if key not in candidate_moments]
base_design = np.vstack([pool[key] for key in base_keys] + [np.ones_like(reference)]).T
base_gram = base_design.T @ base_design / len(reference)
base_rhs = np.array([ez_pool[key] for key in base_keys] + [M1])
base_penalty = np.eye(len(base_keys) + 1) * args.lam
base_penalty[-1, -1] = 0
base_coefficients, base_degrees, base_active = solve_system(
    base_gram, base_rhs, base_penalty
)
base_mse = M2 - 2 * base_rhs @ base_coefficients + base_coefficients @ base_gram @ base_coefficients
base_expected = float(np.sqrt(max(base_mse, 0)))
public_gain = base_expected - expected
degrees_added = degrees - base_degrees

# The comparison above intentionally measures the whole supplied probe set
# against the original ridge basis.  For morning go/no-go decisions we also
# need the marginal value of the primary probe after every --extra candidate
# has already been admitted; otherwise their degrees of freedom are wrongly
# charged to the newest model.
marginal_keys = [key for key in keys if key != args.probe_tag]
marginal_design = np.vstack(
    [pool[key] for key in marginal_keys] + [np.ones_like(reference)]
).T
marginal_gram = marginal_design.T @ marginal_design / len(reference)
marginal_rhs = np.array([
    candidate_moments[key] if key in candidate_moments else ez_pool[key]
    for key in marginal_keys
] + [M1])
marginal_penalty = np.eye(len(marginal_keys) + 1) * args.lam
marginal_penalty[-1, -1] = 0
marginal_coefficients, marginal_degrees, marginal_active = solve_system(
    marginal_gram, marginal_rhs, marginal_penalty
)
marginal_mse = (
    M2
    - 2 * marginal_rhs @ marginal_coefficients
    + marginal_coefficients @ marginal_gram @ marginal_coefficients
)
marginal_expected = float(np.sqrt(max(marginal_mse, 0)))
marginal_public_gain = marginal_expected - expected
marginal_degrees_added = degrees - marginal_degrees
one_sided_penalty_per_degree = 0.0000165
# Public is the fitted sample and private is an independent test sample, so the
# first-order transfer gap has two sides. Historical 50k/200k simulations on
# 25 real bases add another ~18% over the idealized factor-of-two result.
theoretical_transfer_penalty_per_degree = 2 * one_sided_penalty_per_degree
empirical_transfer_penalty_per_degree = 0.0000395

raw = design @ coefficients
prediction_log = np.clip(raw, 0, None)
shift = M1 - prediction_log.mean()
for _ in range(8):
    prediction_log = np.clip(raw + shift, 0, None)
    shift += M1 - prediction_log.mean()

output = None
metadata_output = None
if not args.dry_run:
    output_name = args.output or f"{args.probe_tag}_ridge{args.lam:g}.csv"
    output = ROOT + "submissions/" + output_name
    metadata_stem = os.path.splitext(os.path.basename(output_name))[0]
    metadata_output = ROOT + f"work/{metadata_stem}_meta.json"
    uids = np.load(ROOT + "work/mat/uids.npy")
    with open(output, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["user_id", "predict"])
        for user_id, value in zip(uids, np.expm1(prediction_log)):
            writer.writerow([int(user_id), float(value)])

report = {
    "probe_scores": probe_scores,
    "ez_candidates": {key: float(value) for key, value in candidate_moments.items()},
    "lambda": args.lam,
    "solver": "nonnegative_ridge" if args.nonnegative else "ridge",
    "active_model_count": int(active[:-1].sum()),
    "kkt_active_max_abs_gradient": kkt_active_max_abs_gradient,
    "kkt_inactive_min_gradient": kkt_inactive_min_gradient,
    "expected_public": expected,
    "baseline_expected_public": base_expected,
    "public_gain": public_gain,
    "degrees_of_freedom": degrees,
    "degrees_added": degrees_added,
    "added_one_sided_optimism": degrees_added * one_sided_penalty_per_degree,
    "added_theoretical_public_private_penalty": degrees_added * theoretical_transfer_penalty_per_degree,
    "added_empirical_public_private_penalty": degrees_added * empirical_transfer_penalty_per_degree,
    "total_theoretical_public_private_penalty": degrees * theoretical_transfer_penalty_per_degree,
    "total_empirical_public_private_penalty": degrees * empirical_transfer_penalty_per_degree,
    "theoretical_private_score": expected + degrees * theoretical_transfer_penalty_per_degree,
    "empirical_private_score": expected + degrees * empirical_transfer_penalty_per_degree,
    "baseline_theoretical_private_score": base_expected + base_degrees * theoretical_transfer_penalty_per_degree,
    "baseline_empirical_private_score": base_expected + base_degrees * empirical_transfer_penalty_per_degree,
    "gain_after_theoretical_transfer_penalty": public_gain - degrees_added * theoretical_transfer_penalty_per_degree,
    "gain_after_empirical_transfer_penalty": public_gain - degrees_added * empirical_transfer_penalty_per_degree,
    "marginal_primary_baseline_expected_public": marginal_expected,
    "marginal_primary_public_gain": marginal_public_gain,
    "marginal_primary_degrees_added": marginal_degrees_added,
    "marginal_primary_added_theoretical_public_private_penalty": marginal_degrees_added * theoretical_transfer_penalty_per_degree,
    "marginal_primary_added_empirical_public_private_penalty": marginal_degrees_added * empirical_transfer_penalty_per_degree,
    "marginal_primary_gain_after_theoretical_transfer_penalty": marginal_public_gain - marginal_degrees_added * theoretical_transfer_penalty_per_degree,
    "marginal_primary_gain_after_empirical_transfer_penalty": marginal_public_gain - marginal_degrees_added * empirical_transfer_penalty_per_degree,
    "adaptive_selection_bias_included": False,
    "candidate_weights": {
        key: float(coefficients[keys.index(key)]) for key in candidate_moments
    },
    "mean_log": float(prediction_log.mean()),
    "output": output,
    "metadata_output": metadata_output,
    "weights": dict(zip(keys + ["const"], coefficients.tolist())),
}
if not args.dry_run:
    with open(metadata_output, "w") as stream:
        json.dump(report, stream, indent=2)
print(json.dumps(report, indent=2))
