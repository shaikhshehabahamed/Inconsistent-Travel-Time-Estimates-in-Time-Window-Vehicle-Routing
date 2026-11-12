"""Sensitivity analysis: does B1's on-time rate hold up under the modelling choices this
study had to assume rather than observe -- vehicle capacity (not given anywhere in the
source data) and the target service level (a design choice, 0.995 in the main
experiment)?

Two genuinely different questions live under "sensitivity," and each sub-analysis uses
the design that actually matches its question.

Capacity is a property of the WORLD this study had to assume a value for. The relevant
question is robustness to that assumption being wrong: if B1 is tuned once (at its real
LODO-CV operating point) and then deployed into a world where the true capacity differs,
does it hold up? That means holding B1's hyperparameter FIXED at the main experiment's
modal tuned point (z=2.0, chosen in 4 of 9 folds -- see results/tables/per_fold_results.csv
for the per-fold z values this is read from) and only varying the environment.

An earlier version of this script instead re-tuned in-sample at every single capacity
scenario with an independent small grid search. That was a real design mistake, not just
a noise problem: it answered a different question ("if I knew capacity was X and
retuned for it, would B1 still perform well?") than the one motivating the check ("is
the deployed solution robust to capacity being different than assumed?"), and a coarse
grid searched independently at each scenario produced a visually erratic curve because
each scenario's argmax landed on a different, weakly-distinguished grid corner. Fixed by
holding hyperparameters fixed instead of re-tuning per scenario.

Target service level is different: it is a REQUIREMENT the decision-maker sets, not a
fact about the world, and it only has meaning through the tuning objective (src/tuning.py
score()) that hyperparameters are chosen to satisfy -- there is nothing to hold "fixed"
across different targets, re-optimizing for each one is the correct design. This uses the
main experiment's actual (fine) grid and a two-stage tune-then-evaluate precision split
(matching run_lodo_cv.py's TUNE_*/FINAL_* pattern) instead of searching the whole grid at
low precision throughout.

Deliberately still lighter-weight than the main LODO-CV experiment in one respect: all 9
days are used in-sample (not held out per fold) for both sub-analyses. That is
appropriate for a sensitivity check (does the qualitative finding hold as an assumption
varies?) and is NOT a substitute for the main experiment's out-of-sample result, which is
what the paper's headline numbers come from.
"""

import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capacity import CAPACITY_SCENARIOS, assert_feasible_capacity, capacity_for_scenario, min_vehicles_needed
from src.clustering import select_n_clusters
from src.data_loading import N_DAYS, load_all_days
from src.evaluation import build_arrays, evaluate_method
from src.reconstruction import reconstruct
from src.tuning import Z_GRID, score

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "raw" / "sensitivity.json"

# Fixed operating point for the capacity check: the main LODO-CV experiment's modal
# tuned z for B1 (see module docstring). Read off results/tables/per_fold_results.csv
# after the main LODO-CV run, not guessed -- re-derive if that file changes.
FIXED_B1_Z = 2.0

# Full precision throughout capacity (no grid search there, so this is affordable) and as
# the final-evaluation stage for service-target's tune-then-evaluate.
N_DRAWS = 5000
TIME_LIMIT = 20.0
# Tuning-stage precision for service-target's grid search only (mirrors run_lodo_cv.py's
# TUNE_DRAWS/TUNE_TIME_LIMIT -- only the relative ranking across the grid matters here).
TUNE_N_DRAWS = 200
TUNE_TIME_LIMIT = 4.0
N_JOBS = 6


def _prep_with_capacity(inst, capacity_multiplier, rho=0.3, vehicle_slack=2):
    cap = capacity_for_scenario(capacity_multiplier)
    assert_feasible_capacity(inst, cap)
    n_veh = min_vehicles_needed(inst, cap) + vehicle_slack
    arrays = build_arrays(inst)
    recon_os = reconstruct(inst.time_optimistic, inst.time_mostlikely, inst.time_pessimistic, "order_statistics")
    clusters = select_n_clusters(recon_os.mostlikely)
    return dict(day=inst.day, arrays=arrays, clusters=clusters, cap=cap, n_veh=n_veh, rho=rho)


def _eval_fixed(preps, method, z, seed_base):
    """Evaluate one method at one fixed z across every day, full precision. No tuning --
    the hyperparameter is given, not searched."""
    def job(d):
        rng = np.random.default_rng(seed_base + d)
        p = preps[d]
        return evaluate_method(method, p["arrays"], p["clusters"], p["n_veh"], p["cap"].weight_kg,
                                p["cap"].volume_m3, safety_z=z, n_draws=N_DRAWS,
                                rng=rng, rho=p["rho"], time_limit_seconds=TIME_LIMIT)

    results = Parallel(n_jobs=N_JOBS)(delayed(job)(d) for d in range(len(preps)))
    return dict(z=z, on_time_rate=float(np.mean([r["on_time_rate"] for r in results])),
                distance_km=float(np.mean([r["distance_km"] for r in results])))


def capacity_sensitivity(days):
    """Fixed hyperparameter, varying capacity -- robustness to a misspecified
    environment (see module docstring for why this replaced in-scenario re-tuning)."""
    results = []
    for mult in CAPACITY_SCENARIOS:
        preps = [_prep_with_capacity(days[d], mult) for d in range(1, N_DAYS + 1)]
        b1 = _eval_fixed(preps, "B1_global_robust", FIXED_B1_Z, seed_base=int(mult * 1e6))
        print(f"capacity {mult:.3f}x: B1 on_time={b1['on_time_rate']:.4f} dist={b1['distance_km']:.1f}")
        results.append(dict(capacity_multiplier=mult, B1=b1))
    return results


def _tune_then_evaluate(preps, method, z_values, target, seed_base):
    """Coarse-precision grid search for the best worst-case z at this target, then one
    full-precision evaluation of the winner -- mirrors run_lodo_cv.py's TUNE_*/FINAL_*
    two-stage design instead of paying full precision at every grid point."""
    tasks = [(z, d) for z in z_values for d in range(len(preps))]

    def tune_job(z, d):
        rng = np.random.default_rng(seed_base + int(z * 100) + d)
        r = evaluate_method(method, preps[d]["arrays"], preps[d]["clusters"], preps[d]["n_veh"],
                             preps[d]["cap"].weight_kg, preps[d]["cap"].volume_m3,
                             safety_z=z, n_draws=TUNE_N_DRAWS, rng=rng,
                             rho=preps[d]["rho"], time_limit_seconds=TUNE_TIME_LIMIT)
        return z, score(r["on_time_rate"], r["distance_km"], target)

    outcomes = Parallel(n_jobs=N_JOBS)(delayed(tune_job)(*t) for t in tasks)
    grouped = {}
    for key, s in outcomes:
        grouped.setdefault(key, []).append(s)
    # worst-case (minimum) across days, not mean -- same minimax principle as the main
    # experiment's tuning objective (src/tuning.py), applied for the same reason.
    worst_case = {k: float(np.min(v)) for k, v in grouped.items()}
    best_z = max(worst_case, key=worst_case.get)
    return _eval_fixed(preps, method, best_z, seed_base=seed_base + 900_000)


def service_target_sensitivity(days, targets=(0.98, 0.99, 0.995, 0.999)):
    """Re-tunes per target (the target only has meaning through the tuning objective, so
    unlike capacity there is no sensible 'fixed hyperparameter' version of this check --
    see module docstring) using the main experiment's actual fine grid, not a coarse one."""
    preps = [_prep_with_capacity(days[d], 1.0) for d in range(1, N_DAYS + 1)]
    results = []
    for target in targets:
        b1 = _tune_then_evaluate(preps, "B1_global_robust", Z_GRID, target, seed_base=int(target * 1e6))
        print(f"target {target:.3f}: B1 on_time={b1['on_time_rate']:.4f} dist={b1['distance_km']:.1f} "
              f"(z={b1['z']})")
        results.append(dict(target=target, B1=b1))
    return results


def main():
    days = load_all_days()
    print(f"=== Capacity sensitivity (fixed B1 z={FIXED_B1_Z}) ===")
    cap_results = capacity_sensitivity(days)
    print("\n=== Service-level target sensitivity (re-tuned per target, full grid) ===")
    target_results = service_target_sensitivity(days)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(dict(capacity=cap_results, service_target=target_results), f, indent=2)
    print("\nSaved to", RESULTS_PATH)


if __name__ == "__main__":
    main()
