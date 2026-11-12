"""Main experiment: leave-one-day-out cross-validated comparison of a tuned global
robust margin (B1) against a deterministic no-margin baseline (B0), across all 9 days.
For each held-out day, both methods' hyperparameters are tuned on the other 8 days only,
then evaluated once on the held-out day with a larger solver time budget and Monte Carlo
sample size. This is the routing side of the paper's evidence: it shows how much a
tuned safety margin buys over no margin at all, motivating why the reconstruction-
ambiguity finding in Section 3 (the paper's main result) is not merely academic --
methods that plan against a travel-time estimate at all are sensitive to how that
estimate was built.

Checkpointed per fold to results/raw/lodo_checkpoint.json so a late failure doesn't
lose earlier folds; a per-(fold,method) try/except logs failures without aborting the
whole run, since a single edge case should not cost hours of otherwise-valid compute.
Re-run with --folds to restrict to specific held-out days (used for smoke-testing before
committing to the full 9-fold run).
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import N_DAYS, load_all_days
from src.evaluation import evaluate_method, prep_day
from src.tuning import tune_b1

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "raw"
CHECKPOINT_PATH = RESULTS_DIR / "lodo_checkpoint.json"

TUNE_TIME_LIMIT = 4.0
TUNE_DRAWS = 200
FINAL_TIME_LIMIT = 20.0
FINAL_DRAWS = 5000
CAPACITY_MULTIPLIER = 1.0
N_JOBS = 6
SEED_BASE = 20260815


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {"folds": {}}


def save_checkpoint(state: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(CHECKPOINT_PATH)


def run_fold(held_out_day: int, all_preps: dict, fold_seed: int) -> dict:
    train_preps = [all_preps[d] for d in range(1, N_DAYS + 1) if d != held_out_day]
    test_prep = all_preps[held_out_day]

    t0 = time.time()
    best_z_b1 = tune_b1(train_preps, fold_seed + 1, TUNE_TIME_LIMIT, TUNE_DRAWS, N_JOBS)
    print(f"    [fold {held_out_day}] B1 tuned: z={best_z_b1} ({time.time()-t0:.1f}s elapsed)", flush=True)
    tuning_time = time.time() - t0

    hyperparams = {
        "B0_deterministic": dict(z=0.0),
        "B1_global_robust": dict(z=best_z_b1),
    }

    t0 = time.time()
    fold_results = {}
    for method, hp in hyperparams.items():
        rng = np.random.default_rng(fold_seed + 1000)
        try:
            res = evaluate_method(
                method, test_prep["arrays"], test_prep["clusters"], test_prep["n_veh"],
                test_prep["cap"].weight_kg, test_prep["cap"].volume_m3,
                safety_z=hp["z"], n_draws=FINAL_DRAWS, rng=rng,
                rho=test_prep["rho"], time_limit_seconds=FINAL_TIME_LIMIT,
            )
            res["hyperparams"] = hp
            fold_results[method] = res
        except Exception as e:
            fold_results[method] = dict(method=method, error=str(e), traceback=traceback.format_exc(),
                                         hyperparams=hp)
    eval_time = time.time() - t0

    return dict(held_out_day=held_out_day, results=fold_results,
                tuning_seconds=tuning_time, eval_seconds=eval_time)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="*", default=list(range(1, N_DAYS + 1)))
    args = parser.parse_args()

    print(f"Loading data and preparing {N_DAYS} days (clustering, capacity)...")
    days = load_all_days()
    all_preps = {d: prep_day(inst, capacity_multiplier=CAPACITY_MULTIPLIER) for d, inst in days.items()}

    state = load_checkpoint()
    for held_out in args.folds:
        key = str(held_out)
        if key in state["folds"]:
            print(f"Fold {held_out}: already checkpointed, skipping.")
            continue
        print(f"Fold {held_out}: tuning + evaluating...", flush=True)
        t0 = time.time()
        fold_seed = SEED_BASE + held_out * 100_000
        result = run_fold(held_out, all_preps, fold_seed)
        elapsed = time.time() - t0
        print(f"Fold {held_out}: done in {elapsed:.1f}s "
              f"(tuning {result['tuning_seconds']:.1f}s, eval {result['eval_seconds']:.1f}s)", flush=True)
        for method, res in result["results"].items():
            if "error" in res:
                print(f"  !! {method} FAILED: {res['error']}", flush=True)
            else:
                print(f"  {method:24s} on_time={res['on_time_rate']:.4f} "
                      f"dist={res['distance_km']:7.1f}km veh={res['n_vehicles']}", flush=True)
        state["folds"][key] = result
        save_checkpoint(state)

    print("All requested folds complete. Checkpoint saved to", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
