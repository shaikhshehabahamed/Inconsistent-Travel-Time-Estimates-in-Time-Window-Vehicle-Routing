"""OR-Tools' RoutingSearchParameters exposes no random-seed field (verified directly
against its protobuf descriptor -- see routing_solver.py), so the guided-local-search
trajectory cannot be seeded from this wrapper. That means a single solve at a fixed time
budget can land on a marginally different local optimum between runs, which is a real
source of noise sitting underneath whatever the margin design itself contributes to the
fold-to-fold spread seen in the main experiment.

This script quantifies that noise directly: re-solves each method REPEATS times at its
already-tuned hyperparameters (read from the completed LODO-CV checkpoint, not
re-tuned), on every held-out day, and reports the within-day standard deviation of
on-time rate and distance. If that spread is small relative to the between-method
differences reported in the main results, the headline comparisons are not being driven
by solver noise; if it is comparable, that is an honest limitation to report as such.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import N_DAYS, load_all_days
from src.evaluation import evaluate_method, prep_day

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "results" / "raw" / "lodo_checkpoint.json"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "raw" / "solver_repeatability.json"
REPEATS = 5
N_DRAWS = 3000
TIME_LIMIT = 20.0
N_JOBS = 6


def main():
    with open(CHECKPOINT_PATH) as f:
        state = json.load(f)

    days = load_all_days()
    preps = {d: prep_day(days[d]) for d in range(1, N_DAYS + 1)}

    tasks = []
    for d in range(1, N_DAYS + 1):
        fold = state["folds"][str(d)]
        for method, res in fold["results"].items():
            if "error" in res:
                continue
            hp = res["hyperparams"]
            for rep in range(REPEATS):
                tasks.append((d, method, hp["z"], rep))

    def job(d, method, z, rep):
        rng = np.random.default_rng(d * 100_000 + hash(method) % 10_000 + rep)
        r = evaluate_method(method, preps[d]["arrays"], preps[d]["clusters"], preps[d]["n_veh"],
                             preps[d]["cap"].weight_kg, preps[d]["cap"].volume_m3,
                             safety_z=z, n_draws=N_DRAWS, rng=rng,
                             rho=preps[d]["rho"], time_limit_seconds=TIME_LIMIT)
        return dict(day=d, method=method, rep=rep, on_time_rate=r["on_time_rate"], distance_km=r["distance_km"])

    print(f"Running {len(tasks)} repeat-solves ({REPEATS}x per day-method)...")
    results = Parallel(n_jobs=N_JOBS)(delayed(job)(*t) for t in tasks)
    df = pd.DataFrame(results)

    summary = df.groupby(["day", "method"]).agg(
        on_time_mean=("on_time_rate", "mean"), on_time_std=("on_time_rate", "std"),
        dist_mean=("distance_km", "mean"), dist_std=("distance_km", "std"),
    ).reset_index()

    print(summary.to_string(index=False))
    print()
    print("Overall solver-noise summary (std across repeats, averaged over day/method cells):")
    print(f"  on_time_rate std: mean={summary.on_time_std.mean():.5f}  max={summary.on_time_std.max():.5f}")
    print(f"  distance_km std:  mean={summary.dist_std.mean():.3f}  max={summary.dist_std.max():.3f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_json(RESULTS_PATH, orient="records", indent=2)
    print("Saved to", RESULTS_PATH)


if __name__ == "__main__":
    main()
