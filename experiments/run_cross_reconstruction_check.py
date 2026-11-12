"""Downstream-impact check for the paper's central data-quality finding: given that the
three reconstruction rules disagree with each other (src/reconstruction.py's
compare_methods(), Figure 1), how much does that disagreement actually move the
assessed reliability of a routing method built on top of it?

B0 and B1 are each built and headline-evaluated against a single reconstruction
(order_statistics) -- the right number for what each method is on its own terms. This
script re-solves both methods at their ALREADY-TUNED LODO-CV hyperparameters (read from
the completed checkpoint, not re-tuned -- so this is not a new tuning run, just a
re-evaluation of the same operating points the headline results already use) and scores
the resulting route under all 3 official reconstructions via
evaluate_method(..., cross_evaluate=True). It reports the per-reconstruction breakdown
and reruns the same paired statistical comparison (src/stats_tests.compare_to_baselines,
B1 against B0) on the cross-reconstruction-averaged metric, to check whether the
routing-side pattern (B1 significantly beats B0) is stable across which reconstruction
is assumed correct, or is itself sensitive to that choice -- direct evidence for how much
the reconstruction-ambiguity problem matters in practice, beyond the raw disagreement
statistics in Figure 1.

This does NOT replace or alter the main LODO-CV results in results/tables/ -- it is an
additional check of whether they are robust to which reconstruction is assumed, in the
same spirit as run_solver_repeatability.py (solver-seed noise) and run_mc_convergence.py
(Monte Carlo sampling noise).

Checkpointed per (day, method) to results/raw/cross_reconstruction_checkpoint.json, same
reason and same pattern as run_lodo_cv.py: solves take longer than any single interactive
session comfortably allows, so this script is safe to invoke repeatedly -- each call does
as much of the remaining work as it can in --max-seconds (default 480s) and picks up
where it left off next time. Call with --finalize once
results/raw/cross_reconstruction_checkpoint.json shows all tasks complete to write the
CSVs and run the statistical comparison.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import N_DAYS, load_all_days
from src.evaluation import evaluate_method, prep_day
from src.stats_tests import compare_to_baselines

ROOT = Path(__file__).resolve().parent.parent
LODO_CHECKPOINT_PATH = ROOT / "results" / "raw" / "lodo_checkpoint.json"
CHECKPOINT_PATH = ROOT / "results" / "raw" / "cross_reconstruction_checkpoint.json"
RESULTS_PATH = ROOT / "results" / "raw" / "cross_reconstruction_check.json"
TABLES_DIR = ROOT / "results" / "tables"

# Match run_lodo_cv.py's FINAL_* precision exactly -- this check is meant to be directly
# comparable to the paper's headline numbers, not a cheaper approximation of them.
FINAL_TIME_LIMIT = 20.0
FINAL_DRAWS = 5000
N_JOBS = 6
SEED_BASE = 20260815  # same base run_lodo_cv.py uses, offset below so seeds never collide

METHODS = ("B0_deterministic", "B1_global_robust")


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {}


def save_checkpoint(state: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(CHECKPOINT_PATH)


def _all_tasks() -> list:
    with open(LODO_CHECKPOINT_PATH) as f:
        lodo_state = json.load(f)
    tasks = []
    for d in range(1, N_DAYS + 1):
        fold = lodo_state["folds"][str(d)]
        for method in METHODS:
            res = fold["results"].get(method, {})
            if "error" in res:
                continue
            hp = res["hyperparams"]
            tasks.append((d, method, hp["z"]))
    return tasks


def _job(d, method, z, preps):
    seed = SEED_BASE + d * 1_000_000 + METHODS.index(method) * 1000 + 7
    rng = np.random.default_rng(seed)
    r = evaluate_method(method, preps[d]["arrays"], preps[d]["clusters"], preps[d]["n_veh"],
                         preps[d]["cap"].weight_kg, preps[d]["cap"].volume_m3,
                         safety_z=z, n_draws=FINAL_DRAWS, rng=rng,
                         rho=preps[d]["rho"], time_limit_seconds=FINAL_TIME_LIMIT, cross_evaluate=True)
    return dict(day=d, method=method, z=z, official_on_time_rate=r["on_time_rate"],
                cross_recon_mean_on_time_rate=r["cross_recon"]["mean_on_time_rate"],
                per_reconstruction=r["cross_recon"]["per_reconstruction"],
                distance_km=r["distance_km"], n_vehicles=r["n_vehicles"])


def run_pending(max_seconds: float = 480.0) -> bool:
    """Do as much of the remaining (day, method) work as fits in max_seconds, checkpointing
    after every batch. Returns True iff all tasks are complete after this call."""
    all_tasks = _all_tasks()
    state = load_checkpoint()
    remaining = [t for t in all_tasks if f"{t[0]}_{t[1]}" not in state]
    print(f"{len(all_tasks) - len(remaining)}/{len(all_tasks)} tasks already checkpointed; "
          f"{len(remaining)} remaining.", flush=True)
    if not remaining:
        return True

    days = load_all_days()
    preps = {d: prep_day(days[d]) for d in range(1, N_DAYS + 1)}

    t_start = time.time()
    i = 0
    while i < len(remaining):
        if time.time() - t_start > max_seconds:
            print(f"Time budget ({max_seconds:.0f}s) reached; stopping early. "
                  f"Re-run this script to continue.", flush=True)
            break
        batch = remaining[i:i + N_JOBS]
        batch_results = Parallel(n_jobs=N_JOBS)(delayed(_job)(*t, preps) for t in batch)
        for r in batch_results:
            state[f"{r['day']}_{r['method']}"] = r
        save_checkpoint(state)
        i += len(batch)
        elapsed = time.time() - t_start
        print(f"  checkpointed {i}/{len(remaining)} pending tasks this run "
              f"({len(state)}/{len(all_tasks)} total) -- {elapsed:.0f}s elapsed", flush=True)

    return len(state) == len(all_tasks)


def finalize() -> None:
    """Once run_pending() returns True: build the flat table, the apples-to-apples
    summary, and the statistical comparison, exactly as a single-shot run would have."""
    state = load_checkpoint()
    all_tasks = _all_tasks()
    assert len(state) == len(all_tasks), (
        f"checkpoint incomplete ({len(state)}/{len(all_tasks)}) -- run run_pending() first")

    results = list(state.values())
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved raw results to", RESULTS_PATH)

    rows = []
    for r in results:
        row = dict(day=r["day"], method=r["method"], z=r["z"],
                    official_on_time_rate=r["official_on_time_rate"],
                    cross_recon_mean_on_time_rate=r["cross_recon_mean_on_time_rate"],
                    distance_km=r["distance_km"], n_vehicles=r["n_vehicles"])
        for e, v in r["per_reconstruction"].items():
            row[f"on_time_under_{e}"] = v
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["day", "method"])
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "cross_reconstruction_eval.csv", index=False)
    print(f"Wrote cross_reconstruction_eval.csv ({len(df)} rows)")

    summary = df.groupby("method")["cross_recon_mean_on_time_rate"].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["method", "cross_recon_on_time_mean", "cross_recon_on_time_std", "n_folds"]
    summary.to_csv(TABLES_DIR / "cross_reconstruction_summary.csv", index=False)
    print("\nApples-to-apples summary (mean on_time_rate under the all-3-reconstruction "
          "average, n=9 folds):")
    print(summary.to_string(index=False))

    if "B1_global_robust" in df.method.unique() and "B0_deterministic" in df.method.unique():
        b1_by_day = {int(r.day): r.cross_recon_mean_on_time_rate for _, r in
                     df[df.method == "B1_global_robust"].iterrows()}
        b0_by_day = {int(r.day): r.cross_recon_mean_on_time_rate for _, r in
                     df[df.method == "B0_deterministic"].iterrows()}
        baselines = {"B0_deterministic": b0_by_day} if len(b0_by_day) == N_DAYS else {}

        if baselines:
            comps = compare_to_baselines(b1_by_day, baselines, metric="cross_recon_mean_on_time_rate", seed=3)
            stats_df = pd.DataFrame([c.__dict__ for c in comps])
            stats_df.to_csv(TABLES_DIR / "cross_reconstruction_statistical_comparisons.csv", index=False)
            print("\nStatistical comparison (B1 vs B0) under the apples-to-apples "
                  "(cross-reconstruction-mean) metric:")
            print(stats_df.to_string(index=False))

            orig_path = TABLES_DIR / "statistical_comparisons.csv"
            if orig_path.exists():
                orig = pd.read_csv(orig_path)
                orig = orig[orig.metric == "on_time_rate"][["baseline", "mean_diff", "p_value_corrected", "reject_null"]]
                orig = orig.rename(columns={"mean_diff": "orig_mean_diff", "p_value_corrected": "orig_p_holm",
                                             "reject_null": "orig_reject"})
                cmp2 = stats_df[["baseline", "mean_diff", "p_value_corrected", "reject_null"]].rename(
                    columns={"mean_diff": "xrecon_mean_diff", "p_value_corrected": "xrecon_p_holm",
                             "reject_null": "xrecon_reject"})
                merged = orig.merge(cmp2, on="baseline")
                merged.to_csv(TABLES_DIR / "cross_reconstruction_vs_original_comparison.csv", index=False)
                print("\nOriginal (each method evaluated under its own single reconstruction) vs. "
                      "apples-to-apples (both methods evaluated under the 3-reconstruction "
                      "mean), side by side:")
                print(merged.to_string(index=False))

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seconds", type=float, default=480.0)
    parser.add_argument("--finalize", action="store_true",
                         help="skip solving; just rebuild tables from a complete checkpoint")
    args = parser.parse_args()

    if args.finalize:
        finalize()
        return

    complete = run_pending(max_seconds=args.max_seconds)
    if complete:
        print("\nAll tasks complete.")
        finalize()
    else:
        print("\nNot yet complete -- re-run this script (same command) to continue.")


if __name__ == "__main__":
    main()
