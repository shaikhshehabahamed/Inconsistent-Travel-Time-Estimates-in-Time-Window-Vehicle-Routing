"""Turn the LODO-CV checkpoint into the tables the paper needs: a main results table
(mean +/- 95% CI per method per metric, n=9 folds) and a paired significance test of B1
(global margin) against B0 (deterministic, no margin). Reads
results/raw/lodo_checkpoint.json; writes CSVs to results/tables/ and a combined JSON to
results/raw/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stats_tests import compare_to_baselines

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT / "results" / "raw" / "lodo_checkpoint.json"
TABLES_DIR = ROOT / "results" / "tables"

METHODS = ("B0_deterministic", "B1_global_robust")
METRICS = ("on_time_rate", "distance_km", "n_vehicles")


def load_folds() -> dict:
    with open(CHECKPOINT_PATH) as f:
        state = json.load(f)
    return state["folds"]


def bootstrap_ci_mean(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main():
    folds = load_folds()
    days = sorted(int(d) for d in folds.keys())
    n = len(days)
    print(f"Loaded {n} folds: days {days}")

    # long-form per-fold-per-method table
    rows = []
    for d in days:
        fold = folds[str(d)]
        for method in METHODS:
            r = fold["results"].get(method, {})
            if "error" in r:
                print(f"  WARNING: day {d} method {method} FAILED: {r['error']}")
                continue
            rows.append(dict(day=d, method=method, on_time_rate=r["on_time_rate"],
                              distance_km=r["distance_km"], n_vehicles=r["n_vehicles"],
                              route_on_time_rate=r.get("route_on_time_rate"),
                              mean_lateness_min=r.get("mean_lateness_min"),
                              z=r["hyperparams"]["z"]))
    long_df = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(TABLES_DIR / "per_fold_results.csv", index=False)
    print(f"Wrote per_fold_results.csv ({len(long_df)} rows)")

    # main summary table: mean +/- 95% bootstrap CI per method per metric
    summary_rows = []
    for method in METHODS:
        sub = long_df[long_df.method == method]
        if len(sub) == 0:
            continue
        row = dict(method=method, n_folds=len(sub))
        for metric in METRICS:
            vals = sub[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_ci_mean(vals)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_ci_lo"] = lo
            row[f"{metric}_ci_hi"] = hi
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES_DIR / "main_results_summary.csv", index=False)
    print("\nMain results summary:")
    print(summary_df.to_string(index=False))

    # paired stats: B1 (global margin) vs B0 (deterministic), on both on_time_rate and distance_km
    if "B1_global_robust" in long_df.method.unique() and "B0_deterministic" in long_df.method.unique():
        b1_ontime_by_day = {int(r.day): r.on_time_rate for _, r in
                             long_df[long_df.method == "B1_global_robust"].iterrows()}
        b1_dist_by_day = {int(r.day): r.distance_km for _, r in
                           long_df[long_df.method == "B1_global_robust"].iterrows()}
        b0_sub = long_df[long_df.method == "B0_deterministic"]

        stats_rows = []
        if len(b0_sub) == len(days):
            b0_ontime_by_day = {int(r.day): r.on_time_rate for _, r in b0_sub.iterrows()}
            b0_dist_by_day = {int(r.day): r.distance_km for _, r in b0_sub.iterrows()}
            for comp in compare_to_baselines(b1_ontime_by_day, {"B0_deterministic": b0_ontime_by_day},
                                              metric="on_time_rate", seed=1):
                stats_rows.append(comp.__dict__)
            for comp in compare_to_baselines(b1_dist_by_day, {"B0_deterministic": b0_dist_by_day},
                                              metric="distance_km", seed=2):
                stats_rows.append(comp.__dict__)
        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_csv(TABLES_DIR / "statistical_comparisons.csv", index=False)
        print("\nStatistical comparison (B1 vs B0):")
        print(stats_df.to_string(index=False))

    build_robustness_table()


def build_robustness_table():
    """Summarize the two robustness checks (MC draw-count convergence, OR-Tools
    solver-seed noise) into one small table, so the 'multiple random seed evaluation'
    evidence has a citable table alongside its figure, not just raw JSON."""
    rep_path = ROOT / "results" / "raw" / "solver_repeatability.json"
    mc_path = ROOT / "results" / "raw" / "mc_convergence.json"
    if not (rep_path.exists() and mc_path.exists()):
        print("skip robustness table: run run_solver_repeatability.py and run_mc_convergence.py first")
        return

    with open(rep_path) as f:
        rep = pd.DataFrame(json.load(f))
    with open(mc_path) as f:
        mc = json.load(f)

    rows = []
    for method in METHODS:
        sub = rep[rep.method == method]
        if len(sub) == 0:
            continue
        rows.append(dict(
            method=method,
            solver_on_time_std_mean=float(sub.on_time_std.mean()),
            solver_on_time_std_max=float(sub.on_time_std.max()),
            solver_dist_std_km_mean=float(sub.dist_std.mean()),
            solver_dist_std_km_max=float(sub.dist_std.max()),
            n_day_reps=int(len(sub)),
        ))
    rep_summary = pd.DataFrame(rows)
    rep_summary.to_csv(TABLES_DIR / "robustness_solver_noise.csv", index=False)

    mc_rows = []
    for day, by_count in mc.items():
        for n_draws, stats in by_count.items():
            mc_rows.append(dict(day=int(day), n_draws=int(n_draws), **stats))
    mc_df = pd.DataFrame(mc_rows).sort_values(["day", "n_draws"])
    mc_df.to_csv(TABLES_DIR / "robustness_mc_convergence.csv", index=False)

    print("\nSolver-repeatability summary (std across 5 repeat-solves, by method):")
    print(rep_summary.to_string(index=False))
    print(f"\nWrote robustness_solver_noise.csv ({len(rep_summary)} rows) and "
          f"robustness_mc_convergence.csv ({len(mc_df)} rows)")


if __name__ == "__main__":
    main()
