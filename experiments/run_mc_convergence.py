"""Technical validation: confirm FINAL_DRAWS=5000 (used everywhere in the main
experiment) is enough for the Monte Carlo on-time-rate estimate to have stabilized,
rather than asserting it without evidence. Fixes one solved route per day and
re-estimates on_time_rate at increasing draw counts with multiple independent seeds per
count, reporting how the across-seed standard deviation of the estimate shrinks.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_all_days
from src.evaluation import build_arrays, prep_day
from src.reconstruction import calibrate_beta_pert, reconstruct
from src.routing_solver import solve_cvrptw
from src.simulation import simulate_solution

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "raw" / "mc_convergence.json"
DRAW_COUNTS = (50, 100, 250, 500, 1000, 2500, 5000, 10000)
N_SEEDS_PER_COUNT = 12
TEST_DAYS = (1, 5, 9)  # small, medium, large instance


def main():
    days = load_all_days()
    output = {}
    for day in TEST_DAYS:
        prep = prep_day(days[day])
        recon = reconstruct(prep["arrays"].o_raw, prep["arrays"].m_raw, prep["arrays"].p_raw, "order_statistics")
        params = calibrate_beta_pert(recon)
        eff_time = recon.mostlikely + 1.5 * params.sd  # fixed, moderate margin -- not the point of this check
        sol = solve_cvrptw(prep["arrays"].distance, eff_time, prep["arrays"].service, prep["arrays"].deadlines,
                            prep["arrays"].weights, prep["arrays"].volumes, prep["cap"].weight_kg,
                            prep["cap"].volume_m3, prep["n_veh"], time_limit_seconds=15)
        print(f"Day {day}: solved, {sol.n_vehicles_used} vehicles, {sol.total_distance_km:.1f}km")

        by_count = {}
        for n_draws in DRAW_COUNTS:
            estimates = []
            for s in range(N_SEEDS_PER_COUNT):
                rng = np.random.default_rng(day * 100000 + n_draws * 100 + s)
                metrics = simulate_solution(sol.routes, params, prep["clusters"].labels,
                                             prep["arrays"].service, prep["arrays"].deadlines,
                                             n_draws, rng, prep["rho"])
                estimates.append(metrics["on_time_rate"])
            estimates = np.array(estimates)
            by_count[n_draws] = dict(mean=float(estimates.mean()), std_across_seeds=float(estimates.std()),
                                      min=float(estimates.min()), max=float(estimates.max()))
            print(f"  n_draws={n_draws:6d}  mean={estimates.mean():.5f}  "
                  f"std_across_seeds={estimates.std():.6f}  range=[{estimates.min():.4f},{estimates.max():.4f}]")
        output[str(day)] = by_count

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print("Saved to", RESULTS_PATH)


if __name__ == "__main__":
    main()
