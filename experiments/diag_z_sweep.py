"""One-off diagnostic (not part of the paper pipeline): characterize on_time_rate vs
distance across a wide z range for B1, to find the regime where the service/cost
tradeoff is genuinely non-trivial (not already saturated near-ceiling even at low z) --
needed after fold-1 of the real LODO-CV showed every method converging to the same
(z=0.5, w=0) corner. Written as a real module (not `python -c`) deliberately: joblib's
loky backend uses the 'spawn' start method on Windows, which re-imports __main__ in each
worker process -- for `python -c` inline code there is no file to re-import, which
silently produced NaN results in an earlier attempt at this same check.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_all_days
from src.evaluation import evaluate_method, prep_day

Z_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def job(method, z, d, preps):
    seed = abs(hash(method)) % 1000 + int(z * 100) + d
    rng = np.random.default_rng(seed)
    r = evaluate_method(method, preps[d]["arrays"], preps[d]["clusters"], preps[d]["n_veh"],
                         preps[d]["cap"].weight_kg, preps[d]["cap"].volume_m3,
                         safety_z=z, n_draws=1500, rng=rng, rho=preps[d]["rho"],
                         time_limit_seconds=8)
    return method, z, d, r["on_time_rate"], r["distance_km"], r["solve_status"]


def main():
    days = load_all_days()
    preps = {d: prep_day(inst) for d, inst in days.items()}

    tasks = [("B1_global_robust", z, d) for z in Z_VALUES for d in range(1, 10)]
    print(f"{len(tasks)} tasks...")
    results = Parallel(n_jobs=7)(delayed(job)(*t, preps) for t in tasks)

    by_z = defaultdict(list)
    bad = []
    for method, z, d, ot, dist, status in results:
        if status != "OK" or np.isnan(ot):
            bad.append((d, z, status))
        by_z[z].append((ot, dist))

    if bad:
        print("Non-OK results:", bad)

    for z in Z_VALUES:
        vals = by_z[z]
        ot_vals = [v[0] for v in vals]
        dist_vals = [v[1] for v in vals]
        print(f"z={z:.2f}: mean on_time={np.mean(ot_vals):.4f}  "
              f"min={np.min(ot_vals):.4f}  mean_dist={np.mean(dist_vals):.1f}")


if __name__ == "__main__":
    main()
