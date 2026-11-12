"""Vehicle capacity is not specified anywhere in the source dataset, its Data in Brief
article, or its processing code (independently confirmed via literature review). We
derive a defensible baseline from observed demand and treat it as a swept sensitivity
parameter everywhere else, rather than a single assumed constant.

Baseline weight capacity is 500 kg, not the 400 kg the literature review's one
corroboration point suggested (a pharma CVRPTW study in the same city, independently
assuming 400kg/3m3). 400 kg was tried first and is not usable as-is: Day 4 contains a
single real order of 453.15 kg, which alone exceeds it -- making that day structurally
infeasible (no fleet size or routing can fix a per-vehicle capacity smaller than one
customer's own demand) regardless of anything else in the model. This was caught the
hard way, as OR-Tools silently returning NO_SOLUTION for every hyperparameter setting on
that one day. 500 kg clears the true dataset maximum (453.15 kg) with real headroom and
is still a realistic light-commercial-van payload, so the literature figure is treated as
a useful sanity check on order of magnitude rather than a value to import unchanged.
Volume's 3.0 m3 baseline already clears the observed maximum (1.95 m3) comfortably and
did not need adjusting.
"""

from dataclasses import dataclass

import numpy as np

from src.data_loading import DayInstance

BASELINE_WEIGHT_CAP_KG = 500.0
BASELINE_VOLUME_CAP_M3 = 3.0

# Joint multiplier applied to both dimensions together (a bigger vehicle has more of
# both), used in sensitivity analysis. 1.0 is the value used for the main experiment.
# Range chosen so even the smallest scenario (0.95x = 475kg) still clears the true
# dataset-wide maximum single-order weight (453.15kg, Day 4) -- see module docstring.
CAPACITY_SCENARIOS = (0.95, 1.0, 1.1, 1.25, 1.4, 1.6)


@dataclass(frozen=True)
class VehicleCapacity:
    weight_kg: float
    volume_m3: float
    scenario_multiplier: float


def capacity_for_scenario(multiplier: float) -> VehicleCapacity:
    return VehicleCapacity(
        weight_kg=BASELINE_WEIGHT_CAP_KG * multiplier,
        volume_m3=BASELINE_VOLUME_CAP_M3 * multiplier,
        scenario_multiplier=multiplier,
    )


def min_vehicles_needed(instance: DayInstance, capacity: VehicleCapacity) -> int:
    """Lower bound on fleet size from bin-packing relaxation (max of the two
    capacity dimensions' fractional lower bounds, ceiled)."""
    by_weight = instance.orders.WEIGHT.sum() / capacity.weight_kg
    by_volume = instance.orders.VOLUME.sum() / capacity.volume_m3
    return int(np.ceil(max(by_weight, by_volume)))


def assert_feasible_capacity(instance: DayInstance, capacity: VehicleCapacity) -> None:
    """Fail loudly and immediately if any single order exceeds vehicle capacity in
    either dimension, rather than letting it surface as a silent NO_SOLUTION deep in the
    solver (which then propagates as a confusing NaN even further downstream, in
    simulation or in an aggregated tuning score) -- exactly what happened with Day 4
    before this check existed."""
    max_w = instance.orders.WEIGHT.max()
    max_v = instance.orders.VOLUME.max()
    if max_w > capacity.weight_kg:
        raise ValueError(
            f"Day {instance.day}: an order weighs {max_w:.2f}kg, exceeding vehicle "
            f"capacity {capacity.weight_kg:.2f}kg ({capacity.scenario_multiplier:.3f}x "
            f"scenario) -- infeasible by construction, no fleet size can fix this."
        )
    if max_v > capacity.volume_m3:
        raise ValueError(
            f"Day {instance.day}: an order needs {max_v:.4f}m3, exceeding vehicle "
            f"capacity {capacity.volume_m3:.4f}m3 ({capacity.scenario_multiplier:.3f}x "
            f"scenario) -- infeasible by construction, no fleet size can fix this."
        )


def demand_summary(days: dict[int, DayInstance]) -> "pd.DataFrame":
    import pandas as pd

    rows = []
    baseline = capacity_for_scenario(1.0)
    for d, inst in sorted(days.items()):
        rows.append(dict(
            day=d,
            n_customers=inst.n_customers,
            weight_sum_kg=inst.orders.WEIGHT.sum(),
            volume_sum_m3=inst.orders.VOLUME.sum(),
            max_single_order_kg=inst.orders.WEIGHT.max(),
            max_single_order_m3=inst.orders.VOLUME.max(),
            min_vehicles_baseline=min_vehicles_needed(inst, baseline),
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.data_loading import load_all_days

    days = load_all_days()
    print(demand_summary(days).to_string(index=False))
    print()
    for mult in CAPACITY_SCENARIOS:
        cap = capacity_for_scenario(mult)
        for inst in days.values():
            assert_feasible_capacity(inst, cap)
        vehicles = [min_vehicles_needed(inst, cap) for inst in days.values()]
        print(f"{mult:.3f}x -> {cap.weight_kg:.0f}kg/{cap.volume_m3:.2f}m3, "
              f"min vehicles per day: {vehicles}  [all days feasibility-checked OK]")
