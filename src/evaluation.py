"""Unified solve-and-simulate entry point for every method configuration in the study,
so tuning and final evaluation always exercise the exact same code path -- no risk of
the two silently diverging.

Method configurations:
  B0_deterministic : no margin at all (z=0). The naive floor.
  B1_global_robust : order-statistics reconstruction, tuned uniform margin z. The
                      standard existing-literature approach (a single safety margin
                      applied uniformly, ignoring correlation) and the paper's routing
                      result.

Both methods depend on the same single-method reconstruction (order_statistics) just to
get a valid Beta-PERT calibration from raw travel-time labels that violate
optimistic<=most_likely<=pessimistic in up to 23% of pairs on the worst day -- see
src/reconstruction.py. That reconstruction-choice question is evaluated on its own
terms, not folded into either method: cross_evaluate=True below re-scores a method's
already-solved route under all three official reconstructions
(order_statistics/trust_mode/trust_envelope), which is what
experiments/run_cross_reconstruction_check.py uses to measure how much reconstruction
ambiguity actually affects B0/B1's assessed reliability -- the paper's data-quality
result, not a routing algorithm.
"""

from dataclasses import dataclass

import numpy as np

from src.capacity import assert_feasible_capacity, capacity_for_scenario, min_vehicles_needed
from src.clustering import ClusterAssignment, select_n_clusters
from src.data_loading import DayInstance
from src.margins import effective_time_matrix
from src.reconstruction import ALL_METHODS, calibrate_beta_pert, reconstruct
from src.routing_solver import solve_cvrptw
from src.simulation import DEFAULT_RHO, simulate_solution

METHODS = ("B0_deterministic", "B1_global_robust")


@dataclass(frozen=True)
class ProblemArrays:
    distance: np.ndarray
    o_raw: np.ndarray
    m_raw: np.ndarray
    p_raw: np.ndarray
    weights: np.ndarray
    volumes: np.ndarray
    service: np.ndarray
    deadlines: np.ndarray
    n_customers: int


def build_arrays(inst: DayInstance) -> ProblemArrays:
    return ProblemArrays(
        distance=inst.distance, o_raw=inst.time_optimistic, m_raw=inst.time_mostlikely,
        p_raw=inst.time_pessimistic,
        weights=np.concatenate([[0.0], inst.orders.WEIGHT.values]),
        volumes=np.concatenate([[0.0], inst.orders.VOLUME.values]),
        service=np.concatenate([[0], inst.orders.SERVICE_TIME.values]).astype(float),
        deadlines=np.concatenate([[10 ** 6], inst.orders.LAT.values]).astype(float),
        n_customers=inst.n_customers,
    )


def prep_day(inst: DayInstance, capacity_multiplier: float = 1.0, rho: float = DEFAULT_RHO,
             vehicle_slack: int = 2) -> dict:
    """Bundle everything evaluate_method needs for one day: capacity, a generous fleet
    upper bound, the raw arrays, and a data-driven clustering (computed once, from the
    order-statistics reconstruction -- the clustering itself uses no outcome/label
    information, so reusing it across methods and across CV folds within a day is not a
    source of leakage). Checks capacity feasibility up front: a capacity smaller than
    some single order's own weight/volume makes the whole day unsolvable regardless of
    fleet size, which previously surfaced only as a silent OR-Tools NO_SOLUTION deep in
    the solver (see capacity.py docstring)."""
    cap = capacity_for_scenario(capacity_multiplier)
    assert_feasible_capacity(inst, cap)
    n_veh = min_vehicles_needed(inst, cap) + vehicle_slack
    arrays = build_arrays(inst)
    recon_os = reconstruct(inst.time_optimistic, inst.time_mostlikely, inst.time_pessimistic, "order_statistics")
    clusters = select_n_clusters(recon_os.mostlikely)
    return dict(day=inst.day, arrays=arrays, clusters=clusters, cap=cap, n_veh=n_veh, rho=rho)


def _cross_reconstruction_breakdown(sol_routes: list, arrays: ProblemArrays, labels,
                                     rho: float, n_draws: int, rng: np.random.Generator) -> dict:
    """Evaluate an already-solved, fixed set of routes under each of the 3 official
    scenario reconstructions (order_statistics, trust_mode, trust_envelope) in turn, plus
    their unweighted mean. Both B0 and B1 are built and headline-evaluated against a
    single reconstruction (order_statistics) -- the right number for what each method is
    on its own terms -- but that leaves open how much the choice of reconstruction method
    would have changed the assessment. This function answers that directly, without
    changing what either method's official on_time_rate means. See
    experiments/run_cross_reconstruction_check.py, which calls this (via
    evaluate_method's cross_evaluate=True) at each method's already-tuned LODO-CV
    operating point and reports the per-reconstruction breakdown."""
    per_recon = {}
    for e in ALL_METHODS:
        recon_e = reconstruct(arrays.o_raw, arrays.m_raw, arrays.p_raw, e)
        params_e = calibrate_beta_pert(recon_e)
        metrics_e = simulate_solution(sol_routes, params_e, labels, arrays.service, arrays.deadlines,
                                       n_draws, rng, rho)
        per_recon[e] = metrics_e["on_time_rate"]
    return dict(per_reconstruction=per_recon, mean_on_time_rate=float(np.mean(list(per_recon.values()))))


def evaluate_method(
    method: str, arrays: ProblemArrays, clusters: ClusterAssignment | None,
    num_vehicles: int, cap_weight_kg: float, cap_volume_m3: float,
    safety_z: float, n_draws: int, rng: np.random.Generator,
    rho: float, time_limit_seconds: float, cross_evaluate: bool = False,
) -> dict:
    """cross_evaluate=False (the default, used everywhere in tuning.py and
    run_lodo_cv.py) reproduces this function's behavior exactly as before -- it never
    changes on_time_rate, distance_km, or any other existing field, for any method. The
    main LODO-CV results are unaffected by this parameter's existence.

    cross_evaluate=True additionally attaches result['cross_recon'], the method's
    solved route evaluated under all 3 official reconstructions plus their mean (see
    _cross_reconstruction_breakdown docstring for why this is needed, and
    experiments/run_cross_reconstruction_check.py for its use)."""
    labels = clusters.labels if clusters is not None else None

    if method == "B0_deterministic":
        recon = reconstruct(arrays.o_raw, arrays.m_raw, arrays.p_raw, "order_statistics")
        params = calibrate_beta_pert(recon)
        eff_time = recon.mostlikely  # z=0, no margin
        sol = solve_cvrptw(arrays.distance, eff_time, arrays.service, arrays.deadlines,
                            arrays.weights, arrays.volumes, cap_weight_kg, cap_volume_m3,
                            num_vehicles, time_limit_seconds=time_limit_seconds)
        metrics = simulate_solution(sol.routes, params, labels, arrays.service, arrays.deadlines,
                                     n_draws, rng, rho)

    elif method == "B1_global_robust":
        recon = reconstruct(arrays.o_raw, arrays.m_raw, arrays.p_raw, "order_statistics")
        params = calibrate_beta_pert(recon)
        eff_time = effective_time_matrix(recon.mostlikely, params.sd, safety_z)
        sol = solve_cvrptw(arrays.distance, eff_time, arrays.service, arrays.deadlines,
                            arrays.weights, arrays.volumes, cap_weight_kg, cap_volume_m3,
                            num_vehicles, time_limit_seconds=time_limit_seconds)
        metrics = simulate_solution(sol.routes, params, labels, arrays.service, arrays.deadlines,
                                     n_draws, rng, rho)

    else:
        raise ValueError(f"unknown method {method!r}")

    cross_recon = None
    if cross_evaluate:
        cross_recon = _cross_reconstruction_breakdown(sol.routes, arrays, labels, rho, n_draws, rng)

    return dict(
        method=method, on_time_rate=metrics["on_time_rate"],
        route_on_time_rate=metrics.get("route_on_time_rate"),
        mean_lateness_min=metrics.get("mean_lateness_min"),
        distance_km=sol.total_distance_km, n_vehicles=sol.n_vehicles_used,
        solve_status=sol.solve_status, cross_recon=cross_recon,
    )
