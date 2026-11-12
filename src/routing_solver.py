"""OR-Tools CVRPTW wrapper shared by every method (deterministic baseline B0, global
margin B1). Only the *travel-time matrix fed in* differs between methods -- the
solver itself, capacity handling, and objective are identical, which is what makes the
baseline comparisons fair (same solver, same search budget, only the uncertainty
treatment differs).

Time windows are soft (heavy lateness penalty, not a hard cutoff): a hard deadline
constraint can make the solver report outright infeasible under a large safety margin or
a tight capacity scenario, which would silently break a hyperparameter sweep. The
solver's job is to plan against its (possibly margin-inflated) travel-time estimate;
whether the plan actually holds up is measured separately, by Monte Carlo simulation
against the calibrated Beta-PERT distributions (simulation.py) -- that is the real
evaluation metric, not solver-reported feasibility.

Note on reproducibility: OR-Tools' RoutingSearchParameters proto has no random-seed field
(checked directly against its descriptor), so the guided-local-search trajectory is not
directly seedable from this wrapper -- a single solve at a fixed time budget can, in
principle, land on a marginally different local optimum between runs. This is a real,
bounded source of run-to-run noise on top of whatever the margin design itself
contributes, and experiments/run_solver_repeatability.py characterizes its size directly
rather than leaving it unquantified.
"""

from dataclasses import dataclass

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Integer scaling: OR-Tools requires integer arc costs/dimensions.
DIST_SCALE = 1000   # km -> meters
TIME_SCALE = 1      # minutes are already coarse; no scaling needed
WEIGHT_SCALE = 100  # kg -> hectograms (avoid float rounding at low weights)
VOLUME_SCALE = 100000  # m3 -> cm3/10 (volumes are small, e.g. 0.0002 m3)

LATENESS_PENALTY_PER_MIN = 500  # cost units per minute late, per stop; dominates distance
FIXED_COST_PER_VEHICLE = 50     # discourages using more vehicles than necessary
HORIZON_MINUTES = 1440          # generous upper bound on any single day's schedule


@dataclass
class RouteSolution:
    routes: list  # list of node-index lists (each starts/ends at depot=0), one per used vehicle
    objective: int
    total_distance_km: float
    n_vehicles_used: int
    solve_status: str


def solve_cvrptw(
    distance_matrix, effective_time_matrix, service_times, deadlines,
    weights, volumes, vehicle_capacity_weight_kg, vehicle_capacity_volume_m3,
    num_vehicles, time_limit_seconds=15,
    first_solution_strategy="PATH_CHEAPEST_ARC", local_search_metaheuristic="GUIDED_LOCAL_SEARCH",
) -> RouteSolution:
    n = distance_matrix.shape[0]  # includes depot at index 0
    depot = 0

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(from_index, to_index):
        i, j = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return int(round(distance_matrix[i, j] * DIST_SCALE))

    distance_cb_idx = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_cb_idx)

    def time_cb(from_index, to_index):
        i, j = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        svc = service_times[i] if i != depot else 0
        return int(round((effective_time_matrix[i, j] + svc) * TIME_SCALE))

    time_cb_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_cb_idx, HORIZON_MINUTES, HORIZON_MINUTES, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    for node in range(1, n):
        index = manager.NodeToIndex(node)
        time_dim.CumulVar(index).SetRange(0, HORIZON_MINUTES)
        time_dim.SetCumulVarSoftUpperBound(index, int(deadlines[node]), LATENESS_PENALTY_PER_MIN)

    def weight_cb(index):
        node = manager.IndexToNode(index)
        return int(round(weights[node] * WEIGHT_SCALE)) if node != depot else 0

    weight_cb_idx = routing.RegisterUnaryTransitCallback(weight_cb)
    routing.AddDimensionWithVehicleCapacity(
        weight_cb_idx, 0, [int(vehicle_capacity_weight_kg * WEIGHT_SCALE)] * num_vehicles, True, "Weight"
    )

    def volume_cb(index):
        node = manager.IndexToNode(index)
        return int(round(volumes[node] * VOLUME_SCALE)) if node != depot else 0

    volume_cb_idx = routing.RegisterUnaryTransitCallback(volume_cb)
    routing.AddDimensionWithVehicleCapacity(
        volume_cb_idx, 0, [int(vehicle_capacity_volume_m3 * VOLUME_SCALE)] * num_vehicles, True, "Volume"
    )

    for v in range(num_vehicles):
        routing.SetFixedCostOfVehicle(FIXED_COST_PER_VEHICLE, v)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = getattr(
        routing_enums_pb2.FirstSolutionStrategy, first_solution_strategy
    )
    search_parameters.local_search_metaheuristic = getattr(
        routing_enums_pb2.LocalSearchMetaheuristic, local_search_metaheuristic
    )
    whole_seconds = int(time_limit_seconds)
    search_parameters.time_limit.seconds = whole_seconds
    search_parameters.time_limit.nanos = int(round((time_limit_seconds - whole_seconds) * 1e9))
    search_parameters.log_search = False

    solution = routing.SolveWithParameters(search_parameters)

    if solution is None:
        return RouteSolution(routes=[], objective=-1, total_distance_km=float("nan"),
                              n_vehicles_used=0, solve_status="NO_SOLUTION")

    routes = []
    total_distance = 0.0
    for v in range(num_vehicles):
        index = routing.Start(v)
        route = []
        route_distance = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            prev_index = index
            index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(index)
            route_distance += distance_cb(prev_index, index)
        route.append(manager.IndexToNode(index))  # closing depot
        if len(route) > 2:  # more than just depot->depot
            routes.append(route)
            total_distance += route_distance / DIST_SCALE

    return RouteSolution(
        routes=routes,
        objective=solution.ObjectiveValue(),
        total_distance_km=total_distance,
        n_vehicles_used=len(routes),
        solve_status="OK",
    )
