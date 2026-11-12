"""Monte Carlo evaluation of a fixed route against the calibrated Beta-PERT travel-time
model. This is the actual ground-truth evaluation used everywhere in the experiment --
the solver's own soft-penalty objective is only what it plans against, not what gets
reported.

Within-cluster edges share a correlated random shock (Gaussian copula: a shared cluster
factor plus idiosyncratic edge noise, transformed through each edge's own Beta-PERT
quantile function so the marginal distribution stays correct) -- modelling that a single
local traffic event plausibly affects several nearby trips at once, matching the
diagnosed spatial block structure in the uncertainty data. Cross-cluster edges are drawn
independently. The within-cluster correlation strength rho is a model parameter, not
something estimated from data: this dataset has no repeated-day panel to estimate a true
edge-level correlation from, so rho is fixed at a moderate default (0.3) throughout.

rho is deliberately NOT included in the fixed-hyperparameter sensitivity sweeps in
experiments/run_sensitivity.py (contrast capacity_sensitivity, which holds z fixed and
varies a genuinely external world parameter). B1's margin (src/margins.py) does not use
rho at all -- it is a plain uniform z*SD margin -- so rho only ever affects this
evaluator, identically for B0 and B1, with no risk of a swept value leaking into how
either method plans. The reason not to sweep it is simpler: this dataset has no
repeated-day panel to estimate a true edge-level correlation from, so there is no
empirical basis for what a "true, different" rho should even be set to -- a sweep would
be exploring an arbitrary range rather than testing against anything observed. rho is
instead reported as a stated, fixed modelling assumption and flagged as a limitation in
the paper rather than presented as tested.

GPU note: CUDA-enabled PyTorch was attempted for this simulation loop but abandoned after
repeated environment-level install failures (DNS resolution to the CUDA wheel host failed
consistently; the one build that did install was missing required runtime DLLs). Not
pursued further because it would only accelerate a part of the pipeline that was never the
bottleneck: OR-Tools' routing solve is CPU-only regardless, and this NumPy/SciPy
implementation already runs the full 9-day study end to end in a practical amount of
time. No accelerator module exists; this is the only implementation.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.reconstruction import BetaPertParams

DEFAULT_RHO = 0.3


@dataclass(frozen=True)
class RouteSimResult:
    route: list
    n_draws: int
    arrival_times: np.ndarray      # (n_draws, n_customer_stops)
    on_time: np.ndarray            # (n_draws, n_customer_stops) bool
    lateness_min: np.ndarray       # (n_draws, n_customer_stops)
    completion_time: np.ndarray    # (n_draws,)
    stop_nodes: list               # node id for each column of arrival_times/on_time

    @property
    def on_time_rate(self) -> float:
        return float(self.on_time.mean())

    @property
    def route_on_time_rate(self) -> float:
        """Fraction of draws where every stop on the route was on time."""
        return float(self.on_time.all(axis=1).mean())

    @property
    def mean_lateness_min(self) -> float:
        return float(self.lateness_min.mean())


def simulate_route(route: list, params: BetaPertParams, labels: np.ndarray | None,
                    service_times: np.ndarray, deadlines: np.ndarray,
                    n_draws: int, rng: np.random.Generator, rho: float = DEFAULT_RHO) -> RouteSimResult:
    edges = list(zip(route[:-1], route[1:]))
    n_edges = len(edges)
    idx_from = np.array([e[0] for e in edges])
    idx_to = np.array([e[1] for e in edges])

    a = params.a[idx_from, idx_to]
    b = params.b[idx_from, idx_to]
    alpha = params.alpha[idx_from, idx_to]
    beta_ = params.beta[idx_from, idx_to]
    degenerate = params.degenerate[idx_from, idx_to]

    if labels is not None:
        cluster_of_edge = np.where(labels[idx_from] == labels[idx_to], labels[idx_from], -1)
    else:
        cluster_of_edge = np.full(n_edges, -1)

    eps = rng.standard_normal((n_draws, n_edges))
    unique_clusters = sorted(c for c in set(cluster_of_edge.tolist()) if c != -1)
    z_per_edge = np.zeros((n_draws, n_edges))
    if unique_clusters:
        z = rng.standard_normal((n_draws, len(unique_clusters)))
        cluster_col = {c: k for k, c in enumerate(unique_clusters)}
        for e in range(n_edges):
            c = cluster_of_edge[e]
            if c != -1:
                z_per_edge[:, e] = z[:, cluster_col[c]]

    rho_eff = np.where(cluster_of_edge != -1, rho, 0.0)
    x = np.sqrt(rho_eff)[None, :] * z_per_edge + np.sqrt(1.0 - rho_eff)[None, :] * eps
    u = np.clip(stats.norm.cdf(x), 1e-6, 1.0 - 1e-6)

    realized = np.empty((n_draws, n_edges))
    for e in range(n_edges):
        if degenerate[e]:
            realized[:, e] = a[e]
        else:
            realized[:, e] = a[e] + (b[e] - a[e]) * stats.beta.ppf(u[:, e], alpha[e], beta_[e])

    dest_is_depot = idx_to == 0
    service_at_arrival = np.where(dest_is_depot, 0.0, service_times[idx_to])
    combined = realized + service_at_arrival[None, :]
    cum_departure = np.cumsum(combined, axis=1)
    arrival_all = cum_departure - service_at_arrival[None, :]

    customer_cols = ~dest_is_depot
    arrival_times = arrival_all[:, customer_cols]
    stop_nodes = idx_to[customer_cols].tolist()
    deadline_arr = deadlines[idx_to[customer_cols]]

    on_time = arrival_times <= deadline_arr[None, :]
    lateness = np.maximum(0.0, arrival_times - deadline_arr[None, :])
    completion_time = cum_departure[:, -1]

    return RouteSimResult(
        route=route, n_draws=n_draws, arrival_times=arrival_times, on_time=on_time,
        lateness_min=lateness, completion_time=completion_time, stop_nodes=stop_nodes,
    )


def simulate_solution(routes: list, params: BetaPertParams, labels: np.ndarray | None,
                       service_times: np.ndarray, deadlines: np.ndarray,
                       n_draws: int, rng: np.random.Generator, rho: float = DEFAULT_RHO) -> dict:
    """Simulate every route in a solution; return pooled fleet-level metrics."""
    results = [simulate_route(r, params, labels, service_times, deadlines, n_draws, rng, rho)
               for r in routes if len(r) > 2]
    if not results:
        return dict(on_time_rate=float("nan"), route_on_time_rate=float("nan"),
                     mean_lateness_min=float("nan"), n_routes=0, per_route=[])

    all_on_time = np.concatenate([r.on_time.reshape(-1) for r in results])
    all_lateness = np.concatenate([r.lateness_min.reshape(-1) for r in results])
    route_all_on_time = np.stack([r.on_time.all(axis=1) for r in results], axis=1)  # (n_draws, n_routes)
    fleet_on_time_rate = route_all_on_time.all(axis=1).mean()  # every route fully on-time, per draw

    return dict(
        on_time_rate=float(all_on_time.mean()),
        route_on_time_rate=float(np.mean([r.route_on_time_rate for r in results])),
        fleet_fully_on_time_rate=float(fleet_on_time_rate),
        mean_lateness_min=float(all_lateness.mean()),
        p95_lateness_min=float(np.percentile(all_lateness, 95)),
        n_routes=len(results),
        per_route=results,
    )


if __name__ == "__main__":
    # minimal hand-checkable unit test: 2 edges, zero variance (a==m==b), so the
    # simulated arrival time must exactly equal the deterministic sum.
    from src.reconstruction import reconstruct, calibrate_beta_pert

    o = np.array([[0, 10, 20], [10, 0, 5], [20, 5, 0]], dtype=float)
    m = o.copy()
    p = o.copy()
    recon = reconstruct(o, m, p, "order_statistics")
    params = calibrate_beta_pert(recon)
    service = np.array([0, 3, 0])
    deadlines = np.array([10**6, 100, 100])
    rng = np.random.default_rng(0)
    res = simulate_route([0, 1, 2, 0], params, None, service, deadlines, n_draws=50, rng=rng)
    expected_arrival_1 = 10.0
    expected_arrival_2 = 10.0 + 3.0 + 5.0
    print("deterministic-edge sanity check:")
    print(" arrival at stop 1 (expect 10.0):", res.arrival_times[0, 0])
    print(" arrival at stop 2 (expect 18.0):", res.arrival_times[0, 1])
    assert np.allclose(res.arrival_times[:, 0], expected_arrival_1)
    assert np.allclose(res.arrival_times[:, 1], expected_arrival_2)
    print(" PASSED")
