"""Reconstruct internally-consistent optimistic/most-likely/pessimistic (O/M/P)
travel-time matrices from raw scenario data that violates O<=M<=P in a nontrivial
fraction of pairs (up to 23% of pairs on the worst observed day; magnitudes are real,
not rounding noise), then calibrate a Beta-PERT distribution per edge from the
corrected triple.

Three reconstruction methods are provided, each trusting a different part of the raw
elicitation as authoritative. The source dataset's own documentation for these matrices
is a single unvalidated sentence ("a Beta distribution was used to estimate the standard
deviation"), and there is no observed realization to validate any single choice against
-- so no method here is asserted as "ground truth". That is the point of carrying all
three rather than picking one: compare_methods() below quantifies how much the choice
among them actually matters, which is the paper's central data-quality finding, not
just a preprocessing detail. order_statistics is the one used to feed the routing
baselines (B0, B1) elsewhere in this codebase, chosen because it treats no raw label as
more authoritative than another; trust_mode and trust_envelope exist to measure
disagreement against it, not to be routed on themselves.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

ReconstructionMethod = Literal["order_statistics", "trust_mode", "trust_envelope"]
ALL_METHODS: tuple[ReconstructionMethod, ...] = ("order_statistics", "trust_mode", "trust_envelope")


@dataclass(frozen=True)
class ReconstructedScenarios:
    method: ReconstructionMethod
    optimistic: np.ndarray
    mostlikely: np.ndarray
    pessimistic: np.ndarray
    n_changed: dict


def reconstruct(o_raw: np.ndarray, m_raw: np.ndarray, p_raw: np.ndarray,
                 method: ReconstructionMethod) -> ReconstructedScenarios:
    """Return a corrected (O,M,P) triple guaranteeing O<=M<=P everywhere.

    - order_statistics: treat the 3 raw numbers as an unordered sample; O'=min, M'=median,
      P'=max. Symmetric, no label is treated as more reliable than another.
    - trust_mode: keep the "most likely" label as the authoritative mode, only widen O/P
      to enclose it.
    - trust_envelope: keep the "optimistic"/"pessimistic" labels as the authoritative
      bounds, only clip the mode into that envelope.
    """
    o_raw = np.asarray(o_raw, dtype=float)
    m_raw = np.asarray(m_raw, dtype=float)
    p_raw = np.asarray(p_raw, dtype=float)

    if method == "order_statistics":
        stacked = np.stack([o_raw, m_raw, p_raw], axis=0)
        o = np.min(stacked, axis=0)
        m = np.median(stacked, axis=0)
        p = np.max(stacked, axis=0)
    elif method == "trust_mode":
        o = np.minimum(o_raw, m_raw)
        p = np.maximum(p_raw, m_raw)
        m = m_raw.copy()
    elif method == "trust_envelope":
        lo = np.minimum(o_raw, p_raw)
        hi = np.maximum(o_raw, p_raw)
        o, p = lo.copy(), hi.copy()
        m = np.clip(m_raw, lo, hi)
    else:
        raise ValueError(f"unknown method {method!r}")

    np.fill_diagonal(o, 0.0)
    np.fill_diagonal(m, 0.0)
    np.fill_diagonal(p, 0.0)

    assert np.all(o <= m + 1e-9) and np.all(m <= p + 1e-9), f"{method}: ordering not enforced"

    n_changed = {
        "optimistic": int(np.sum(~np.isclose(o, o_raw))),
        "mostlikely": int(np.sum(~np.isclose(m, m_raw))),
        "pessimistic": int(np.sum(~np.isclose(p, p_raw))),
    }
    return ReconstructedScenarios(method=method, optimistic=o, mostlikely=m, pessimistic=p, n_changed=n_changed)


@dataclass(frozen=True)
class BetaPertParams:
    a: np.ndarray       # optimistic (min support)
    m: np.ndarray        # mode (most likely)
    b: np.ndarray         # pessimistic (max support)
    mean: np.ndarray
    sd: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    degenerate: np.ndarray  # True where b - a ~ 0 (deterministic edge, e.g. depot self)


def calibrate_beta_pert(scenarios: ReconstructedScenarios, shape_lambda: float = 4.0) -> BetaPertParams:
    """Modified-PERT calibration (Vose 2008); shape_lambda=4 recovers the classical
    PERT mean (a+4m+b)/6 and SD (b-a)/6 (Malcolm et al. 1959)."""
    a, m, b = scenarios.optimistic, scenarios.mostlikely, scenarios.pessimistic
    width = b - a
    degenerate = width < 1e-9
    safe_width = np.where(degenerate, 1.0, width)

    mean = (a + shape_lambda * m + b) / (shape_lambda + 2.0)
    sd = width / 6.0

    alpha = np.where(degenerate, np.nan, 1.0 + shape_lambda * (mean - a) / safe_width)
    beta = np.where(degenerate, np.nan, 1.0 + shape_lambda * (b - mean) / safe_width)

    return BetaPertParams(a=a, m=m, b=b, mean=mean, sd=sd, alpha=alpha, beta=beta, degenerate=degenerate)


def compare_methods(o_raw: np.ndarray, m_raw: np.ndarray, p_raw: np.ndarray) -> dict:
    """Pairwise disagreement between the 3 reconstruction methods, on the mean
    travel-time estimate they each imply. This is the direct evidence for how much
    reconstruction choice matters: if the methods mostly agree, the choice among them is
    close to immaterial; where they disagree, any downstream routing decision that
    assumed only one of them is exposed to that disagreement without knowing it.
    experiments/run_cross_reconstruction_check.py follows this through to its effect on
    B0/B1's assessed reliability."""
    recons = {meth: reconstruct(o_raw, m_raw, p_raw, meth) for meth in ALL_METHODS}
    params = {meth: calibrate_beta_pert(r) for meth, r in recons.items()}

    n = o_raw.shape[0]
    offdiag = ~np.eye(n, dtype=bool)
    means = {meth: p.mean[offdiag] for meth, p in params.items()}

    out = {"n_changed": {meth: r.n_changed for meth, r in recons.items()}}
    pairs = [("order_statistics", "trust_mode"), ("order_statistics", "trust_envelope"),
             ("trust_mode", "trust_envelope")]
    disagreement = {}
    for a_name, b_name in pairs:
        diff = np.abs(means[a_name] - means[b_name])
        disagreement[f"{a_name}_vs_{b_name}"] = {
            "mean_abs_diff_min": float(diff.mean()),
            "max_abs_diff_min": float(diff.max()),
            "pct_pairs_diff_gt_1min": float(100.0 * np.mean(diff > 1.0)),
        }
    out["mean_disagreement"] = disagreement
    return out


if __name__ == "__main__":
    from src.data_loading import load_all_days

    days = load_all_days()
    for d, inst in days.items():
        cmp = compare_methods(inst.time_optimistic, inst.time_mostlikely, inst.time_pessimistic)
        os_changed = cmp["n_changed"]["order_statistics"]
        worst = max(v["pct_pairs_diff_gt_1min"] for v in cmp["mean_disagreement"].values())
        print(f"Day {d}: order_statistics changed O/M/P = "
              f"{os_changed['optimistic']}/{os_changed['mostlikely']}/{os_changed['pessimistic']} "
              f"pairs; worst inter-method disagreement (>1min) = {worst:.1f}% of pairs")
