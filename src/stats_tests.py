"""Paired statistical comparison of one method (e.g. B1, the global margin) against a set
of baselines (e.g. B0, the deterministic no-margin plan) across the 9 leave-one-day-out
folds. Wilcoxon signed-rank (nonparametric, appropriate for n=9 without assuming
normality) with Holm-Bonferroni correction across the family of baseline comparisons;
matched-pairs rank-biserial effect size; percentile bootstrap confidence intervals on the
mean paired difference. Method-agnostic: which method plays "proposed" and which play
"baselines" is decided by the caller, not by anything in this module.

n=9 is small. Every function here returns n explicitly alongside its result so a reader
of the output tables is never left to assume a larger sample than actually exists.
"""

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


@dataclass
class PairedComparison:
    baseline: str
    metric: str
    n: int
    mean_diff: float
    ci_lower: float
    ci_upper: float
    wilcoxon_stat: float
    p_value: float
    p_value_corrected: float
    rank_biserial: float
    reject_null: bool


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                         rng: np.random.Generator | None = None) -> tuple:
    rng = rng if rng is not None else np.random.default_rng(0)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def rank_biserial_from_diffs(diffs: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation, the standard effect size to accompany
    a Wilcoxon signed-rank test: (sum of ranks favouring proposed - sum favouring
    baseline) / total rank sum. +1 = proposed wins every pair, -1 = loses every pair."""
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    pos_sum = ranks[nonzero > 0].sum()
    neg_sum = ranks[nonzero < 0].sum()
    total = pos_sum + neg_sum
    return float((pos_sum - neg_sum) / total) if total > 0 else 0.0


def compare_to_baselines(proposed: dict, baselines: dict, metric: str, alpha: float = 0.05,
                          n_boot: int = 10000, seed: int = 0) -> list:
    """proposed: {day: value}. baselines: {baseline_name: {day: value}}, same day keys.
    Returns one PairedComparison per baseline, p_value_corrected via Holm-Bonferroni
    across the whole family of baselines being compared here."""
    rng = np.random.default_rng(seed)
    days = sorted(proposed.keys())
    p = np.array([proposed[d] for d in days], dtype=float)

    raw = []
    for name, bvals in baselines.items():
        b = np.array([bvals[d] for d in days], dtype=float)
        diffs = p - b
        if np.allclose(diffs, 0):
            stat, pval = 0.0, 1.0
        else:
            stat, pval = stats.wilcoxon(p, b, zero_method="wilcox", alternative="two-sided")
        ci_lo, ci_hi = paired_bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, rng=rng)
        rb = rank_biserial_from_diffs(diffs)
        raw.append(dict(baseline=name, metric=metric, n=len(days), mean_diff=float(diffs.mean()),
                         ci_lower=ci_lo, ci_upper=ci_hi, wilcoxon_stat=float(stat), p_value=float(pval),
                         rank_biserial=rb))

    pvals = [r["p_value"] for r in raw]
    reject, p_corrected, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    results = []
    for r, rej, pc in zip(raw, reject, p_corrected):
        r["p_value_corrected"] = float(pc)
        r["reject_null"] = bool(rej)
        results.append(PairedComparison(**r))
    return results


if __name__ == "__main__":
    # synthetic validation: proposed is drawn 0.03 higher than baseline A on average with
    # real per-fold noise (should detect a significant, positive effect), and IDENTICAL
    # to baseline B (should NOT reject the null for B).
    rng = np.random.default_rng(0)
    days = list(range(1, 10))
    base_a = {d: v for d, v in zip(days, rng.normal(0.85, 0.03, size=9))}
    proposed = {d: base_a[d] + 0.03 + rng.normal(0, 0.005) for d in days}
    base_b = {d: proposed[d] + rng.normal(0, 1e-9) for d in days}  # essentially identical

    comps = compare_to_baselines(proposed, {"baseline_A": base_a, "baseline_B_identical": base_b},
                                  metric="on_time_rate", seed=1)
    for c in comps:
        print(f"{c.baseline:22s} mean_diff={c.mean_diff:+.4f} CI=[{c.ci_lower:+.4f},{c.ci_upper:+.4f}] "
              f"p={c.p_value:.4f} p_holm={c.p_value_corrected:.4f} rank_biserial={c.rank_biserial:+.3f} "
              f"reject={c.reject_null} n={c.n}")
    assert comps[0].mean_diff > 0.02 and comps[0].reject_null, "expected to detect the real +0.03 effect"
    assert not comps[1].reject_null, "expected NOT to reject for the near-identical baseline"
    print("PASSED: detects the real effect, does not falsely flag the null case")
