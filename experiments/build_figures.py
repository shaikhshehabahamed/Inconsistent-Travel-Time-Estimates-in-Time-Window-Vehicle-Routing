"""Generate every figure the paper needs, at Q1 publication quality: PNG (300dpi) and
PDF (vector) for each, consistent theme (plotting_style.py), colorblind-safe, method
colors fixed across every figure. Reads results/raw/lodo_checkpoint.json,
results/raw/sensitivity.json, results/tables/statistical_comparisons.csv, and recomputes
the data-diagnostic figures directly from the raw dataset (not from hardcoded numbers).
The pipeline diagram (fig_pipeline_diagram) is the one figure with no data dependency --
it is a static description of the method, not a plot of results. Writes to
results/figures/.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plotting_style import DOUBLE_COL_IN, METHOD_COLORS, METHOD_LABELS, METHOD_ORDER, SINGLE_COL_IN, \
    SQUARE_IN, apply_style, save_fig

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
TABLES_DIR = ROOT / "results" / "tables"
RAW_DIR = ROOT / "results" / "raw"


def _bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return values.mean(), lo, hi


def fig_scenario_inconsistency():
    """Recomputed directly from the raw matrices (src.reconstruction), not hardcoded.
    This is the paper's headline figure: the data-quality problem that motivates
    everything else."""
    from src.data_loading import load_all_days
    days = load_all_days()
    pct_om = []
    for d, inst in days.items():
        n = inst.n_nodes
        offdiag = ~np.eye(n, dtype=bool)
        om_viol = (inst.time_optimistic > inst.time_mostlikely) & offdiag
        pct_om.append(100.0 * om_viol.sum() / offdiag.sum())

    fig, ax = plt.subplots(figsize=SINGLE_COL_IN)
    ax.bar(range(1, 10), pct_om, color="#1F6F6B", edgecolor="white", linewidth=0.6)
    for i, v in enumerate(pct_om):
        ax.text(i + 1, v + max(pct_om) * 0.02, f"{v:.1f}%", ha="center", fontsize=7)
    ax.set_xlabel("Day")
    ax.set_ylabel("Pairs with optimistic > most-likely (%)")
    ax.set_title("Raw scenario-label inconsistency, by day")
    ax.set_xticks(range(1, 10))
    fig.tight_layout()
    save_fig(fig, FIG_DIR / "fig1_scenario_inconsistency")
    plt.close(fig)


def fig_pipeline_diagram():
    """Static description of the pipeline, not a plot of results -- has no data
    dependency, but lives here so every figure the paper uses is produced by this one
    script. Shows how the two strands of evidence relate: the reconstruction-disagreement
    diagnosis (top, feeding Figure 1) and the routing comparison it motivates (bottom,
    B0 vs B1), plus the check that ties them together (the cross-reconstruction
    downstream-impact check, right)."""
    COL_GRAY = "#999999"
    COL_ORANGE = "#E69F00"
    COL_PINK = "#CC79A7"
    COL_LBLUE = "#56B4E9"
    COL_DBLUE = "#0072B2"
    COL_GREEN = "#1F6F6B"
    COL_TEXT_LIGHT = "white"
    COL_TEXT_DARK = "#222222"

    fig, ax = plt.subplots(figsize=(7.9, 9.6))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 15.4)
    ax.axis("off")

    def box(cx, cy, w, h, text, color, textcolor=COL_TEXT_LIGHT, fontsize=8.3, weight="normal"):
        b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                            boxstyle="round,pad=0.06,rounding_size=0.08",
                            facecolor=color, edgecolor="none", zorder=2)
        ax.add_patch(b)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, color=textcolor,
                weight=weight, zorder=3, linespacing=1.35, wrap=True)
        return (cx, cy, w, h)

    def arrow(b_from, b_to, style="-", color="#444444", connectionstyle="arc3,rad=0.0", lw=1.1):
        x0, y0, w0, h0 = b_from
        x1, y1, w1, h1 = b_to
        start = (x0, y0 - h0 / 2) if y0 > y1 else (x0, y0 + h0 / 2)
        end = (x1, y1 + h1 / 2) if y0 > y1 else (x1, y1 - h1 / 2)
        a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                             color=color, linewidth=lw, linestyle=style,
                             connectionstyle=connectionstyle, zorder=1)
        ax.add_patch(a)

    def side_arrow(pt_from, pt_to, color="#888888", connectionstyle="arc3,rad=0.25", lw=1.0):
        a = FancyArrowPatch(pt_from, pt_to, arrowstyle="-|>", mutation_scale=9,
                             color=color, linewidth=lw, linestyle="--",
                             connectionstyle=connectionstyle, zorder=1)
        ax.add_patch(a)

    ax.text(5.2, 15.05, "Pipeline: reconstruction-ambiguity diagnosis and its downstream routing impact",
            ha="center", va="center", fontsize=11.0, weight="bold")

    # Row A: raw data
    raw = box(5.2, 13.9, 8.2, 0.85,
              "Raw optimistic / most likely / pessimistic travel-time matrices\n(9 days, Attica dataset)",
              COL_GRAY)

    # Row B: three reconstruction rules
    r1 = box(1.95, 12.35, 3.1, 1.05, "Order-statistics\nreconstruction", COL_ORANGE)
    r2 = box(5.2, 12.35, 3.1, 1.05, "Trust-mode\nreconstruction", COL_ORANGE)
    r3 = box(8.45, 12.35, 3.1, 1.05, "Trust-envelope\nreconstruction", COL_ORANGE)
    for r in (r1, r2, r3):
        arrow(raw, r)

    # Row C: disagreement quantification -> Figure 1
    disagree = box(5.2, 10.75, 8.2, 1.0,
                    "Disagreement quantification across the three repaired estimates\n"
                    "(up to 22.9% of pairs, worst day)  ->  Figure 1",
                    COL_PINK, textcolor=COL_TEXT_DARK)
    for r in (r1, r2, r3):
        arrow(r, disagree)

    ax.text(0.2, 13.25, "Data-quality\ndiagnosis", ha="left", va="center", fontsize=8.6,
            weight="bold", rotation=90, color="#333333")

    # Row D: order-statistics feeds the routing strand (clustering)
    cluster = box(5.2, 9.1, 6.4, 0.95,
                   "Order-statistics reconstruction + Beta-PERT calibration,\n"
                   "combined with data-driven spatial clustering",
                   COL_LBLUE, textcolor=COL_TEXT_DARK)
    arrow(r1, cluster, style="--", connectionstyle="arc3,rad=-0.3", color="#888888")

    # Row E: B0 / B1 solve
    b0 = box(2.85, 7.5, 4.3, 1.05, "B0: CVRPTW solve\nno margin", COL_GRAY)
    b1 = box(7.55, 7.5, 4.3, 1.05,
             "B1: CVRPTW solve\ntuned uniform margin z*SD\n(same solver & objective as B0)",
             COL_DBLUE, fontsize=7.6)
    arrow(cluster, b0, connectionstyle="arc3,rad=0.12")
    arrow(cluster, b1, connectionstyle="arc3,rad=-0.12")

    # Row F: Monte Carlo evaluation
    mc = box(5.2, 5.95, 8.2, 0.9,
             "Monte Carlo ground-truth evaluation\n(within-cluster correlated Beta-PERT draws)",
             COL_PINK, textcolor=COL_TEXT_DARK)
    arrow(b0, mc, connectionstyle="arc3,rad=-0.1")
    arrow(b1, mc, connectionstyle="arc3,rad=0.1")

    # Row G: LODO-CV comparison
    lodo = box(5.2, 4.45, 8.2, 1.0,
               "Leave-one-day-out cross-validated comparison: B0 vs. B1\n-> Section 6 routing result",
               COL_GREEN)
    arrow(mc, lodo)

    ax.text(0.2, 6.9, "Routing\ncomparison", ha="left", va="center", fontsize=8.6,
            weight="bold", rotation=90, color="#333333")

    # Row H: cross-reconstruction downstream-impact check
    xcheck = box(5.2, 2.5, 8.2, 1.35,
                 "Cross-reconstruction downstream-impact check\n"
                 "Re-score B0's and B1's already-tuned routes under all three\n"
                 "reconstructions from Row B -> does the Figure 1 disagreement\n"
                 "actually change the B0-vs-B1 conclusion?",
                 "#D55E00")
    arrow(lodo, xcheck, style="--", color="#888888")

    # Feedback arrow routed along the right margin as one dashed polyline (clear of
    # every box in between), with a single arrowhead where it re-enters "disagree".
    x_margin = 10.05
    y_start = xcheck[1] + 0.25
    y_end = disagree[1]
    ax.plot([xcheck[0] + xcheck[2] / 2 - 0.1, x_margin], [y_start, y_start],
            color="#888888", linewidth=1.0, linestyle="--", zorder=1)
    ax.plot([x_margin, x_margin], [y_start, y_end], color="#888888", linewidth=1.0,
            linestyle="--", zorder=1)
    side_arrow((x_margin, y_end), (disagree[0] + disagree[2] / 2 - 0.05, y_end),
               connectionstyle="arc3,rad=0.0")

    ax.text(0.2, 2.5, "Downstream-\nimpact check", ha="left", va="center", fontsize=8.6,
            weight="bold", rotation=90, color="#333333")

    fig.tight_layout()
    save_fig(fig, FIG_DIR / "fig2_pipeline_diagram")
    plt.close(fig)


def fig_main_tradeoff(long_df: pd.DataFrame):
    """Distance (x, lower=better) vs on-time rate (y, higher=better), one point per
    method with bootstrap 95% CI error bars. A method up-and-left of another dominates
    it on both axes at once -- the strongest possible visual claim."""
    fig, ax = plt.subplots(figsize=SQUARE_IN)
    for method in METHOD_ORDER:
        sub = long_df[long_df.method == method]
        if len(sub) == 0:
            continue
        ot_mean, ot_lo, ot_hi = _bootstrap_ci(sub.on_time_rate.values)
        d_mean, d_lo, d_hi = _bootstrap_ci(sub.distance_km.values)
        color = METHOD_COLORS[method]
        ax.errorbar(d_mean, ot_mean, xerr=[[d_mean - d_lo], [d_hi - d_mean]],
                    yerr=[[ot_mean - ot_lo], [ot_hi - ot_mean]],
                    fmt="o", color=color, ecolor=color, elinewidth=1.2, capsize=3,
                    markersize=7, markeredgecolor="white", markeredgewidth=0.8, zorder=3,
                    label=METHOD_LABELS[method])
    ax.legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.9,
              edgecolor="none", facecolor="white", handletextpad=0.4, borderpad=0.6,
              labelspacing=0.4)
    ax.set_xlabel("Mean route distance (km) — lower is better")
    ax.set_ylabel("Mean on-time rate — higher is better")
    ax.set_title("Service–cost tradeoff, 9-fold leave-one-day-out mean ± 95% CI")
    fig.tight_layout()
    save_fig(fig, FIG_DIR / "fig3_main_tradeoff")
    plt.close(fig)


def fig_method_bars(long_df: pd.DataFrame):
    """Two-panel bar chart: on-time rate and distance per method, with error bars,
    B0 vs B1, for readers who prefer bars to a scatter."""
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL_IN)
    methods = [m for m in METHOD_ORDER if m in long_df.method.unique()]
    x = np.arange(len(methods))

    for ax, col, ylabel in zip(axes, ["on_time_rate", "distance_km"],
                                ["On-time rate", "Distance (km)"]):
        means, los, his = [], [], []
        for m in methods:
            vals = long_df[long_df.method == m][col].values
            mean, lo, hi = _bootstrap_ci(vals)
            means.append(mean); los.append(mean - lo); his.append(hi - mean)
        colors = [METHOD_COLORS[m] for m in methods]
        ax.bar(x, means, yerr=[los, his], color=colors, capsize=3,
               edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        if col == "on_time_rate":
            ax.set_ylim(min(means) - 0.03, 1.005)
    fig.suptitle("Method comparison, 9-fold mean ± 95% CI", y=1.02)
    fig.tight_layout()
    save_fig(fig, FIG_DIR / "fig4_method_bars")
    plt.close(fig)


def fig_sensitivity(sens: dict):
    """Two panels: capacity sensitivity (hyperparameter held fixed, environment varied)
    and service-target sensitivity (re-tuned per target). Both B1-only -- B0 has no
    tunable margin, so it is not a meaningful subject for either sweep."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_IN[0] * 0.95, DOUBLE_COL_IN[1]))

    ax = axes[0]
    caps = sens["capacity"]
    mults = [r["capacity_multiplier"] for r in caps]
    ax.plot(mults, [r["B1"]["on_time_rate"] for r in caps], "o-",
            color=METHOD_COLORS["B1_global_robust"], markersize=4)
    ax.set_xlabel("Capacity multiplier (1.0 = 500kg/3.0m3)")
    ax.set_ylabel("On-time rate (B1)")
    ax.set_title("(a) Capacity sensitivity")

    ax = axes[1]
    targets = sens["service_target"]
    tv = [r["target"] for r in targets]
    dist = [r["B1"]["distance_km"] for r in targets]
    met = [r["B1"]["on_time_rate"] >= r["target"] for r in targets]
    color = METHOD_COLORS["B1_global_robust"]
    ax.plot(tv, dist, "-", color=color, linewidth=1.3, zorder=2)
    any_missed = False
    for x, y, m in zip(tv, dist, met):
        any_missed = any_missed or not m
        ax.plot(x, y, "o", color=color if m else "white", markeredgecolor=color,
                 markeredgewidth=1.2, markersize=5.5, zorder=3)
    ax.set_xlabel("Target service level")
    ax.set_ylabel("Distance at tuned operating point (km)")
    ax.set_title("(b) Service-target sensitivity")
    if any_missed:
        ax.annotate("hollow = target not reached\nby grid's worst-case day", xy=(0.5, 0.5),
                    xycoords="axes fraction", fontsize=6.3, color="#555555", ha="center", va="center")

    fig.tight_layout()
    save_fig(fig, FIG_DIR / "fig6_sensitivity")
    plt.close(fig)


def fig_robustness_checks(mc_conv: dict, solver_rep: list):
    """Two supplementary robustness checks in one figure: (a) does the Monte Carlo
    on-time-rate estimate stabilize by FINAL_DRAWS=5000 (justifies the draw count used
    everywhere in the main experiment), and (b) is OR-Tools' unseeded solver noise small
    relative to the B1-vs-B0 effect size the paper's routing comparison rests on, or
    could it be driving it? Both are the "multiple random seed evaluation" robustness
    evidence."""
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL_IN)

    ax = axes[0]
    day_colors = {"1": "#0072B2", "5": "#D55E00", "9": "#009E73"}
    day_labels = {"1": "Day 1 (small, 79 nodes)", "5": "Day 5 (medium, 76 nodes)", "9": "Day 9 (large, 85 nodes)"}
    for day, by_count in mc_conv.items():
        counts = sorted(int(k) for k in by_count.keys())
        stds = [by_count[str(c)]["std_across_seeds"] for c in counts]
        ax.plot(counts, stds, "o-", color=day_colors[day], label=day_labels[day], markersize=4, linewidth=1.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(5000, color="#999999", linewidth=0.9, linestyle="--", zorder=1)
    ax.annotate("used in main\nexperiment", xy=(0.83, 0.95), xycoords="axes fraction",
                fontsize=6.3, color="#666666", ha="center", va="top")
    ax.set_xlabel("Monte Carlo draws")
    ax.set_ylabel("On-time rate std across 12 seeds")
    ax.set_title("(a) MC draw-count convergence")
    ax.legend(fontsize=6.2, loc="lower left")

    ax = axes[1]
    rep_df = pd.DataFrame(solver_rep)
    methods = [m for m in METHOD_ORDER if m in rep_df.method.unique()]
    mean_std = [rep_df[rep_df.method == m].on_time_std.mean() for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    y = np.arange(len(methods))

    # Raw std and the reported B1-vs-B0 effect size sit an order of magnitude apart --
    # sharing one linear axis made every bar an invisible sliver. Normalizing to "% of
    # the B1-vs-B0 effect size" puts every quantity of interest on one readable axis and
    # directly answers the question this panel exists for: could solver noise alone
    # explain the reported effect?
    b1_vs_b0_effect = None
    stats_path = TABLES_DIR / "statistical_comparisons.csv"
    if stats_path.exists():
        sdf = pd.read_csv(stats_path)
        sdf = sdf[sdf.metric == "on_time_rate"]
        row = sdf[sdf.baseline == "B0_deterministic"]
        b1_vs_b0_effect = float(row.mean_diff.iloc[0]) if len(row) else None

    if b1_vs_b0_effect:
        pct = [100 * s / b1_vs_b0_effect for s in mean_std]
        ax.barh(y, pct, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
        for yi, p in zip(y, pct):
            ax.text(p + max(pct) * 0.03, yi, f"{p:.1f}%", fontsize=6.5, va="center", color="#333333")
        ax.set_xlim(0, max(pct) * 1.28)
        ax.set_xlabel("Solver noise, % of the B1-vs-B0 effect size")
        ax.annotate(f"B1-vs-B0 effect size = {b1_vs_b0_effect*100:.2f}pp", xy=(0.5, -0.24),
                    xycoords="axes fraction", fontsize=6.2, color="#666666", ha="center")
    else:
        ax.barh(y, mean_std, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_xlabel("Mean on-time-rate std across 5 repeat-solves")
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS[m] for m in methods], fontsize=6.8)
    ax.set_title("(b) Solver noise vs. reported effect size")

    fig.suptitle("Robustness checks: simulation precision and solver-seed sensitivity", fontsize=9.5, y=1.03)
    fig.tight_layout()
    save_fig(fig, FIG_DIR / "fig5_robustness_checks")
    plt.close(fig)


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig_scenario_inconsistency()
    fig_pipeline_diagram()

    long_df_path = TABLES_DIR / "per_fold_results.csv"
    if long_df_path.exists():
        long_df = pd.read_csv(long_df_path)
        fig_main_tradeoff(long_df)
        fig_method_bars(long_df)
    else:
        print("skip main-result figures: run build_results_tables.py first")

    sens_path = RAW_DIR / "sensitivity.json"
    if sens_path.exists():
        with open(sens_path) as f:
            sens = json.load(f)
        fig_sensitivity(sens)
    else:
        print("skip sensitivity figure: run run_sensitivity.py first")

    mc_path = RAW_DIR / "mc_convergence.json"
    rep_path = RAW_DIR / "solver_repeatability.json"
    if mc_path.exists() and rep_path.exists():
        with open(mc_path) as f:
            mc_conv = json.load(f)
        with open(rep_path) as f:
            solver_rep = json.load(f)
        fig_robustness_checks(mc_conv, solver_rep)
    else:
        print("skip robustness-checks figure: run run_mc_convergence.py and run_solver_repeatability.py first")

    print("Figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
