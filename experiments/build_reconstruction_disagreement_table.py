"""Persist src.reconstruction.compare_methods()'s per-day disagreement diagnostics to a
table. This is the paper's central data-quality evidence (Section 3.3 / Section 6): how
much the three reconstruction rules disagree with each other, and how often the raw
labels violate optimistic<=most_likely<=pessimistic in the first place. Figure 1 plots
only the optimistic>most_likely violation rate; this table is the source for the other
disagreement numbers cited in the text (mean/max travel-time disagreement in minutes,
percentage of pairs disagreeing by more than a minute).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_all_days
from src.reconstruction import ALL_METHODS, compare_methods

TABLES_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"


def main():
    days = load_all_days()
    rows = []
    pair_rows = []
    for d, inst in days.items():
        n = inst.n_nodes
        offdiag = ~np.eye(n, dtype=bool)
        n_pairs = int(offdiag.sum())

        om_viol = (inst.time_optimistic > inst.time_mostlikely) & offdiag
        pm_viol = (inst.time_pessimistic < inst.time_mostlikely) & offdiag
        po_viol = (inst.time_pessimistic < inst.time_optimistic) & offdiag

        cmp = compare_methods(inst.time_optimistic, inst.time_mostlikely, inst.time_pessimistic)

        row = dict(
            day=d, n_pairs=n_pairs,
            pct_optimistic_gt_mostlikely=100.0 * om_viol.sum() / n_pairs,
            pct_pessimistic_lt_mostlikely=100.0 * pm_viol.sum() / n_pairs,
            pct_pessimistic_lt_optimistic=100.0 * po_viol.sum() / n_pairs,
        )
        for meth in ALL_METHODS:
            row[f"n_changed_optimistic_{meth}"] = cmp["n_changed"][meth]["optimistic"]
            row[f"n_changed_mostlikely_{meth}"] = cmp["n_changed"][meth]["mostlikely"]
            row[f"n_changed_pessimistic_{meth}"] = cmp["n_changed"][meth]["pessimistic"]
        rows.append(row)

        for pair_name, stats in cmp["mean_disagreement"].items():
            pair_rows.append(dict(day=d, method_pair=pair_name, **stats))

    df = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "reconstruction_violation_by_day.csv", index=False)
    print(f"Wrote reconstruction_violation_by_day.csv ({len(df)} rows)")
    print(df[["day", "pct_optimistic_gt_mostlikely", "pct_pessimistic_lt_mostlikely",
              "pct_pessimistic_lt_optimistic"]].to_string(index=False))

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(TABLES_DIR / "reconstruction_disagreement_by_day.csv", index=False)
    print(f"\nWrote reconstruction_disagreement_by_day.csv ({len(pair_df)} rows)")

    summary = pair_df.groupby("method_pair")[["mean_abs_diff_min", "max_abs_diff_min",
                                                "pct_pairs_diff_gt_1min"]].agg(["mean", "max"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(TABLES_DIR / "reconstruction_disagreement_summary.csv", index=False)
    print("\nAcross-day summary of pairwise reconstruction-method disagreement:")
    print(summary.to_string(index=False))

    print(f"\nOverall O>M violation range across days: "
          f"{df.pct_optimistic_gt_mostlikely.min():.2f}% to {df.pct_optimistic_gt_mostlikely.max():.2f}%")
    print(f"Overall P<M violation range across days: "
          f"{df.pct_pessimistic_lt_mostlikely.min():.2f}% to {df.pct_pessimistic_lt_mostlikely.max():.2f}%")
    print(f"Overall P<O violation range across days: "
          f"{df.pct_pessimistic_lt_optimistic.min():.2f}% to {df.pct_pessimistic_lt_optimistic.max():.2f}%")


if __name__ == "__main__":
    main()
