"""Shared matplotlib configuration so every figure in results/figures/ reads as one
consistent, publication-grade system: same fonts, same method-color mapping across every
figure, same DPI/format conventions. Import `apply_style()` once at the top of any
plotting script; use METHOD_COLORS / METHOD_LABELS for anything that plots per-method
series so the same method always gets the same color everywhere.
"""

import matplotlib.pyplot as plt

# Colorblind-safe categorical palette (Okabe & Ito, 2008). Fixed per-method assignment
# so a color always means the same method across every figure in the paper.
METHOD_COLORS = {
    "B0_deterministic": "#999999",
    "B1_global_robust": "#0072B2",
}

METHOD_LABELS = {
    "B0_deterministic": "Deterministic (no margin)",
    "B1_global_robust": "Global margin",
}

METHOD_ORDER = ("B0_deterministic", "B1_global_robust")

FONT_STACK = ["Arial", "DejaVu Sans", "sans-serif"]  # Helvetica isn't shipped on Windows; skip it to avoid findfont warnings

SINGLE_COL_IN = (3.6, 2.9)
DOUBLE_COL_IN = (7.4, 3.4)
SQUARE_IN = (4.6, 4.2)

PNG_DPI = 300


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": FONT_STACK,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": PNG_DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,   # embed as real text, not paths, in vector PDFs
        "ps.fonttype": 42,
    })


def save_fig(fig, path_stem) -> None:
    """Save both PNG (raster, presentation/quick-view) and PDF (vector, submission-grade)."""
    fig.savefig(f"{path_stem}.png")
    fig.savefig(f"{path_stem}.pdf")
