"""Shared publication-style plotting defaults."""

import matplotlib.pyplot as plt


SANS_SERIF_STACK = ["Arial", "Liberation Sans", "DejaVu Sans"]
HEATMAP_CMAP = "YlGnBu"
PRIMARY_BLUE = "#2563eb"
PRIMARY_TEAL = "#0f766e"
ACCENT_CYAN = "#0891b2"
SOFT_BLUE = "#93c5fd"
SOFT_TEAL = "#99f6e4"
GRID_COLOR = "#cbd5e1"
TEXT_DARK = "#0f172a"
TEXT_MID = "#475569"


def apply_publication_style():
    """Use Arial-first typography and compact manuscript-friendly defaults."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": SANS_SERIF_STACK,
            "axes.titlesize": 9,
            "axes.labelsize": 7,
            "axes.titleweight": "semibold",
            "axes.labelcolor": TEXT_DARK,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "text.color": TEXT_DARK,
            "axes.edgecolor": "#94a3b8",
            "axes.linewidth": 0.8,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.8,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "figure.titlesize": 10,
            "savefig.bbox": "tight",
            "axes.unicode_minus": False,
        }
    )
