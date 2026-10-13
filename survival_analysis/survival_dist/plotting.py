"""Shared matplotlib style: fixed categorical slots, recessive axes, light surface.

Slot order follows the dataviz reference palette (validated for CVD on adjacent
pairs; the first three slots validate all-pairs, so scatter-type figures use at
most three series). Identity is never color-alone: every series also has its own
marker and a legend entry.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

SERIES = {
    "gt": {"color": "#2a78d6", "marker": "o", "label": "GT (Epileptor)"},
    "rc": {"color": "#eb6834", "marker": "s", "label": "SCDT-RC"},
    "oracle": {"color": "#1baf7a", "marker": "^", "label": "GT, independent seeds"},
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": TEXT_SECONDARY,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.5,
            "font.size": 10,
        }
    )
