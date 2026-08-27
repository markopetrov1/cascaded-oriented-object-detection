"""Shared figure style for the paper.

One place to define column geometry and the categorical palette, so every
figure in reports/figures/ reads as one system.

The palette is the Okabe-Ito colour-vision-safe set, subset and ordered so that
*every* pair -- not merely adjacent pairs -- clears dE >= 15 in normal vision
and dE >= 8 under simulated protanopia and deuteranopia. Verified with

    python3 scripts/check_palette.py "#0072B2,#E69F00,#009E73,#D55E00,#56B4E9,#000000"

which reports zero hard failures. Four pairs sit below the greyscale-separation
threshold, so colour never carries identity alone: bars hatch, lines take
distinct dash patterns and markers, and series are direct-labelled where there
is room.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEEtran geometry, in inches.
COL_WIDTH = 3.5
TEXT_WIDTH = 7.16

# Fixed categorical order. Never cycled, never reassigned by rank.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#000000"]

# Secondary encodings, index-matched to PALETTE.
HATCHES = ["", "///", "...", "\\\\\\", "xxx", "+++"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
DASHES = [(None, None), (4, 1.5), (1, 1.2), (5, 1.2, 1, 1.2), (3, 1, 1, 1), (2, 2)]

# Ink, never a series colour.
INK = "#141413"
INK_MUTED = "#6b6b66"
GRID = "#d8d6cf"


def use_paper_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.4,
        "patch.linewidth": 0.5,
        "legend.frameon": False,
    })


def save(fig, name: str, outdir: str = "reports/figures") -> None:
    """Write a figure as both PDF (for LaTeX) and PNG (for quick review)."""
    from pathlib import Path

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {out}/{name}.{{pdf,png}}")
