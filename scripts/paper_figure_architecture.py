#!/usr/bin/env python3
"""Pipeline schematic: where the compute goes, and what the gate can remove.

The draft had no architecture figure at all. This one is drawn to carry the
paper's argument rather than merely depict the method: every box is annotated
with its measured cost, and the accounting identity is stated underneath, so a
reader meets the cost model in the same place they meet the pipeline.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np

from figstyle import INK, INK_MUTED, PALETTE, TEXT_WIDTH, save, use_paper_style

import matplotlib.pyplot as plt

GATE_G, DET_G = 0.61, 184.17          # reports/speed/*_b1.json
GATE_MS, DET_MS = 5.96, 21.55         # ms/img at batch 1, same source


def box(ax, x, y, w, h, label, sub=None, colour=PALETTE[0], alpha=0.14):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=colour, alpha=alpha, edgecolor=colour, linewidth=1.1, zorder=3))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), label, ha="center",
            va="center", fontsize=7.2, color=INK, zorder=4)
    if sub:
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=5.8, color=INK_MUTED, zorder=4)


def arrow(ax, x0, y0, x1, y1, label=None, colour=INK_MUTED, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, lw=1.0, color=colour,
                                shrinkA=1, shrinkB=1))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.03, label, ha="center",
                va="bottom", fontsize=5.8, color=INK_MUTED)


def main() -> int:
    use_paper_style()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.35))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # --- scene and tiling -------------------------------------------------
    box(ax, 0.005, 0.50, 0.13, 0.34, "RS scene", "$10^4\\times10^4$ px",
        colour=INK_MUTED, alpha=0.10)
    ax.text(0.07, 0.40, "tile $1024^2$\noverlap 200", ha="center", va="top",
            fontsize=5.8, color=INK_MUTED)

    # a 4x3 grid of tiles, most of them empty
    gx0, gy0, cw, ch = 0.175, 0.50, 0.032, 0.113
    rng = np.random.default_rng(3)
    occupied = {(1, 1), (2, 1), (2, 2)}
    for r in range(3):
        for c in range(4):
            filled = (c, r) in occupied
            ax.add_patch(mpatches.Rectangle(
                (gx0 + c * cw, gy0 + r * ch), cw * 0.9, ch * 0.9,
                facecolor=PALETTE[0] if filled else "white",
                alpha=0.55 if filled else 1.0,
                edgecolor=INK_MUTED, linewidth=0.5, zorder=3))
    ax.text(gx0 + 2 * cw, 0.40, f"$p_+$ of tiles\ncontain a target",
            ha="center", va="top", fontsize=5.8, color=INK_MUTED)

    # --- gate -------------------------------------------------------------
    arrow(ax, 0.312, 0.67, 0.352, 0.67)
    box(ax, 0.355, 0.52, 0.155, 0.30, "gate $g_\\theta$",
        f"MobileNetV3 @256\n{GATE_G} GFLOPs · {GATE_MS} ms", colour=PALETTE[2])
    ax.text(0.4325, 0.44, "runs on every tile", ha="center", va="top",
            fontsize=5.8, color=INK_MUTED)

    # --- branch -----------------------------------------------------------
    arrow(ax, 0.512, 0.67, 0.556, 0.67, "$c(g_\\theta(x))\\geq\\tau$")
    box(ax, 0.60, 0.60, 0.175, 0.245, "OBB detector $f_\\phi$",
        f"YOLO11m @1024\n{DET_G} GFLOPs · {DET_MS} ms", colour=PALETTE[0])
    # reject path
    ax.annotate("", xy=(0.60, 0.30), xytext=(0.534, 0.62),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=PALETTE[1],
                                connectionstyle="arc3,rad=-0.25"))
    box(ax, 0.60, 0.20, 0.175, 0.16, "discarded", None, colour=PALETTE[1])
    ax.text(0.5, 0.35, "reject", fontsize=5.8, color=PALETTE[1], ha="center")

    arrow(ax, 0.777, 0.72, 0.817, 0.72)
    box(ax, 0.820, 0.60, 0.155, 0.245, "detections", None,
        colour=INK_MUTED, alpha=0.10)

    # --- the identity, stated where the reader meets the pipeline ---------
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.355, 0.015), 0.62, 0.145,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor="#f4f2ec", edgecolor="#ddd9cf", linewidth=0.7, zorder=2))
    ax.text(0.665, 0.125,
            r"$C = N\,G_g + N\,a(\tau)\,G_f$,    "
            r"$a(\tau)=p_+\mathrm{TPR}(\tau)+(1-p_+)\mathrm{FPR}(\tau)$",
            ha="center", va="center", fontsize=6.6, color=INK, zorder=4)
    # matplotlib's mathtext has no \underbrace, so the three terms are named
    # in a colour key that matches the provenance figure's segments instead.
    ax.text(0.372, 0.052, r"saved $=$", ha="left", va="center",
            fontsize=6.6, color=INK, zorder=4)
    terms = [
        (r"$(1-p_+)(1-\mathrm{FPR})$", "empty tiles skipped", PALETTE[2]),
        (r"$+\;p_+(1-\mathrm{TPR})$", "detections dropped", PALETTE[1]),
        (r"$-\;G_g/G_f$", "gate overhead", PALETTE[3]),
    ]
    x = 0.445
    for expr, name, colour in terms:
        ax.text(x, 0.068, expr, ha="left", va="center", fontsize=6.6,
                color=colour, zorder=4)
        ax.text(x, 0.026, name, ha="left", va="center", fontsize=5.4,
                color=INK_MUTED, zorder=4)
        x += 0.185

    save(fig, "architecture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
