#!/usr/bin/env python3
"""What tile-gating systems report, and what they leave unmeasured.

Restricted to systems that gate whole tiles before detection, which is the
family this paper belongs to. Region-cropping methods (ClusDet, DMNet, GLSAN,
AdaZoom) solve a different problem -- where to look more closely, rather than
whether to look at all -- and are discussed in the text instead of charted
here, because scoring them on these columns would misrepresent them.

Every cell is taken from the cited paper and cross-checked against its tables;
the notes in scripts/prior_work_prediction.py record the supporting quotes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from figstyle import INK, INK_MUTED, PALETTE, TEXT_WIDTH, save, use_paper_style

import matplotlib.pyplot as plt

YES, PARTIAL, NO = 1.0, 0.5, 0.0

COLUMNS = [
    "FLOPs\nreported",
    "Wall-clock\nreported",
    "Operating\npoint swept",
    "Dataset\nsparsity stated",
    "Gate\ncalibration",
    "Cross-dataset\nzero-shot",
    "Savings\ndecomposed",
]

ROWS = [
    ("Plastiras et al.\nICDSC 2018", [NO, YES, YES, PARTIAL, NO, NO, NO]),
    ("R$^2$-CNN\nTGRS 2019", [NO, YES, YES, YES, NO, NO, NO]),
    ("OAN\nSci. China 2023", [NO, NO, YES, YES, NO, NO, NO]),
    ("This work", [YES, YES, YES, YES, YES, YES, YES]),
]


def main() -> int:
    use_paper_style()
    n_rows, n_cols = len(ROWS), len(COLUMNS)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.82, 2.5))

    for r, (_, values) in enumerate(ROWS):
        ours = r == n_rows - 1
        for c, v in enumerate(values):
            colour = PALETTE[0] if ours else INK_MUTED
            if v == YES:
                ax.scatter(c, r, s=110, marker="o", color=colour,
                           edgecolor="white", linewidth=0.8, zorder=3)
            elif v == PARTIAL:
                # Half-filled: reports a skip rate but not the underlying sparsity.
                ax.scatter(c, r, s=110, marker="o", facecolor="white",
                           edgecolor=colour, linewidth=1.2, zorder=3)
                ax.scatter(c, r, s=110, marker=matplotlib_halfmarker(), color=colour,
                           linewidth=0, zorder=4)
            else:
                ax.plot([c - 0.13, c + 0.13], [r, r], color="#c9c7c0", lw=1.6, zorder=3)

    # Band behind our row so the contrast is structural, not just colour.
    ax.axhspan(n_rows - 1.45, n_rows - 0.55, color=PALETTE[0], alpha=0.07, zorder=0)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(COLUMNS, fontsize=6.4)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([r[0] for r in ROWS], fontsize=6.8)
    ax.set_xlim(-0.6, n_cols - 0.4)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.invert_yaxis()
    ax.tick_params(length=0)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="#e8e6df", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("What tile-gating systems measure", loc="left", pad=8)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=INK_MUTED, ms=6, label="reported"),
        plt.Line2D([], [], marker=matplotlib_halfmarker(), ls="", color=INK_MUTED,
                   ms=6, label="partial"),
        plt.Line2D([], [], marker="_", ls="", color="#c9c7c0", ms=8, label="not reported"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, fontsize=6.4, columnspacing=1.6, handletextpad=0.4)
    save(fig, "taxonomy")
    return 0


def matplotlib_halfmarker():
    """A left-half-filled circle, for 'partially reported'."""
    import matplotlib.path as mpath

    theta = np.linspace(np.pi / 2, 3 * np.pi / 2, 40)
    verts = np.column_stack([np.cos(theta), np.sin(theta)])
    verts = np.vstack([verts, verts[0]])
    return mpath.Path(verts)


if __name__ == "__main__":
    raise SystemExit(main())
