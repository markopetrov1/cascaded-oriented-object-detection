#!/usr/bin/env python3
"""Figures for the savings law.

  savings_provenance  -- where the saved compute actually comes from, per domain
  sparsity_ceiling_v2 -- savings against p+, and the recall-vs-removal curve that
                         shows the ceiling as a knee, with OAN's published
                         ablation overlaid as external validation

Run scripts/savings_model.py and scripts/prior_work_prediction.py first.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from figstyle import HATCHES, INK, INK_MUTED, MARKERS, PALETTE, TEXT_WIDTH, save, use_paper_style

import matplotlib.pyplot as plt  # figstyle has already pinned the Agg backend

DECOMP = Path("reports/figures/savings_decomposition.json")
SUMMARY = Path("reports/figures/savings_summary.json")
PRIOR = Path("reports/figures/prior_work_prediction.json")

# Plot order: sparse to dense. The YOLO11n arm shares DOTA-ships' tile
# population, so it would sit at a duplicate x-position and is left out.
ORDER = ["DOTA-planes", "DOTA-ships", "DOTA-small_vehicle", "HRSC2016"]
SHORT = {
    "DOTA-planes": "planes",
    "DOTA-ships": "ships",
    "DOTA-small_vehicle": "small-veh.",
    "HRSC2016": "HRSC2016",
}


def _summary_by_domain() -> dict:
    return {d["domain"]: d for d in json.load(open(SUMMARY))}


def figure_provenance() -> None:
    """Stacked budget split, normalised to detector-only compute."""
    summary = _summary_by_domain()

    labels, spent_tp, leak, dropped, skipped = [], [], [], [], []
    for dom in ORDER:
        d = summary[dom]
        best = d["by_tolerance_rule"]["absolute"]
        p, tpr, fpr = d["p_plus"], best["tpr"], best["fpr"]
        labels.append(f"{SHORT[dom]}\n$p_+$={p:.2f}")
        spent_tp.append(p * tpr)
        leak.append((1 - p) * fpr)
        dropped.append(p * (1 - tpr))
        skipped.append((1 - p) * (1 - fpr))

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.62, 2.9))
    y = np.arange(len(labels))
    h = 0.62
    gap = 0.004  # 2px-equivalent surface gap between stacked segments

    segments = [
        (spent_tp, "detector runs, tile has objects", PALETTE[0], HATCHES[0]),
        (leak, "detector runs, tile is empty (leak)", PALETTE[3], HATCHES[3]),
        (dropped, "skipped, tile had objects (recall lost)", PALETTE[1], HATCHES[1]),
        (skipped, "skipped, tile was empty (true saving)", PALETTE[2], HATCHES[2]),
    ]
    left = np.zeros(len(labels))
    for values, label, colour, hatch in segments:
        values = np.array(values)
        ax.barh(y, values - gap, left=left, height=h, color=colour, hatch=hatch,
                edgecolor="white", linewidth=0.6, label=label, zorder=3)
        left = left + values

    # Mark the boundary between compute spent and compute saved.
    boundary = np.array(spent_tp) + np.array(leak)
    for yi, b in zip(y, boundary):
        ax.plot([b, b], [yi - h / 2, yi + h / 2], color=INK, lw=1.6, zorder=5)

    # How much of the saving is really recall sacrifice. Set outside the plot
    # area so it never sits on a fill, with the dense regime called out in bold.
    for yi, drop, skip in zip(y, dropped, skipped):
        total = drop + skip
        if total <= 0.02:
            continue
        share = drop / total
        loud = share > 0.5
        ax.annotate(f"{share:.0%}", xy=(1.02, yi),
                    xycoords=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=7.5, color=INK if loud else INK_MUTED,
                    weight="bold" if loud else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of the full-pass detector budget")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Where the saved compute comes from", loc="left", pad=10)
    ax.annotate("share of the\nsaving that is\ndropped detections",
                xy=(1.02, -0.85), xycoords=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6, color=INK_MUTED)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.24), ncol=2)
    save(fig, "savings_provenance")


def figure_ceiling_v2() -> None:
    """Savings against p+, and the recall-vs-removal knee that explains it."""
    rows = json.load(open(DECOMP))
    summary = _summary_by_domain()
    prior = json.load(open(PRIOR)) if PRIOR.exists() else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.9))

    # ---- panel (a): the ceiling ------------------------------------------
    xs = np.linspace(0, 1, 200)
    ax1.plot(xs * 100, (1 - xs) * 100, "--", color=INK, lw=1.2, zorder=4,
             label=r"lossless ceiling $1-p_+$")
    ax1.fill_between(xs * 100, (1 - xs) * 100, 100, color="0.955", zorder=0)

    by_domain = defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append(r)

    # Three families, so the reader can see that the envelope holds across a
    # 16-class detector, per-class detectors, and a zero-shot transfer alike.
    def family(name: str) -> str:
        if name.startswith("sweep/"):
            return "sweep"
        if name.startswith("HRSC"):
            return "hrsc"
        return "perclass"

    STYLE = {
        "sweep": (PALETTE[0], MARKERS[0], "16-class detector, one gate per class"),
        "perclass": (PALETTE[1], MARKERS[1], "per-class detector (original runs)"),
        "hrsc": (PALETTE[3], MARKERS[3], "HRSC2016 zero-shot"),
    }
    seen_fam = set()
    plotted = 0
    for dom in summary.values():
        if dom["domain"] in {"DOTA-ships/YOLO11n", "OAN-joint/ships",
                             "Independent/ships-matched"}:
            continue  # same tile population as DOTA-ships; reported in the text
        best = dom["by_tolerance_rule"]["absolute"]
        if not best:
            continue
        fam = family(dom["domain"])
        colour, marker, lab = STYLE[fam]
        ax1.scatter(dom["p_plus"] * 100, 100 * best["saved_detector_only"],
                    s=44, color=colour, marker=marker, edgecolor="white",
                    linewidth=0.7, zorder=6,
                    label=lab if fam not in seen_fam else None)
        seen_fam.add(fam)
        plotted += 1

    # The two points that carry the argument, plus the failure mode.
    notes = {
        "sweep/any": ("gating all 16 classes\nat once (OAN's regime)", (-34, 26)),
        "HRSC2016": ("HRSC2016\nzero-shot", (-16, 16)),
    }
    for dom in summary.values():
        if dom["domain"] in notes:
            best = dom["by_tolerance_rule"]["absolute"]
            txt, off = notes[dom["domain"]]
            ax1.annotate(txt, (dom["p_plus"] * 100, 100 * best["saved_detector_only"]),
                         xytext=off, textcoords="offset points", fontsize=5.8,
                         color=INK_MUTED, ha="center",
                         arrowprops=dict(arrowstyle="-", lw=0.5, color=INK_MUTED))
    zero = [d for d in summary.values()
            if d["by_tolerance_rule"]["absolute"]
            and d["by_tolerance_rule"]["absolute"]["saved_detector_only"] < 0.02]
    if zero:
        ax1.annotate(f"{len(zero)} rare classes:\ngate too weak to filter\nwithin tolerance",
                     xy=(4.5, 0.0), xytext=(16, 10), textcoords="data",
                     fontsize=5.6, color=INK_MUTED, ha="left", va="bottom",
                     arrowprops=dict(arrowstyle="-", lw=0.5, color=INK_MUTED,
                                     connectionstyle="arc3,rad=-0.2"))

    ax1.set_xlabel(r"Positive-tile rate $p_+$ (%)")
    ax1.set_ylabel("Compute saved (%)")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-4, 100)
    ax1.set_title(f"(a) {plotted} gating tasks, one envelope", loc="left")
    # The region above the ceiling is unreachable, so the legend costs nothing there.
    ax1.legend(loc="upper right", fontsize=5.4, labelspacing=0.35,
               borderpad=0.3, handletextpad=0.4)

    # ---- panel (b): recall against removal, with OAN overlaid -------------
    for i, dom in enumerate(ORDER):
        best = summary[dom]["by_tolerance_rule"]["absolute"]
        curve = [r for r in by_domain[dom]
                 if r["label"] == best["gate"] and r["calibration"] == best["calibration"]]
        curve.sort(key=lambda r: 1 - r["accept_rate"])
        removed = [100 * (1 - r["accept_rate"]) for r in curve]
        recall = [100 * r["tpr"] for r in curve]
        ax2.plot(removed, recall, color=PALETTE[i], marker=MARKERS[i], ms=3,
                 lw=1.3, label=SHORT[dom], zorder=3)
        ax2.axvline(100 * (1 - summary[dom]["p_plus"]), color=PALETTE[i],
                    lw=0.8, ls=":", alpha=0.75, zorder=1)

    if prior:
        t9 = prior["oan"]["table9_removal_curve"]
        ax2.plot([100 * r["removed"] for r in t9], [100 * r["recall"] for r in t9],
                 color=PALETTE[5], marker=MARKERS[5], ms=3.5, lw=1.6, ls="--",
                 label="OAN (published)", zorder=5)
        lo, hi = prior["oan"]["predicted_lossless_removal_band"]
        ax2.axvspan(100 * lo, 100 * hi, color=PALETTE[5], alpha=0.10, zorder=0)
        ax2.annotate("OAN's\nempty-patch\nrate", xy=(100 * (lo + hi) / 2, 30),
                     ha="center", va="center", fontsize=5.6, color=INK_MUTED,
                     rotation=90)

    ax2.set_xlabel("Tiles removed by the gate (%)")
    ax2.set_ylabel("Gate recall on positive tiles (%)")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 103)
    ax2.set_title("(b) Recall collapses past the ceiling", loc="left")
    # Dotted verticals mark each domain's 1 - p+; stated in the caption rather
    # than as in-plot text, which collided with the curves.
    ax2.legend(loc="lower left", fontsize=6)

    save(fig, "sparsity_ceiling_v2")


def main() -> int:
    for path in (DECOMP, SUMMARY):
        if not path.exists():
            print(f"  missing {path} -- run scripts/savings_model.py first")
            return 1
    use_paper_style()
    figure_provenance()
    figure_ceiling_v2()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
