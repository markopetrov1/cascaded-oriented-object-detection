#!/usr/bin/env python3
"""Sparsity ceiling: achievable cascade savings against the positive-tile rate.

Consumes the verified decomposition from scripts/savings_model.py rather than
re-deriving savings from the report globs, which fixes three defects in the
original version of this script:

  * HRSC2016 was silently dropped. The positive rate was inferred from an oracle
    run's filter rate, and reports/hrsc/ has no oracle rows, so the figure and
    its JSON carried three points while the paper's text and caption claimed
    four. p+ now comes from the tiling metadata and is always defined.

  * The full-pass baseline was whichever tau = 0 row `glob.glob` happened to
    return first, so a cascade running a 0.56 GFLOPs MobileNetV3 gate could be
    scored against a baseline charged 10.79 GFLOPs for a ResNet-50 gate it never
    runs. That inflated DOTA-planes by 0.9 pp and made the number depend on
    filesystem ordering. Savings are now measured against the detector-only cost
    N * G_f, identical for every row in a domain.

  * The tolerance was 97 % of full-pass mAP while the manuscript states 3 pp.
    Both are computed; which one the paper reports is now an explicit choice.

Run scripts/savings_model.py first.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUMMARY = Path("reports/figures/savings_summary.json")
# Which mAP-tolerance rule the figure reports. "absolute" is the 3 pp budget the
# manuscript describes; "relative" is the 97 % rule that produced the published
# numbers.
TOLERANCE_RULE = "absolute"
# The YOLO11n arm is the same tile population as DOTA-ships with a different
# detector, so it would plot as a duplicate x-position; it is reported in the
# orthogonality section instead.
EXCLUDE = {"DOTA-ships/YOLO11n", "OAN-joint/ships", "Independent/ships-matched"}


def main() -> int:
    if not SUMMARY.exists():
        print(f"  missing {SUMMARY} -- run scripts/savings_model.py first")
        return 1
    summary = json.load(open(SUMMARY))

    points = []
    for dom in summary:
        if dom["domain"] in EXCLUDE:
            continue
        best = dom["by_tolerance_rule"].get(TOLERANCE_RULE)
        if not best:
            continue
        points.append({
            "dataset": dom["domain"],
            "positive_rate": dom["p_plus"],
            "savings_pct": 100.0 * best["saved_detector_only"],
            "ceiling_pct": 100.0 * (1.0 - dom["p_plus"]),
            "full_pass_mAP": dom["full_pass_mAP@0.50"],
            "gate": best["gate"],
            "calibration": best["calibration"],
            "tpr": best["tpr"],
            "fpr": best["fpr"],
            "n_tiles": dom["n_tiles"],
            "tolerance_rule": TOLERANCE_RULE,
        })
    points.sort(key=lambda p: p["positive_rate"])

    for p in points:
        print(f"  {p['dataset']:<22} p+={p['positive_rate']:.4f}  saved={p['savings_pct']:.1f}%"
              f"  (ceiling {100*(1-p['positive_rate']):.1f}%)  TPR={p['tpr']:.3f} FPR={p['fpr']:.3f}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    xs = np.linspace(0, 1, 200)
    ax.plot(xs * 100, (1 - xs) * 100, "k--", alpha=0.5,
            label=r"perfect-gate ceiling $1-p_+$")
    ax.fill_between(xs * 100, (1 - xs) * 100, 100, color="0.9", zorder=0)
    ax.text(52, 92, "unreachable", fontsize=9, color="0.45", style="italic")

    px = [p["positive_rate"] * 100 for p in points]
    py = [p["savings_pct"] for p in points]
    ax.scatter(px, py, s=130, zorder=5, edgecolor="k", linewidth=0.6)
    for p in points:
        ax.annotate(p["dataset"], (p["positive_rate"] * 100, p["savings_pct"]),
                    xytext=(8, -4), textcoords="offset points", fontsize=9)

    ax.set_xlabel(r"Positive-tile rate $p_+$ (%)")
    ax.set_ylabel("Best cascade compute saved (%)")
    ax.set_title("The sparsity ceiling")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")

    out = Path("reports/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "sparsity_ceiling.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / "sparsity_ceiling.pdf", bbox_inches="tight")
    plt.close(fig)
    (out / "sparsity_ceiling.json").write_text(json.dumps(points, indent=2))
    print(f"\n  {len(points)} points -> {out}/sparsity_ceiling.{{png,pdf,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
