#!/usr/bin/env python3
"""Sparsity-ceiling empirical law: max cascade savings ≈ 1 − positive_rate.

Plots `positive_rate` (x) vs the BEST cascade-Pareto compute-saved%
at the ≥97% mAP@0.5 threshold (y) across the 4 datasets we have:
DOTA-ships / DOTA-planes / DOTA-small_vehicle / HRSC2016. Overlays the
y = 100·(1 − x) line (the theoretical max from oracle filtering).
"""
from __future__ import annotations
import json, glob
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def cascade_best_savings(glob_pat: str) -> tuple[float, float, float]:
    """Return (positive_rate, best_savings_%, full_pass_mAP). Excludes distill+codesign."""
    rows = []
    seen = set()
    for f in glob.glob(glob_pat):
        if "stratif" in f or "codesign" in f or "distill" in f or "smoke" in f:
            continue
        # Skip OLD stratified-step main outputs whose label ends with the calibration
        # name (these are duplicates of the proper per-calibration JSONs from Step 3).
        for r in json.load(open(f)):
            lbl = r.get('label', '')
            calib = r.get('calibration', '')
            if calib and lbl.endswith('_' + calib):
                continue
            # Dedupe by (label, calibration, threshold) — same row can be in multiple files
            key = (lbl, calib, round(r['threshold'], 3))
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    real = [r for r in rows if r.get('gate') != 'oracle']
    fp = next((r for r in real if r['threshold'] == 0.0), None)
    if not fp:
        return None, None, None
    threshold = fp['mAP@0.50'] * 0.97
    cand = [r for r in real if r['mAP@0.50'] >= threshold]
    # Pick best by max compute-saved (i.e. min total_gflops), not max filter_rate —
    # since gates have very different compute costs (mbv3small=0.14 G vs resnet50=10.79 G).
    best = min(cand, key=lambda r: r['total_gflops'])
    saved = 100.0 * (1 - best['total_gflops'] / fp['total_gflops'])
    # Positive rate ≈ 1 − filter_rate at the oracle's natural operating point
    oracle_rows = [r for r in rows if r.get('gate') == 'oracle']
    o_best = max(oracle_rows, key=lambda r: r['mAP@0.50']) if oracle_rows else None
    pos_rate = 1.0 - o_best['filter_rate'] if o_best else None
    return pos_rate, saved, fp['mAP@0.50']


def main() -> int:
    datasets = [
        ("DOTA-ships",         "reports/cascade/*.json"),
        ("DOTA-planes",        "reports/planes/cascade/*.json"),
        ("DOTA-small_vehicle", "reports/small_vehicle/cascade/*.json"),
        ("HRSC2016",           "reports/hrsc/*.json"),
    ]
    points = []
    for name, pat in datasets:
        pr, save, fp_map = cascade_best_savings(pat)
        if pr is None:
            continue
        points.append((name, pr, save, fp_map))
        print(f"  {name:<22} positive_rate={pr:.3f}  saved={save:.1f}%  full_pass_mAP@0.5={fp_map:.3f}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    xs = np.linspace(0, 1, 100)
    ax.plot(xs * 100, (1 - xs) * 100, "k--", alpha=0.4, label="oracle ceiling y = 100(1 − x)")
    for name, pr, save, _ in points:
        ax.scatter(pr * 100, save, s=120, zorder=5)
        ax.annotate(name, (pr * 100, save), xytext=(7, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Positive-tile rate (%)")
    ax.set_ylabel("Best cascade compute saved at ≥97% full-pass mAP@0.5 (%)")
    ax.set_title("Sparsity-ceiling empirical law")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    out = Path("reports/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "sparsity_ceiling.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / "sparsity_ceiling.pdf", bbox_inches="tight")
    plt.close(fig)
    # JSON dump for the paper
    Path(out).joinpath("sparsity_ceiling.json").write_text(json.dumps(
        [{"dataset": n, "positive_rate": pr, "savings_pct": s, "full_pass_mAP": m} for n, pr, s, m in points],
        indent=2,
    ))
    print(f"\n  Wrote {out}/sparsity_ceiling.{{png,pdf,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
