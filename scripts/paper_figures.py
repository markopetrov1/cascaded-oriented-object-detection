#!/usr/bin/env python3
"""Generate the three remaining paper figures:

  1. Master Pareto (3 subplots, ships/planes/small_vehicle, all gates × calibrations + oracle)
  2. Stratified per-bucket Pareto (ships, 4 subplots: size/GSD/imagesource/boundary)
  3. Reliability diagrams compact (ships, 6 backbones, 5 calibrations: ECE bar chart + best
     reliability curve)

All figures saved to ``reports/figures/`` as both PNG and PDF.
"""
from __future__ import annotations
import json, glob
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_clean(glob_pat: str) -> list[dict]:
    """Load cascade JSONs, dedup duplicates, exclude distill/codesign/stratif/smoke."""
    rows = []
    seen = set()
    for f in glob.glob(glob_pat):
        if any(s in f for s in ("stratif", "codesign", "distill", "smoke")):
            continue
        for r in json.load(open(f)):
            lbl = r.get("label", "")
            calib = r.get("calibration", "")
            if calib and lbl.endswith("_" + calib):
                continue
            key = (lbl, calib, round(r["threshold"], 3))
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    return rows


def _plot_pareto_panel(ax, cls_name: str, glob_pat: str, fp_label: str = "YOLO11m full"):
    rows = _load_clean(glob_pat)
    real = [r for r in rows if r.get("gate") != "oracle"]
    oracle = [r for r in rows if r.get("gate") == "oracle"]
    fp = next((r for r in real if r["threshold"] == 0.0), None)

    # Group by (label, calibration) and plot a curve per group
    by_key = defaultdict(list)
    for r in real:
        by_key[(r.get("label", "?"), r.get("calibration", ""))].append(r)

    # Pick a backbone-keyed colour map
    backbones = ["resnet18", "resnet50", "mbv3small", "mbv3large", "effb0", "tiny"]
    bb_colors = dict(zip(backbones, plt.cm.tab10(np.linspace(0, 1, len(backbones)))))

    def backbone_of(label: str) -> str:
        for bb in backbones:
            if bb in label:
                return bb
        return "?"

    plotted_bbs = set()
    for (lbl, calib), pts in by_key.items():
        bb = backbone_of(lbl)
        if bb == "?":
            continue
        # Only show identity calibration to keep the figure readable; calibration
        # ablation is in a separate panel.
        if calib != "identity":
            continue
        pts = sorted(pts, key=lambda r: r["total_gflops"])
        xs = [r["total_gflops"] / 1e3 for r in pts]
        ys = [r["mAP@0.50"] for r in pts]
        ax.plot(xs, ys, "-o", color=bb_colors[bb], label=bb if bb not in plotted_bbs else None,
                markersize=3, linewidth=1.0, alpha=0.8)
        plotted_bbs.add(bb)

    if oracle:
        oxs = [r["total_gflops"] / 1e3 for r in sorted(oracle, key=lambda r: r["total_gflops"])]
        oys = [r["mAP@0.50"] for r in sorted(oracle, key=lambda r: r["total_gflops"])]
        ax.plot(oxs, oys, "k--", label="oracle", linewidth=1.5, alpha=0.7)
    if fp:
        ax.scatter([fp["total_gflops"] / 1e3], [fp["mAP@0.50"]], marker="*", s=180, color="red",
                   zorder=10, label=fp_label)

    ax.set_xlabel("Total GFLOPs (×10³)")
    ax.set_ylabel("mAP$@0.5$")
    ax.set_title(cls_name)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=7, ncol=2)


def make_master_pareto():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, (cls, gp) in zip(axes, [
        ("DOTA-1.5 ships",         "reports/cascade/*.json"),
        ("DOTA-1.5 planes",        "reports/planes/cascade/*.json"),
        ("DOTA-1.5 small-vehicle", "reports/small_vehicle/cascade/*.json"),
    ]):
        _plot_pareto_panel(ax, cls, gp)
    fig.suptitle("Cascade Pareto frontier (identity calibration; gate × calibration sweeps in supplementary)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = Path("reports/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "master_pareto.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / "master_pareto.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}/master_pareto.{{png,pdf}}")


def _load_strat(glob_pat: str) -> list[dict]:
    rows = []
    for f in glob.glob(glob_pat):
        rows.extend(json.load(open(f)))
    return [r for r in rows if "mAP@0.50" in r]


def make_stratified():
    """4 subplots, ships only: per-bucket Pareto for size/GSD/imagesource/boundary."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    strata_cfg = [
        ("size_bucket",   "Object size",        "reports/cascade/*.stratified_size_bucket.json"),
        ("gsd_bucket",    "Ground sampling distance", "reports/cascade/*.stratified_gsd_bucket.json"),
        ("imagesource",   "Sensor source",      "reports/cascade/*.stratified_imagesource.json"),
        ("boundary",      "Tile-boundary",      "reports/cascade/*.stratified_boundary.json"),
    ]
    for ax, (stratum, title, gp) in zip(axes, strata_cfg):
        rows = _load_strat(gp)
        # Aggregate per-bucket curves
        by_bucket = defaultdict(list)
        for r in rows:
            by_bucket[r["bucket"]].append(r)
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(by_bucket), 4)))
        for i, (bk, rs) in enumerate(sorted(by_bucket.items())):
            rs = sorted([r for r in rs if "filter_rate" in r], key=lambda r: r["filter_rate"])
            if len(rs) < 2:
                continue
            xs = [r["filter_rate"] for r in rs]
            ys = [r["mAP@0.50"] for r in rs]
            ax.plot(xs, ys, "-o", color=cmap[i % len(cmap)], label=bk, markersize=3, linewidth=1.0, alpha=0.85)
        ax.set_xlabel("Filter rate (fraction of tiles dropped)")
        ax.set_ylabel("mAP$@0.5$")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left", ncol=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle("Stratified Pareto curves (DOTA-1.5 ships, ResNet-18 + temperature)", fontsize=12)
    fig.tight_layout()
    out = Path("reports/figures")
    fig.savefig(out / "stratified_ships.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / "stratified_ships.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}/stratified_ships.{{png,pdf}}")


def make_reliability():
    """Compact reliability summary: ECE bar chart per backbone × calibration on ships,
    plus one example reliability curve panel for the best gate (mbv3large)."""
    # Aggregate ECEs per (backbone, method) on ships
    rel = {}
    for f in sorted(glob.glob("reports/calibration/ships/gate_*/reliability_summary.json")):
        bb = Path(f).parent.name.replace("gate_", "")
        rel[bb] = json.load(open(f))

    methods = ["identity", "temperature", "platt", "isotonic", "context_adaptive"]
    backbones = sorted(rel.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Bar chart: ECE per backbone × method
    width = 0.16
    x = np.arange(len(backbones))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    for i, m in enumerate(methods):
        vals = [rel[bb].get(m, {}).get("ece", float("nan")) for bb in backbones]
        ax1.bar(x + (i - 2) * width, vals, width, color=cmap[i], label=m)
    ax1.set_xticks(x)
    ax1.set_xticklabels(backbones, rotation=15, fontsize=9)
    ax1.set_ylabel("ECE")
    ax1.set_title("Expected calibration error per (backbone × method) on ships val")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3, axis="y")
    ax1.set_yscale("log")

    # Best-gate reliability curves (mbv3large): plot identity vs platt on the same axes
    from src.cascade import load_tile_scores  # type: ignore
    from src.calibration import build_calibration  # type: ignore

    scores = load_tile_scores("reports/gate_scores/gate_mbv3large_val.jsonl")
    raw = np.array([t.prob for t in scores])
    labels = np.array([t.label for t in scores], dtype=np.int64)

    def reliability_curve(probs, labels, n_bins=15):
        bins = np.linspace(0, 1, n_bins + 1)
        accs, confs, counts = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins, dtype=int)
        for i in range(n_bins):
            m = (probs >= bins[i]) & ((probs <= bins[i+1]) if i == n_bins - 1 else (probs < bins[i+1]))
            n = int(m.sum())
            counts[i] = n
            if n > 0:
                accs[i] = labels[m].mean()
                confs[i] = probs[m].mean()
        return confs[counts > 0], accs[counts > 0]

    ax2.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    for m, color in zip(["identity", "temperature", "platt", "isotonic"], ["C0", "C1", "C2", "C3"]):
        if m == "identity":
            cal_probs = raw
        else:
            cal = build_calibration(m).fit(raw, labels)
            cal_probs = cal.transform(raw)
        cx, cy = reliability_curve(cal_probs, labels)
        ax2.plot(cx, cy, "-o", label=m, color=color, markersize=4)
    ax2.set_xlabel("confidence")
    ax2.set_ylabel("accuracy")
    ax2.set_title("Reliability curves — ships, MobileNetV3-large gate")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    fig.tight_layout()
    out = Path("reports/figures")
    fig.savefig(out / "reliability_compact.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / "reliability_compact.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}/reliability_compact.{{png,pdf}}")


def main() -> int:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    print("(1) master_pareto ...")
    make_master_pareto()
    print("(2) stratified_ships ...")
    make_stratified()
    print("(3) reliability_compact ...")
    make_reliability()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
