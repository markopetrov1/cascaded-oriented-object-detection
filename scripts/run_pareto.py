#!/usr/bin/env python3
"""Aggregate per-cascade evaluations into the master Pareto plot.

Reads the per-config JSON files written by ``scripts/eval_cascade.py`` and
emits:

  - ``reports/pareto.csv`` — all rows from all configs
  - ``reports/figures/pareto.png`` / ``.pdf`` — mAP-vs-GFLOPs plot, one curve
    per (gate, calibration) pair
  - Optional stratified plots when ``--strata`` is supplied

Usage::

    python scripts/run_pareto.py \\
        --inputs reports/cascade/resnet18_temperature.json \\
                 reports/cascade/mbv3small_temperature.json \\
                 reports/cascade/oracle.json \\
                 reports/cascade/yolo11m_baseline.json \\
        --out reports/pareto.csv \\
        --figure reports/figures/pareto
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help="One or more eval_cascade JSON outputs")
    p.add_argument("--out", default="reports/pareto.csv")
    p.add_argument("--figure", default="reports/figures/pareto")
    p.add_argument(
        "--metric",
        default="mAP@0.50",
        help="y-axis metric. Falls back to mAP@0.5 if @0.50 is missing.",
    )
    p.add_argument("--x", default="total_gflops", choices=("total_gflops", "total_latency_ms"))
    return p.parse_args()


def _read_rows(path: Path) -> list[dict]:
    if path.suffix == ".csv":
        with path.open() as handle:
            return list(csv.DictReader(handle))
    with path.open() as handle:
        return json.load(handle)


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _series_key(row: dict) -> str:
    label = row.get("label", "cascade")
    calib = row.get("calibration", "identity")
    return f"{label}|{calib}"


def main() -> int:
    args = parse_args()
    rows: list[dict] = []
    for src in args.inputs:
        for r in _read_rows(Path(src)):
            r["__source__"] = src
            rows.append(r)
    if not rows:
        raise SystemExit("No rows to aggregate")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[run_pareto] wrote {len(rows)} rows to {out_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[run_pareto] matplotlib unavailable, skipping plot: {exc}")
        return 0

    series: dict[str, list[tuple[float, float]]] = {}
    metric_alt = args.metric.replace("@0.50", "@0.5")
    for r in rows:
        y = _to_float(r.get(args.metric, r.get(metric_alt)))
        x = _to_float(r.get(args.x))
        series.setdefault(_series_key(r), []).append((x, y))
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, points in sorted(series.items()):
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", label=key, linewidth=1.4)
    ax.set_xlabel(args.x)
    ax.set_ylabel(args.metric)
    ax.set_title("Cascade Pareto: accuracy vs compute")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig_path = Path(args.figure)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_path.with_suffix(".png"), dpi=200)
    fig.savefig(fig_path.with_suffix(".pdf"))
    print(f"[run_pareto] wrote plots to {fig_path}.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
