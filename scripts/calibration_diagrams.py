#!/usr/bin/env python3
"""Reliability diagrams + ECE/MCE for the 5 cascade calibrations (Pillar 2 §11).

Reads a gate's val score JSONL (raw probs + binary labels) plus the calibration
fitting logic in ``src.calibration``, applies each calibration to the raw probs,
and plots the calibration curve and ECE/MCE per method. Produces a single
multi-panel PNG/PDF per gate.

Usage::

    python scripts/calibration_diagrams.py \
        --scores reports/gate_scores/gate_resnet18_val.jsonl \
        --metadata-jsonl data/processed/dota_ships/metadata/tiles.jsonl \
        --out-dir reports/calibration/gate_resnet18

Calls the same calibrators as ``eval_cascade.py`` so the diagrams correspond
exactly to the Pareto curves.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.calibration import build_calibration, fit_context_adaptive  # noqa: E402
from src.cascade import load_tile_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--scores", required=True, help="Val score JSONL")
    p.add_argument("--metadata-jsonl", help="Required if including context_adaptive")
    p.add_argument("--out-dir", required=True, help="Where to write PNG/PDF + JSON summary")
    p.add_argument("--n-bins", type=int, default=15)
    p.add_argument("--context-adaptive-bucket", default="imagesource",
                   choices=("imagesource", "gsd_bucket", "size_bucket", "boundary"))
    p.add_argument("--context-adaptive-target-recall", type=float, default=0.95)
    return p.parse_args()


def reliability(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> tuple:
    """Return (bin_centers, bin_accuracies, bin_confidences, bin_counts, ece, mce)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        n = int(mask.sum())
        bin_counts[i] = n
        if n > 0:
            bin_accs[i] = float(labels[mask].mean())
            bin_confs[i] = float(probs[mask].mean())
    weights = bin_counts / max(bin_counts.sum(), 1)
    gaps = np.abs(bin_accs - bin_confs)
    ece = float((weights * gaps).sum())
    mce = float(gaps[bin_counts > 0].max()) if (bin_counts > 0).any() else 0.0
    return centers, bin_accs, bin_confs, bin_counts, ece, mce


def _build_tile_to_bucket(metadata_jsonl: str, stratum: str) -> dict:
    from src.cascade_eval import stratify_tiles_by
    bk2tids = stratify_tiles_by(metadata_jsonl, stratum)
    return {tid: bk for bk, tids in bk2tids.items() for tid in tids}


def _apply(name: str, tile_scores: list, fit_probs: np.ndarray, fit_labels: np.ndarray,
           args: argparse.Namespace) -> np.ndarray:
    if name == "identity":
        return np.array([t.prob for t in tile_scores])
    if name == "context_adaptive":
        if not args.metadata_jsonl:
            raise SystemExit("context_adaptive requires --metadata-jsonl")
        tile_to_bucket = _build_tile_to_bucket(args.metadata_jsonl, args.context_adaptive_bucket)
        bucket_of = lambda tid: tile_to_bucket.get(tid, "unknown")
        ca = fit_context_adaptive(
            tile_ids=[t.tile_id for t in tile_scores],
            probs=fit_probs, labels=fit_labels,
            bucket_of=bucket_of,
            target_recall=args.context_adaptive_target_recall,
        )
        out = np.array([
            float(np.clip(t.prob + (0.5 - ca.threshold_for(bucket_of(t.tile_id))), 0.0, 1.0))
            for t in tile_scores
        ])
        return out
    cal = build_calibration(name).fit(fit_probs, fit_labels)
    return cal.transform(np.array([t.prob for t in tile_scores]))


def main() -> int:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tile_scores = load_tile_scores(args.scores)
    raw_probs = np.array([t.prob for t in tile_scores])
    labels = np.array([t.label for t in tile_scores], dtype=np.int64)

    methods = ["identity", "temperature", "platt", "isotonic", "context_adaptive"]
    summary = {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, m in zip(axes, methods + [None]):
        if m is None:
            # Last panel: bar chart of ECE/MCE
            mlist = list(summary.keys())
            x = np.arange(len(mlist))
            ece_vals = [summary[mm]["ece"] for mm in mlist]
            mce_vals = [summary[mm]["mce"] for mm in mlist]
            ax.bar(x - 0.2, ece_vals, width=0.4, label="ECE")
            ax.bar(x + 0.2, mce_vals, width=0.4, label="MCE")
            ax.set_xticks(x)
            ax.set_xticklabels(mlist, rotation=20, ha="right", fontsize=8)
            ax.set_ylabel("calibration error")
            ax.set_title("ECE / MCE per method")
            ax.legend()
            ax.grid(alpha=0.3)
            continue
        try:
            cal_probs = _apply(m, tile_scores, raw_probs, labels, args)
        except Exception as e:
            ax.text(0.5, 0.5, f"{m} failed:\n{e}", transform=ax.transAxes, ha="center")
            ax.set_title(m)
            continue
        centers, accs, confs, counts, ece, mce = reliability(cal_probs, labels, n_bins=args.n_bins)
        summary[m] = {"ece": ece, "mce": mce}
        # Plot
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
        present = counts > 0
        ax.plot(confs[present], accs[present], "o-", label=f"{m}", color="C0")
        ax.bar(centers, counts / max(counts.sum(), 1), width=1.0 / args.n_bins,
               alpha=0.2, color="C1", label="bin mass")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy / bin frac")
        ax.set_title(f"{m}  ECE={ece:.4f}  MCE={mce:.4f}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Reliability diagrams — {Path(args.scores).stem}", fontsize=12)
    fig.tight_layout()
    png = out / "reliability.png"
    pdf = out / "reliability.pdf"
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)

    (out / "reliability_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[calib-diag] wrote {png}, {pdf}, reliability_summary.json")
    print(f"[calib-diag] ECE per method: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
