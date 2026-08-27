#!/usr/bin/env python3
"""Evaluate one cascade configuration end-to-end.

Inputs:
  - Gate scores JSONL produced by ``scripts/score_tiles.py``
  - Detector predictions directory (Ultralytics ``predict`` output)
  - Ground-truth label directory (the YOLO-OBB labels under
    ``data/processed/<run>/labels/<split>/``)
  - Stage costs (gate + detector) measured via ``src.experiments.speed``

Outputs:
  - Per-threshold rows of (mAP, filter_rate, gate_recall, GFLOPs, latency).
  - One JSON summary at ``--out`` and a CSV companion alongside.

Special modes:
  - ``--gate oracle``: synthesize a perfect gate from tiles.jsonl. Required
    upper-bound baseline (research plan §7.1 / §15.x).
  - ``--calibration {identity|temperature|platt|isotonic|context_adaptive}``: fit a calibrator
    on the supplied scores (or a separate ``--calibration-fit-scores`` file)
    and threshold the calibrated probs. ``context_adaptive`` fits a per-bucket
    recall-targeted threshold (Method 5 in the research plan, see
    ``src.calibration.fit_context_adaptive``) and shifts each tile's prob by
    ``(0.5 - threshold_for(bucket))`` so a downstream threshold sweep at
    ``t=0.5`` reproduces the per-bucket operating points exactly, and other
    thresholds trace a Pareto curve through them.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.calibration import build_calibration  # noqa: E402
from src.cascade import TileScore  # noqa: E402
from src.cascade_eval import (  # noqa: E402
    ComputeAccountant,
    StageCost,
    assemble_for_evaluation,
    cascade_pareto_row,
    oracle_tile_scores,
    precompute_iou,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--scores", help="Path to gate scores JSONL (omit when --gate oracle)")
    p.add_argument("--gate", default="real", choices=("real", "oracle"))
    p.add_argument(
        "--metadata-jsonl",
        help="Path to tiles.jsonl. Required for --gate oracle and recommended otherwise so labels are authoritative.",
    )
    p.add_argument("--split", default="val", help="Used to filter the oracle scores")
    p.add_argument("--detector-runs", required=True, help="Ultralytics predict output dir")
    p.add_argument("--gt-labels", required=True, help="Ground-truth YOLO-OBB labels dir")
    p.add_argument("--image-root", required=True, help="Path to images/<split>/")
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument(
        "--gt-classes",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Restrict ground truth to these class ids. mAP averages over the classes "
            "present in the GT, so a single id yields that class's AP. Use this to pair "
            "a single-class gate with the multi-class detector without writing a "
            "filtered copy of every label file."
        ),
    )
    p.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=None,
        help="Explicit thresholds. Defaults to a 21-point sweep [0, 1].",
    )
    p.add_argument("--calibration", default="identity",
                   choices=("identity", "temperature", "platt", "isotonic", "context_adaptive"))
    p.add_argument("--all-calibrations", action="store_true",
                   help="Run all 5 calibrations in one pass (precompute IoU once). "
                        "--out is treated as the *identity* output path; sibling files "
                        "are written for temperature/platt/isotonic/context_adaptive automatically.")
    p.add_argument("--calibration-fit-scores", default=None, help="Optional: separate score JSONL to FIT the calibrator on")
    p.add_argument("--context-adaptive-bucket", default="imagesource",
                   choices=("imagesource", "gsd_bucket", "size_bucket", "boundary"),
                   help="Stratum used by --calibration context_adaptive to define buckets "
                        "(default: imagesource). Requires --metadata-jsonl.")
    p.add_argument("--context-adaptive-target-recall", type=float, default=0.99,
                   help="Per-bucket target recall used to fit context_adaptive thresholds (default: 0.99).")
    p.add_argument("--gate-flops-g", type=float, default=0.0)
    p.add_argument("--gate-latency-ms", type=float, default=0.0)
    p.add_argument("--detector-flops-g", type=float, default=0.0)
    p.add_argument("--detector-latency-ms", type=float, default=0.0)
    p.add_argument("--label", default="cascade", help="Tag for the resulting rows")
    p.add_argument("--out", required=True, help="Output JSON path; .csv emitted alongside")
    p.add_argument(
        "--stratum",
        default=None,
        choices=("imagesource", "gsd_bucket", "size_bucket", "boundary", "split"),
        help=(
            "If set, additionally emit per-bucket mAP at every threshold. "
            "Requires --metadata-jsonl. Output is a sibling JSON at "
            "<out>.stratified_<stratum>.json."
        ),
    )
    return p.parse_args()


def _load_scores(args: argparse.Namespace) -> list[TileScore]:
    if args.gate == "oracle":
        if not args.metadata_jsonl:
            raise SystemExit("--gate oracle requires --metadata-jsonl")
        return oracle_tile_scores(args.metadata_jsonl, split_filter={args.split})
    if not args.scores:
        raise SystemExit("--scores is required when --gate real")
    from src.cascade import load_tile_scores

    return load_tile_scores(args.scores)


def _build_tile_to_bucket(metadata_jsonl: str, stratum: str) -> dict:
    """Inverse of stratify_tiles_by: tile_id -> bucket name."""
    from src.cascade_eval import stratify_tiles_by
    bk2tids = stratify_tiles_by(metadata_jsonl, stratum)
    return {tid: bk for bk, tids in bk2tids.items() for tid in tids}


def _apply_calibration(
    args: argparse.Namespace, tile_scores: list, calibration: str
) -> list:
    from src.cascade import load_tile_scores as _lts
    if calibration == "identity" or args.gate == "oracle":
        return tile_scores
    fit_path = args.calibration_fit_scores or args.scores
    fit_scores = _lts(fit_path)
    fit_probs = np.array([t.prob for t in fit_scores])
    fit_labels = np.array([t.label for t in fit_scores], dtype=np.int64)

    if calibration == "context_adaptive":
        from src.calibration import fit_context_adaptive
        if not args.metadata_jsonl:
            raise SystemExit("--calibration context_adaptive requires --metadata-jsonl")
        tile_to_bucket = _build_tile_to_bucket(args.metadata_jsonl, args.context_adaptive_bucket)
        bucket_of = lambda tid: tile_to_bucket.get(tid, "unknown")
        ca = fit_context_adaptive(
            tile_ids=[t.tile_id for t in fit_scores],
            probs=fit_probs,
            labels=fit_labels,
            bucket_of=bucket_of,
            target_recall=args.context_adaptive_target_recall,
        )
        # Shift each tile's prob by (0.5 - per-bucket threshold) so the
        # downstream threshold sweep at t=0.5 reproduces the per-bucket
        # recall-targeted operating points; other thresholds offset uniformly.
        out = []
        for t in tile_scores:
            shift = 0.5 - ca.threshold_for(bucket_of(t.tile_id))
            new_prob = float(np.clip(t.prob + shift, 0.0, 1.0))
            out.append(TileScore(tile_id=t.tile_id, label=t.label, prob=new_prob))
        return out

    calib = build_calibration(calibration).fit(fit_probs, fit_labels)
    raw = np.array([t.prob for t in tile_scores])
    calibrated = calib.transform(raw)
    return [TileScore(tile_id=t.tile_id, label=t.label, prob=float(p))
            for t, p in zip(tile_scores, calibrated)]


def _run_one_calibration(
    args: argparse.Namespace,
    raw_scores: list,
    detections: dict,
    ground_truth: dict,
    accountant,
    thresholds: list,
    iou_cache: dict,
    calibration: str,
    out_path: Path,
) -> None:
    scores = _apply_calibration(args, raw_scores, calibration)
    rows: list[dict] = []
    stratified_rows: list[dict] = []
    for t in thresholds:
        row = cascade_pareto_row(
            tile_scores=scores,
            detections=detections,
            ground_truth=ground_truth,
            threshold=float(t),
            accountant=accountant,
            label=args.label,
            _precomputed=iou_cache,
        )
        row["calibration"] = calibration
        row["gate"] = args.gate
        rows.append(row)
        print(
            f"[{calibration}] t={row['threshold']:.3f} "
            f"filter={row['filter_rate']:.3f} "
            f"gate_rec={row.get('gate_recall_on_positive_tiles', float('nan')):.3f} "
            f"mAP@0.5={row.get('mAP@0.50', row.get('mAP@0.5', 0.0)):.4f} "
            f"GFLOPs={row['total_gflops']:.1f} "
            f"latency_ms={row['total_latency_ms']:.1f}"
        )

        if args.stratum:
            if not args.metadata_jsonl:
                raise SystemExit("--stratum requires --metadata-jsonl")
            from src.cascade import apply_threshold
            from src.cascade_eval import evaluate_stratified

            outputs = apply_threshold(scores, detections, float(t))
            tile_label_map = {ts.tile_id: ts.label for ts in scores}
            strat = evaluate_stratified(
                outputs, ground_truth, args.metadata_jsonl, args.stratum,
                tile_label_map=tile_label_map,
            )
            for bucket_name, metrics in strat.items():
                stratified_rows.append(
                    {
                        "label": args.label,
                        "calibration": calibration,
                        "gate": args.gate,
                        "threshold": float(t),
                        "stratum": args.stratum,
                        "bucket": bucket_name,
                        **metrics,
                    }
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    csv_path = out_path.with_suffix(".csv")
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    print(f"[eval_cascade] wrote {len(rows)} rows to {out_path} and {csv_path}")

    if stratified_rows:
        strat_path = out_path.with_suffix(f".stratified_{args.stratum}.json")
        with strat_path.open("w", encoding="utf-8") as handle:
            json.dump(stratified_rows, handle, indent=2)
        strat_csv = strat_path.with_suffix(".csv")
        keys = sorted({k for r in stratified_rows for k in r.keys()})
        with strat_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for r in stratified_rows:
                writer.writerow(r)
        print(f"[eval_cascade] wrote {len(stratified_rows)} stratified rows to {strat_path}")


def main() -> int:
    args = parse_args()
    raw_scores = _load_scores(args)
    if not raw_scores:
        raise SystemExit("No tile scores loaded")

    _, detections, ground_truth = assemble_for_evaluation(
        score_jsonl=args.scores or "/dev/null",
        detector_runs_dir=args.detector_runs,
        gt_labels_dir=args.gt_labels,
        detector_image_root=args.image_root,
        tile_size=args.tile_size,
        keep_classes=set(args.gt_classes) if args.gt_classes else None,
    )
    accountant = ComputeAccountant(
        gate=StageCost(name="gate", flops_g=args.gate_flops_g, latency_ms=args.gate_latency_ms),
        detector=StageCost(name="detector", flops_g=args.detector_flops_g, latency_ms=args.detector_latency_ms),
    )
    thresholds = args.thresholds if args.thresholds else list(np.linspace(0.0, 1.0, 21))

    print(f"[eval_cascade] precomputing IoU matrices ({len(detections)} tiles) ...", flush=True)
    iou_cache = precompute_iou(detections, ground_truth)
    print(f"[eval_cascade] IoU precomputed; sweeping {len(thresholds)} thresholds ...", flush=True)

    if args.all_calibrations and args.gate != "oracle":
        calibrations = ["identity", "temperature", "platt", "isotonic", "context_adaptive"]
        out_base = Path(args.out)
        # derive sibling paths by replacing the calibration suffix in the filename
        for calib in calibrations:
            # Replace whatever calibration appears in the --out name with the current one
            stem = out_base.stem
            for known in ["identity", "temperature", "platt", "isotonic", "context_adaptive"]:
                if stem.endswith(f"_{known}"):
                    stem = stem[: -len(f"_{known}")]
                    break
            out_path = out_base.parent / f"{stem}_{calib}.json"
            if out_path.exists():
                print(f"[eval_cascade] skip {out_path} (exists)", flush=True)
                continue
            print(f"[eval_cascade] === calibration: {calib} → {out_path} ===", flush=True)
            _run_one_calibration(args, raw_scores, detections, ground_truth,
                                 accountant, thresholds, iou_cache, calib, out_path)
    else:
        _run_one_calibration(args, raw_scores, detections, ground_truth,
                             accountant, thresholds, iou_cache, args.calibration, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
