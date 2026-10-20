#!/usr/bin/env python3
"""Check src/r1_eval.py against the evaluator that produced the first submission.

In tile-level mode with merging switched off (every tile is its own "image",
NMS IoU above 1), the new code must reproduce src/cascade_eval.py on the same
inputs. Run on the ships per-class detector's original conf-0.25 cache and the
MobileNetV3-large gate, it compares mAP@0.5, mAP@0.5:0.95, filter rate and gate
recall at every threshold of reports/cascade/gate_mbv3large_identity.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.r1_eval import (IOU_THRESHOLDS, build_cache, evaluate, load_tile_predictions,  # noqa: E402
                         load_yolo_gt, parse_stem)


def main() -> int:
    root = REPO / "data/processed/dota_ships"
    pred = REPO / "runs/obb/runs/baseline_yolo11m_dota_ships_obb/predict_val"
    scores = {}
    labels = {}
    for line in open(REPO / "reports/gate_scores/gate_mbv3large_val.jsonl"):
        r = json.loads(line)
        scores[r["tile_id"]] = r["prob"]
        labels[r["tile_id"]] = r["label"]
    tiles = sorted(scores)
    sizes = {t: parse_stem(t)[3:5] for t in tiles}
    windows = {t: (t, 0, 0, *sizes[t]) for t in tiles}
    preds = load_tile_predictions(pred, tiles)
    gt = load_yolo_gt(root / "labels/val", tiles, sizes)
    cache = build_cache(preds, gt, windows, nms_iou=1.01)
    prob = np.array([scores[t] for t in cache.tiles])
    lab = np.array([labels[t] for t in cache.tiles])
    ref = {round(r["threshold"], 3): r for r in json.load(open(REPO / "reports/cascade/gate_mbv3large_identity.json"))}

    worst = 0.0
    for thr in sorted(ref):
        acc = prob >= thr
        m = evaluate(cache, acc, classes=[0], iou_thresholds=IOU_THRESHOLDS)
        r = ref[thr]
        rows = [("mAP@0.5", m["mAP50_coco"], r["mAP@0.50"]),
                ("mAP@0.5:0.95", m["mAP50_95_coco"], r["mAP@0.5:0.95"]),
                ("filter", 1 - acc.mean(), r["filter_rate"]),
                ("gate_recall", acc[lab == 1].mean(), r["gate_recall_on_positive_tiles"])]
        diffs = [abs(a - b) for _, a, b in rows]
        worst = max(worst, *diffs)
        print(f"t={thr:.2f}  " + "  ".join(f"{n} new={a:.4f} old={b:.4f}" for n, a, b in rows))
    print(f"\nworst absolute difference: {worst:.2e}")
    return 0 if worst < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
