#!/usr/bin/env python3
"""Build the image-level evaluation cache (src/r1_eval.EvalCache) for one prediction set.

    python scripts/r1_build_cache.py --pred runs/obb/runs/baseline_yolo11m_dota_multiclass_obb/predict_val_c001 \
        --out runs/r1_cache/multiclass_val.npz
    python scripts/r1_build_cache.py --pred <ships detector>/predict_val_c001 --class-map 0:1 --out ...
    python scripts/r1_build_cache.py --pred <ships detector>/predict_hrsc_test_c001 --hrsc --out ...

DOTA: tiles and their windows come from data/processed/dota_multiclass/metadata/tiles.jsonl
(val split), ground truth from the raw labelTxt files with difficult flags. Detector class
ids are mapped to the 16-class ids with --class-map for single-class detectors.

HRSC2016: images are not tiled, so each image is its own tile; ground truth comes from the
processed single-class labels (HRSC difficult flags are not carried by the conversion).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.r1_eval import build_cache, load_dota_image_gt, load_tile_predictions, load_yolo_gt  # noqa: E402

DOTA_META = REPO / "data/processed/dota_multiclass/metadata/tiles.jsonl"
DOTA_RAW_VAL = REPO / "data/raw/DOTA/val/labelTxt"
HRSC_META = REPO / "data/processed/hrsc2016/metadata/tiles.jsonl"
HRSC_LABELS = REPO / "data/processed/hrsc2016/labels/test"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="prediction run dir containing labels/")
    p.add_argument("--out", required=True)
    p.add_argument("--class-map", default=None, help="detector:global ids, e.g. 0:1 for a ships detector")
    p.add_argument("--hrsc", action="store_true")
    p.add_argument("--nms-iou", type=float, default=0.3)
    args = p.parse_args()
    t0 = time.time()

    meta_path = HRSC_META if args.hrsc else DOTA_META
    windows, sizes = {}, {}
    for line in open(meta_path):
        r = json.loads(line)
        if not args.hrsc and r["split"] != "val":
            continue
        tid = r["tile_id"]
        sizes[tid] = (int(r["width"]), int(r["height"]))
        windows[tid] = (tid, 0, 0, 0, 0) if args.hrsc else (r["source_stem"], int(r["x"]), int(r["y"]),
                                                             int(r["width"]), int(r["height"]))
    tiles = sorted(windows)
    preds = load_tile_predictions(args.pred, tiles, sizes=sizes)
    if args.class_map:
        cmap = {int(a): int(b) for a, b in (kv.split(":") for kv in args.class_map.split(","))}
        preds = {t: (np.array([cmap.get(int(c), -1) for c in cls], dtype=np.int64), s, corners)
                 for t, (cls, s, corners) in preds.items()}
    n_det = sum(len(v[0]) for v in preds.values())
    if args.hrsc:
        gt = load_yolo_gt(HRSC_LABELS, tiles, sizes)
    else:
        gt = load_dota_image_gt(DOTA_RAW_VAL, sorted({w[0] for w in windows.values()}))
    cache = build_cache(preds, gt, windows, nms_iou=args.nms_iou,
                        meta={"pred": args.pred, "class_map": args.class_map, "hrsc": args.hrsc})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cache.save(args.out)
    print(f"[r1_build_cache] {len(cache.tiles)} tiles, {len(cache.images)} images, {n_det} detections, "
          f"{len(cache.gt_cls)} objects ({int(cache.gt_difficult.sum())} difficult), "
          f"{len(cache.sup_idx)} NMS edges, {time.time() - t0:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
