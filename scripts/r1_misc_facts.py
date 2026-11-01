#!/usr/bin/env python3
"""Small facts the revised text quotes, each computed here rather than typed.

  * truncation_effect.json: tile-level AP@0.5 (101-point, the first submission's
    evaluator reproduced by src/r1_eval.py in tile mode) of the 50-epoch ships
    detector from predictions cached at Ultralytics' defaults (conf 0.25, at
    most 300 per tile) and at conf 0.001 without a binding cap (F4);
  * object_sizes.json: median object size (sqrt of oriented-box area, px) and
    the share of objects marked difficult in DOTA-1.5 val, per class (F13);
  * small_vehicle_scene.json: how much of the small-vehicle loss at the
    cross-fitted operating point comes from the single worst scene;
  * dataset_facts.json: tile and scene counts, the densest tile's detection
    count, HRSC2016 and SODA-A sizes, and how far each gate's checkpoint (chosen
    by validation PR-AUC during training) is from its last epoch.

    python scripts/r1_misc_facts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import shapely

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from r1_eval_cascade import tile_info  # noqa: E402
from r1_object_recall import found  # noqa: E402
from src.r1_eval import (CLASS_ID, CLASSES, EvalCache, build_cache, evaluate, load_tile_predictions,  # noqa: E402
                         load_yolo_gt, parse_stem, to_polygons)

OUT = REPO / "reports/r1"


def truncation() -> dict:
    root = REPO / "data/processed/dota_ships"
    tiles = sorted(p.stem for p in (root / "images/val").glob("*.png"))
    sizes = {t: parse_stem(t)[3:5] for t in tiles}
    windows = {t: (t, 0, 0, *sizes[t]) for t in tiles}
    gt = load_yolo_gt(root / "labels/val", tiles, sizes)
    out = {"detector": "runs/obb/runs/baseline_yolo11m_dota_ships_obb_p50", "level": "tile",
           "metric": "AP@0.5, 101-point interpolation"}
    for key, sub in (("conf025", "predict_val"), ("conf0001", "predict_val_c001")):
        preds = load_tile_predictions(REPO / out["detector"] / sub, tiles)
        cache = build_cache(preds, gt, windows, nms_iou=1.01)
        m = evaluate(cache, np.ones(len(cache.tiles), bool), classes=[0])
        out[f"ap50_coco_{key}"] = m["mAP50_coco"]
        out[f"ap50_voc07_{key}"] = m["mAP50_voc07"]
        out[f"n_detections_{key}"] = int(len(cache.det_score))
        out[f"min_score_{key}"] = float(cache.det_score.min())
    return out


def sizes(cache: EvalCache) -> dict:
    size = np.sqrt(np.maximum(shapely.area(to_polygons(cache.gt_corners)), 0))
    out = {"all": {"difficult_share": float(cache.gt_difficult.mean()), "n": int(len(size))}}
    for c, name in enumerate(CLASSES):
        m = cache.gt_cls == c
        if m.any():
            out[name] = {"n": int(m.sum()), "median_all": float(np.median(size[m])),
                         "median_nondiff": float(np.median(size[m & ~cache.gt_difficult]))
                         if (m & ~cache.gt_difficult).any() else None,
                         "difficult_share": float(cache.gt_difficult[m].mean())}
    return out


def sv_scene(cache: EvalCache) -> dict:
    run = json.load(open(OUT / "cascade/pillars/small_vehicle.json"))
    acc = np.load(OUT / "cascade/pillars/small_vehicle_masks.npz")["risk_pooled"]
    cls = run["classes"]
    full = np.ones(len(cache.tiles), bool)
    ff, fc = found(cache, full, cls, 0.25), found(cache, acc, cls, 0.25)
    m = np.isin(cache.gt_cls, cls) & ~cache.gt_difficult
    lost = np.bincount(cache.gt_image[m & ff & ~fc], minlength=len(cache.images))
    worst = int(np.argmax(lost))
    lab, _ = tile_info(cache.tiles, cls)
    tm = cache.tile_image == worst
    return {"scene": cache.images[worst], "lost_in_scene": int(lost[worst]), "lost_total": int(lost.sum()),
            "found_by_full_in_scene": int((m & ff & (cache.gt_image == worst)).sum()),
            "tiles": int(tm.sum()), "positive_tiles": int(lab[tm].sum()), "accepted_tiles": int(acc[tm].sum()),
            "conf": 0.25, "objects": "non-difficult small vehicles"}


def dataset_facts(cache: EvalCache) -> dict:
    meta = [json.loads(l) for l in open(REPO / "data/processed/dota_multiclass/metadata/tiles.jsonl")]
    hrsc = [json.loads(l) for l in open(REPO / "data/processed/hrsc2016/metadata/tiles.jsonl")]
    soda = json.load(open(REPO / "data/processed/soda_a/metadata/summary.json"))
    per_tile = np.bincount(cache.det_tile, minlength=len(cache.tiles))
    gaps = []
    for f in sorted(REPO.glob("runs/gate_*/history.json")):
        if "distill" in f.parent.name or "smoke" in f.parent.name:
            continue
        pr = [r["val/pr_auc"] for r in json.load(open(f)) if r.get("val/pr_auc") is not None]
        if pr:
            gaps.append(max(pr) - pr[-1])
    test = [r for r in hrsc if r.get("split") == "test"]
    # positive tiles (any instance of the class, difficult included, as the gates are trained) that
    # hold only difficult instances, which the DOTA matching ignores
    n_per = np.diff(cache.gt_tiles_ptr)
    owner = np.repeat(np.arange(len(n_per)), n_per)
    only_difficult = {}
    for name in ("ship", "plane", "small vehicle"):
        m = cache.gt_cls == CLASS_ID[name]
        any_t = set(cache.gt_tiles_idx[m[owner]].tolist())
        nd_t = set(cache.gt_tiles_idx[(m & ~cache.gt_difficult)[owner]].tolist())
        only_difficult[name.replace(" ", "_")] = {"positive_tiles": len(any_t),
                                                  "only_difficult": len(any_t - nd_t)}
    return {"dota_train_tiles": sum(r["split"] == "train" for r in meta),
            "positive_tiles_only_difficult": only_difficult,
            "dota_val_tiles": sum(r["split"] == "val" for r in meta),
            "dota_val_scenes": len({r["source_stem"] for r in meta if r["split"] == "val"}),
            "densest_tile_detections_multiclass": int(per_tile.max()),
            "hrsc_test_images": len(test), "hrsc_test_negatives": sum(1 for r in test if not int(r["is_positive"])),
            "soda": soda, "gate_checkpoints": len(gaps),
            "gate_prauc_best_minus_last_median": float(np.median(gaps)),
            "gate_prauc_best_minus_last_max": float(np.max(gaps))}


def main() -> int:
    cache = EvalCache.load(REPO / "runs/r1_cache/multiclass_val.npz")
    for name, fn in (("truncation_effect", lambda: truncation()), ("object_sizes", lambda: sizes(cache)),
                     ("small_vehicle_scene", lambda: sv_scene(cache)),
                     ("dataset_facts", lambda: dataset_facts(cache))):
        d = fn()
        (OUT / f"{name}.json").write_text(json.dumps(d, indent=2))
        print(name, json.dumps(d)[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
