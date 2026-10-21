#!/usr/bin/env python3
"""How much of a stratum's AP loss one scene carries (WP13 check of Section XII).

For every stratum of scripts/r1_stratified.py whose cascade AP change exceeds --min-loss
points, recompute the change with each scene of the stratum left out in turn and report
the scene whose removal moves it most. Same masks (cross-fitted headline operating
points, reports/r1/cascade/pillars/<task>_masks.npz) and the same image-level 101-point AP
as the strata table. Writes reports/r1/stratified/scene_effects.json.

    python scripts/r1_strata_scenes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from r1_stratified import GSD_BUCKETS, image_gsd  # noqa: E402
from src.r1_eval import CLASS_ID, EvalCache, evaluate  # noqa: E402

CACHE = REPO / "runs/r1_cache/multiclass_val.npz"
META = REPO / "data/processed/dota_multiclass/metadata/tiles.jsonl"
OUT = REPO / "reports/r1/stratified/scene_effects.json"
TASKS = {"ship": "ship", "plane": "plane", "small_vehicle": "small vehicle"}


def dap(cache, acc, classes, sel) -> float:
    full = evaluate(cache, np.ones(len(cache.tiles), bool), classes, images=sel)["mAP50_coco"]
    return evaluate(cache, acc, classes, images=sel)["mAP50_coco"] - full


def main(min_loss: float = 5.0) -> int:
    cache = EvalCache.load(CACHE)
    source = {}
    for line in open(META):
        r = json.loads(line)
        if r["split"] == "val":
            source[r["source_stem"]] = r.get("imagesource") or "unknown"
    gsd = {im: image_gsd(im) for im in cache.images}
    strata = {f"source {s}": np.array([source.get(im) == s for im in cache.images])
              for s in sorted(set(source.values()))}
    for lo, hi in GSD_BUCKETS:
        strata[f"gsd {lo}-{hi}"] = np.array([gsd[im] is not None and lo <= gsd[im] < hi for im in cache.images])
    out = {}
    for slug, task in TASKS.items():
        masks = np.load(REPO / f"reports/r1/cascade/pillars/{slug}_masks.npz")
        assert list(masks["tiles"]) == list(cache.tiles)
        acc, classes = masks["risk_pooled"], [CLASS_ID[task]]
        for name, sel in strata.items():
            if not sel.any():
                continue
            d = dap(cache, acc, classes, sel)
            if not d < -min_loss / 100:
                continue
            best = None
            for k in np.nonzero(sel)[0]:
                n_obj = int(((cache.gt_image == k) & (cache.gt_cls == classes[0]) & ~cache.gt_difficult).sum())
                if n_obj == 0:
                    continue
                s2 = sel.copy()
                s2[k] = False
                d2 = dap(cache, acc, classes, s2)
                if best is None or d2 > best["dap_without"]:
                    tiles = np.nonzero(cache.tile_image == k)[0]
                    best = {"scene": cache.images[k], "objects": n_obj, "tiles": int(len(tiles)),
                            "accepted": int(acc[tiles].sum()), "dap_without": float(d2)}
            out[f"{slug} / {name}"] = {"scenes": int(sel.sum()), "dap": float(d), "largest_single_scene": best}
            print(f"{slug:14s} {name:22s} scenes={int(sel.sum()):3d} dAP={100 * d:+.2f}  without "
                  f"{best['scene']} ({best['objects']} objects, {best['accepted']}/{best['tiles']} tiles accepted): "
                  f"{100 * best['dap_without']:+.2f}", flush=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
