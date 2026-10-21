#!/usr/bin/env python3
"""Stratified behaviour at a cross-fitted operating point (F10, Section VII).

The first submission stratified tiles by properties of the largest object of any
class and called a tile "boundary" when that object spanned 70 % of the tile.
Here every stratum is defined from the gated class itself:

  * sensor source and ground sampling distance are properties of the scene, so
    image-level mAP@0.5 (101-point; 11-point VOC07 alongside) is computed per bucket for the full pass and the
    cascade, with p+, accept rate and positive-tile acceptance;
  * size uses the gated-class objects: per tile, the largest gated object
    (sqrt of the oriented-box area); per object, its own size;
  * boundary incidence is measured from the polygons: a gated-class object is
    clipped on a tile when between 10 % and 99 % of its area lies inside the
    tile window. A positive tile is "fragments only" when every gated object on
    it is clipped; an object is "always clipped" when no tile contains 99 % of
    it, so that only a fragment can ever be seen.

Gate acceptance (TPR) is reported for positive tiles in each tile stratum, and
object recall at detector confidence 0.25 (non-difficult objects, after merging)
for each object stratum.

    python scripts/r1_stratified.py --run reports/r1/cascade/pillars/ship.json
"""
from __future__ import annotations

import argparse
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
from src.r1_eval import EvalCache, evaluate, parse_stem, to_polygons  # noqa: E402

SIZE_BUCKETS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, float("inf"))]
GSD_BUCKETS = [(0.0, 0.15), (0.15, 0.30), (0.30, 0.60), (0.60, 1.50), (1.50, float("inf"))]
CLIP_LO, CLIP_HI = 0.10, 0.99
LABELS = REPO / "data/raw/DOTA/val/labelTxt"


def image_gsd(stem: str) -> float | None:
    for line in open(LABELS / f"{stem}.txt"):
        if line.startswith("gsd:"):
            v = line.split(":", 1)[1].strip()
            try:
                return float(v)
            except ValueError:
                return None
        if not line.startswith("imagesource:"):
            break
    return None


def bucket_name(lo: float, hi: float, unit: str) -> str:
    return f"{lo:g}-{hi:g}{unit}" if hi != float("inf") else f">={lo:g}{unit}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--mask", default="risk_pooled")
    args = p.parse_args()
    run = json.load(open(REPO / args.run))
    cache = EvalCache.load(REPO / run["cache"])
    classes = run["classes"]
    acc = np.load(REPO / args.run.replace(".json", "_masks.npz"))[args.mask]
    labels, src = tile_info(cache.tiles, classes)
    full = np.ones(len(cache.tiles), bool)

    # containment of every gated-class object in every tile that holds >= 10 % of it
    gsel = np.isin(cache.gt_cls, classes)
    polys = to_polygons(cache.gt_corners)
    area = shapely.area(polys)
    size = np.sqrt(np.maximum(area, 0))
    n_t = np.diff(cache.gt_tiles_ptr)
    owner = np.repeat(np.arange(len(n_t)), n_t)
    win = np.array([parse_stem(t)[1:] for t in cache.tiles], dtype=float)
    boxes = shapely.box(win[:, 0], win[:, 1], win[:, 0] + win[:, 2], win[:, 1] + win[:, 3])
    pair_g = owner[gsel[owner]]
    pair_t = cache.gt_tiles_idx[gsel[owner]]
    frac = shapely.area(shapely.intersection(polys[pair_g], boxes[pair_t])) / np.maximum(area[pair_g], 1e-9)
    clipped_pair = (frac >= CLIP_LO) & (frac < CLIP_HI)
    max_frac = np.zeros(len(cache.gt_cls))
    np.maximum.at(max_frac, pair_g, frac)
    always_clipped = gsel & (max_frac < CLIP_HI)

    n_tiles = len(cache.tiles)
    whole_on_tile = np.bincount(pair_t, weights=(frac >= CLIP_HI).astype(float), minlength=n_tiles) > 0
    clipped_on_tile = np.bincount(pair_t, weights=clipped_pair.astype(float), minlength=n_tiles) > 0
    largest = np.zeros(n_tiles)
    np.maximum.at(largest, pair_t[frac >= CLIP_LO], size[pair_g[frac >= CLIP_LO]])
    pos = labels.astype(bool)

    def tile_row(m: np.ndarray) -> dict:
        n_pos = int((m & pos).sum())
        return {"n_tiles": int(m.sum()), "n_positive": n_pos,
                "p_plus": float((m & pos).sum() / max(m.sum(), 1)),
                "accept": float(acc[m].mean()) if m.any() else float("nan"),
                "tpr": float(acc[m & pos].mean()) if n_pos else float("nan")}

    def image_row(img: np.ndarray) -> dict:
        tm = img[cache.tile_image]
        r = tile_row(tm)
        f = evaluate(cache, full & tm, classes, images=img)
        c = evaluate(cache, acc & tm, classes, images=img)
        n_obj = int((gsel & ~cache.gt_difficult & img[cache.gt_image]).sum())
        return dict(r, n_images=int(img.sum()), n_objects=n_obj,
                    mAP50_full=f["mAP50_coco"], mAP50_cascade=c["mAP50_coco"],
                    delta_pp=100 * (c["mAP50_coco"] - f["mAP50_coco"]),
                    mAP50_voc07_full=f["mAP50_voc07"], mAP50_voc07_cascade=c["mAP50_voc07"])

    ff = found(cache, full, classes, 0.25)
    fc = found(cache, acc, classes, 0.25)
    nd = gsel & ~cache.gt_difficult

    def object_row(m: np.ndarray) -> dict:
        m = m & nd
        n = int(m.sum())
        rf = float(ff[m].mean()) if n else float("nan")
        rc = float(fc[m].mean()) if n else float("nan")
        return {"n_objects": n, "recall_full": rf, "recall_cascade": rc, "delta_pp": 100 * (rc - rf) if n else float("nan")}

    img_src = np.array([src[np.nonzero(cache.tile_image == i)[0][0]] for i in range(len(cache.images))])
    gsd = np.array([image_gsd(s) if image_gsd(s) is not None else np.nan for s in cache.images])
    out = {"run": args.run, "mask": args.mask, "classes": classes,
           "definitions": {"clipped": f"{CLIP_LO:.0%} <= area inside tile < {CLIP_HI:.0%}",
                           "size": "sqrt of oriented-box area, gated class only",
                           "recall": "non-difficult objects, detector confidence 0.25, after image-level merging"},
           "overall": image_row(np.ones(len(cache.images), bool)),
           "by_source": {s: image_row(img_src == s) for s in sorted(set(img_src))},
           "by_gsd": {}, "tiles_by_largest_gated_object": {}, "tiles_by_boundary": {},
           "objects_by_size": {}, "objects_by_boundary": {}}
    for lo, hi in GSD_BUCKETS:
        m = (gsd >= lo) & (gsd < hi)
        if m.any():
            out["by_gsd"][bucket_name(lo, hi, "m")] = image_row(m)
    if np.isnan(gsd).any():
        out["by_gsd"]["unknown"] = image_row(np.isnan(gsd))
    for lo, hi in SIZE_BUCKETS:
        out["tiles_by_largest_gated_object"][bucket_name(lo, hi, "px")] = tile_row(pos & (largest >= lo) & (largest < hi))
        out["objects_by_size"][bucket_name(lo, hi, "px")] = object_row((size >= lo) & (size < hi))
    out["tiles_by_boundary"] = {
        "whole objects only": tile_row(pos & whole_on_tile & ~clipped_on_tile),
        "whole and clipped": tile_row(pos & whole_on_tile & clipped_on_tile),
        "fragments only": tile_row(pos & ~whole_on_tile & clipped_on_tile),
    }
    out["objects_by_boundary"] = {"whole in some tile": object_row(gsel & ~always_clipped),
                                  "always clipped": object_row(always_clipped)}
    dest = REPO / "reports/r1/stratified"
    dest.mkdir(parents=True, exist_ok=True)
    stem = Path(args.run).parent.name + "_" + Path(args.run).stem.replace(" ", "_")
    (dest / f"{stem}.json").write_text(json.dumps(out, indent=2, default=float))
    o = out["overall"]
    print(f"{stem}: mAP {o['mAP50_full']:.3f} -> {o['mAP50_cascade']:.3f}, accept {o['accept']:.3f}")
    for k in ("by_source", "by_gsd"):
        for b, r in out[k].items():
            print(f"  {k} {b:<14} img={r['n_images']:<4} p+={r['p_plus']:.3f} acc={r['accept']:.3f} "
                  f"tpr={r['tpr']:.3f} mAP {r['mAP50_full']:.3f}->{r['mAP50_cascade']:.3f} ({r['delta_pp']:+.2f})")
    for k in ("tiles_by_largest_gated_object", "tiles_by_boundary"):
        for b, r in out[k].items():
            print(f"  {k} {b:<18} pos tiles={r['n_positive']:<5} tpr={r['tpr']:.3f}")
    for k in ("objects_by_size", "objects_by_boundary"):
        for b, r in out[k].items():
            print(f"  {k} {b:<18} n={r['n_objects']:<5} recall {r['recall_full']:.3f}->{r['recall_cascade']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
