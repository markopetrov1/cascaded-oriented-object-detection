#!/usr/bin/env python3
"""Tile rejection versus detection loss, at object level after merging (R1.3, R2.3).

For a cascade operating point (the accepted-tile masks written by
scripts/r1_eval_cascade.py), every ground-truth object of the task is classified
after image-level merging:

  found by the full pass?  found by the cascade?  how many of its tiles were rejected?

An object whose tiles were all rejected cannot be found by the cascade; it is a
real loss only if the full pass found it ("lost, all tiles rejected"), otherwise
the detector would have missed it anyway. An object with at least one accepted
tile can still be found through the overlap ("recovered"). Reported at the
deployment confidence 0.25, at conf 0.001 (the recall ceiling of the cache), and
at the full pass's F1-optimal confidence, for non-difficult objects (the DOTA
protocol) and for all objects, stratified by object size and sensor source.

    python scripts/r1_object_recall.py --run reports/r1/cascade/pillars/ship.json
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

from src.r1_eval import EvalCache, match, merged_keep, to_polygons  # noqa: E402

SIZE_BUCKETS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, float("inf"))]
META = REPO / "data/processed/dota_multiclass/metadata/tiles.jsonl"


def found(cache: EvalCache, acc: np.ndarray, classes: list[int], conf: float) -> np.ndarray:
    keep = merged_keep(cache, acc, cache.det_score >= conf, classes)
    return match(cache, keep, 0.5, classes)["gt_hit"]


def f1_optimal_conf(cache: EvalCache, classes: list[int]) -> float:
    full = np.ones(len(cache.tiles), bool)
    keep = merged_keep(cache, full, classes=classes)
    m = match(cache, keep, 0.5, classes)
    n_pos = int(m["gt_sel"].sum())
    s = cache.det_score[m["det"]][m["valid"]]
    tp = m["tp"][m["valid"]]
    o = np.argsort(-s)
    ctp = np.cumsum(tp[o]); n = np.arange(1, len(o) + 1)
    f1 = 2 * ctp / (n + n_pos)
    return float(s[o][int(np.argmax(f1))])


def summarise(sel: np.ndarray, ff: np.ndarray, fc: np.ndarray, n_rej: np.ndarray, n_tiles: np.ndarray) -> dict:
    all_rej = (n_tiles > 0) & (n_rej == n_tiles)
    some_rej = (n_rej > 0) & ~all_rej
    s = lambda m: int((sel & m).sum())
    n = s(np.ones_like(sel))
    out = {"n_objects": n,
           "recall_full": s(ff) / n if n else float("nan"),
           "recall_cascade": s(fc) / n if n else float("nan"),
           "lost": s(ff & ~fc), "gained": s(~ff & fc),
           "on_fully_rejected_tiles": s(all_rej),
           "fully_rejected_lost": s(all_rej & ff), "fully_rejected_missed_anyway": s(all_rej & ~ff),
           "on_partly_rejected_tiles": s(some_rej),
           "partly_rejected_recovered": s(some_rej & ff & fc), "partly_rejected_lost": s(some_rej & ff & ~fc),
           "partly_rejected_missed_anyway": s(some_rej & ~ff)}
    out["delta_recall_pp"] = 100 * (out["recall_cascade"] - out["recall_full"]) if n else float("nan")
    out["share_on_rejected_tiles_pp"] = 100 * out["on_fully_rejected_tiles"] / n if n else float("nan")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="reports/r1/cascade/<preset>/<task>.json")
    p.add_argument("--masks", nargs="*", default=["risk_pooled", "plain_pooled", "same_split"])
    args = p.parse_args()
    run = json.load(open(REPO / args.run))
    cache = EvalCache.load(REPO / run["cache"])
    classes = run["classes"]
    mz = np.load(REPO / args.run.replace(".json", "_masks.npz"))
    assert list(mz["tiles"]) == cache.tiles

    gsel_cls = np.isin(cache.gt_cls, classes)
    area = shapely.area(to_polygons(cache.gt_corners))
    size = np.sqrt(np.maximum(area, 0))
    src_of = {}
    for line in open(META):
        r = json.loads(line)
        src_of[r["source_stem"]] = r.get("imagesource") or "unknown"
    gt_src = np.array([src_of.get(cache.images[i], "HRSC2016") for i in cache.gt_image])
    n_tiles = np.diff(cache.gt_tiles_ptr)
    owner = np.repeat(np.arange(len(n_tiles)), n_tiles)

    confs = {"conf_0.25": 0.25, "conf_0.001": 0.001, "conf_f1opt": f1_optimal_conf(cache, classes)}
    out = {"run": args.run, "classes": classes, "conf_thresholds": confs,
           "size_buckets_px": [list(b) for b in SIZE_BUCKETS], "results": {}}
    full = np.ones(len(cache.tiles), bool)
    for mk in args.masks:
        acc = mz[mk]
        n_rej = np.bincount(owner, weights=(~acc[cache.gt_tiles_idx]).astype(float), minlength=len(n_tiles))
        res = {"accept_rate": float(acc.mean())}
        for cname, cval in confs.items():
            ff = found(cache, full, classes, cval)
            fc = found(cache, acc, classes, cval)
            block = {}
            for dname, dsel in (("non_difficult", gsel_cls & ~cache.gt_difficult), ("all_objects", gsel_cls)):
                # difficult objects are ignored by DOTA matching, so for "all objects" a difficult
                # object counts as found when some kept detection overlaps it at IoU >= 0.5
                ffd, fcd = ff, fc
                if dname == "all_objects":
                    ffd = ff | hit_any(cache, full, classes, cval)
                    fcd = fc | hit_any(cache, acc, classes, cval)
                d = {"overall": summarise(dsel, ffd, fcd, n_rej, n_tiles)}
                d["by_size"] = {f"{lo}-{hi}": summarise(dsel & (size >= lo) & (size < hi), ffd, fcd, n_rej, n_tiles)
                                for lo, hi in SIZE_BUCKETS}
                d["by_source"] = {s: summarise(dsel & (gt_src == s), ffd, fcd, n_rej, n_tiles)
                                  for s in sorted(set(gt_src[dsel]))}
                block[dname] = d
            res[cname] = block
        out["results"][mk] = res
    dest = REPO / "reports/r1/object_recall"
    dest.mkdir(parents=True, exist_ok=True)
    stem = Path(args.run).parent.name + "_" + Path(args.run).stem
    (dest / f"{stem}.json").write_text(json.dumps(out, indent=2))
    for mk in args.masks:
        o = out["results"][mk]["conf_0.25"]["non_difficult"]["overall"]
        print(f"{stem} {mk:<13} accept={out['results'][mk]['accept_rate']:.3f} objects={o['n_objects']} "
              f"recall full={o['recall_full']:.3f} cascade={o['recall_cascade']:.3f} ({o['delta_recall_pp']:+.2f} pp) "
              f"on fully-rejected tiles={o['share_on_rejected_tiles_pp']:.1f}% (lost {o['fully_rejected_lost']}, "
              f"missed anyway {o['fully_rejected_missed_anyway']}); partly rejected: recovered "
              f"{o['partly_rejected_recovered']}, lost {o['partly_rejected_lost']}")
    return 0


def hit_any(cache: EvalCache, acc: np.ndarray, classes: list[int], conf: float) -> np.ndarray:
    """Objects (incl. difficult) overlapped at IoU >= 0.5 by a kept detection whose best object it is."""
    keep = merged_keep(cache, acc, cache.det_score >= conf, classes)
    d = np.nonzero(keep & (cache.best_gt >= 0) & (cache.best_iou >= 0.5))[0]
    hit = np.zeros(len(cache.gt_cls), bool)
    hit[cache.best_gt[d]] = True
    return hit


if __name__ == "__main__":
    raise SystemExit(main())
