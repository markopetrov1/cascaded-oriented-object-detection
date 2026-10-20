#!/usr/bin/env python3
"""How much do calibration maps change the set of reachable operating points? (R1.5)

The first submission said that all monotone calibration maps accept exactly the
same tiles at a matched accept rate. That holds for strictly increasing maps
(temperature, Platt with a positive slope). Isotonic regression is only
non-decreasing: it maps runs of different raw scores to one value, and tiles in
such a run are then accepted or rejected together. Any threshold on a
non-decreasing map still accepts a set that some raw threshold also accepts, so
the map cannot add operating points; it can only remove some.

For every gate and map, fitted on one fold of DOTA val and applied to the other
(configs/splits/r1_val_2fold.yaml), this script measures on the report fold:

  * whether the map is order-preserving (no inversions) and strictly so;
  * distinct raw versus calibrated values, the largest tied run and the share of
    tiles that lose their order to a tie;
  * the largest jump in accept rate between neighbouring reachable operating
    points in the working range (accept rate 0.05 to 0.5);
  * the best image-level saving within the 3 pp budget over every reachable
    threshold, raw versus calibrated (an oracle on the report fold, used only to
    size the effect of the ties, never to choose an operating point).

Writes reports/r1/calibration/ties_<task>.json.

    python scripts/r1_calibration_ties.py --task ship
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from r1_eval_cascade import (BB, FOLD_FILE, GATE_GFLOPS, Calibrator, gate_files, load_scores,  # noqa: E402
                             task_classes, tile_info)
from src.r1_eval import EvalCache, evaluate  # noqa: E402

MAPS = ["temperature", "platt", "isotonic"]
G: dict = {}


def _eval(args):
    key, thr, fold = args
    acc = (G["scores"][key] >= thr) & G["fold_tiles"][fold]
    m = evaluate(G["cache"], acc, G["classes"], images=G["fold_images"][fold])
    return key, thr, fold, float(acc[G["fold_tiles"][fold]].mean()), m["mAP50_voc07"]


def tie_stats(raw: np.ndarray, cal: np.ndarray) -> dict:
    o = np.lexsort((cal, raw))                   # sort by raw score, then calibrated
    r, c = raw[o], cal[o]
    inversions = int((np.diff(c) < -1e-12).sum())
    vals, counts = np.unique(c, return_counts=True)
    # tiles whose raw scores differ but whose calibrated values coincide
    lost = 0
    for v, n in zip(vals, counts):
        if n > 1:
            rr = r[c == v]
            if len(np.unique(rr)) > 1:
                lost += n
    return {"n_tiles": len(raw), "distinct_raw": int(len(np.unique(raw))), "distinct_calibrated": int(len(vals)),
            "inversions": inversions, "strictly_increasing": bool(len(vals) == len(np.unique(raw)) and inversions == 0),
            "largest_tied_run": int(counts.max()), "share_tiles_in_ties_pp": 100 * lost / len(raw)}


def reachable_accept_rates(cal: np.ndarray) -> np.ndarray:
    """Accept rates reachable with a threshold on the calibrated score (accept if s >= t)."""
    v = np.unique(cal)
    return np.array([(cal >= t).mean() for t in v] + [0.0])


def max_gap(rates: np.ndarray, lo: float = 0.05, hi: float = 0.5) -> float:
    r = np.sort(rates)
    r = r[(r >= lo - 1e-9) & (r <= hi + 1e-9)]
    return float(np.diff(r).max()) if len(r) > 1 else float("nan")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="ship")
    p.add_argument("--cache", default="runs/r1_cache/multiclass_val.npz")
    p.add_argument("--budget", type=float, default=0.03)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    cache = EvalCache.load(REPO / args.cache)
    classes = task_classes(args.task)
    labels, buckets = tile_info(cache.tiles, classes)
    folds = yaml.safe_load(open(FOLD_FILE))
    img_fold = {im: f for f in ("A", "B") for im in folds[f]}
    fold_images = {f: np.array([img_fold[im] == f for im in cache.images]) for f in ("A", "B")}
    fold_tiles = {f: fold_images[f][cache.tile_image] for f in fold_images}
    gates = gate_files("pillars", args.task)

    scores, stats = {}, {}
    for bb in BB:
        raw = load_scores(gates[bb][0], cache.tiles)
        scores[(bb, "identity", "A")] = scores[(bb, "identity", "B")] = raw
        for fit_f, rep_f in (("A", "B"), ("B", "A")):
            tm, rm = fold_tiles[fit_f], fold_tiles[rep_f]
            for name in MAPS:
                cal = Calibrator(name, raw[tm], labels[tm], buckets[tm])(raw, buckets)
                scores[(bb, name, rep_f)] = cal
                st = tie_stats(raw[rm], cal[rm])
                st["max_accept_gap_pp"] = 100 * max_gap(reachable_accept_rates(cal[rm]))
                st["max_accept_gap_raw_pp"] = 100 * max_gap(reachable_accept_rates(raw[rm]))
                stats.setdefault(bb, {}).setdefault(name, {})[f"fit_{fit_f}_report_{rep_f}"] = st
            # the fitting fold itself, where isotonic ties are exact plateaus
            cal_fit = Calibrator("isotonic", raw[tm], labels[tm], buckets[tm])(raw[tm], buckets[tm])
            stats[bb]["isotonic"][f"fit_{fit_f}_on_fit_fold"] = tie_stats(raw[tm], cal_fit)

    # every reachable threshold on the report fold: all distinct calibrated values in the
    # working range (raw maps: all distinct raw values there)
    # a strictly increasing map reaches exactly the raw operating points, so only the
    # others are evaluated; the strict ones reuse the identity result
    same_as_raw = {(bb, n, rep): stats[bb][n][f"fit_{'B' if rep == 'A' else 'A'}_report_{rep}"]["strictly_increasing"]
                   for bb in BB for n in MAPS for rep in ("A", "B")}
    jobs = []
    for key, s in scores.items():
        if same_as_raw.get(key):
            continue
        rm = fold_tiles[key[2]]
        v = np.unique(s[rm])
        acc = np.array([(s[rm] >= t).mean() for t in v])
        v = v[(acc >= 0.04) & (acc <= 0.6)]
        jobs += [(key, float(t), key[2]) for t in v]
    G.update(cache=cache, classes=classes, scores=scores, fold_tiles=fold_tiles, fold_images=fold_images)
    with Pool(args.workers) as pool:
        res = pool.map(_eval, jobs, chunksize=16)
    full = {f: evaluate(cache, fold_tiles[f], classes, images=fold_images[f])["mAP50_voc07"] for f in ("A", "B")}

    best = {}
    for key, thr, fold, a, m in res:
        if m >= full[fold] - args.budget:
            b = best.setdefault(key, (1.0, None))
            if a < b[0]:
                best[key] = (a, thr)
    det_g = 184.17
    oracle = {}
    for bb in BB:
        for name in ["identity"] + MAPS:
            per = {}
            for f in ("A", "B"):
                src = (bb, "identity", f) if same_as_raw.get((bb, name, f)) else (bb, name, f)
                a, thr = best.get(src, (float("nan"), None))
                per[f] = {"accept": a, "threshold": thr,
                          "saving": 1 - a - GATE_GFLOPS[bb] / det_g,
                          "same_operating_points_as_raw": bool(same_as_raw.get((bb, name, f), name == "identity")),
                          "n_thresholds_in_range": sum(1 for j in jobs if j[0] == src)}
            oracle.setdefault(bb, {})[name] = per
    out = {"task": args.task, "cache": args.cache, "budget_pp": 100 * args.budget,
           "note": "maps fitted on one fold and applied to the other; oracle = best saving within budget "
                   "over every reachable threshold on the report fold, for sizing the effect only",
           "tie_stats": stats, "oracle_best_saving": oracle}
    dest = REPO / "reports/r1/calibration"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"ties_{args.task.replace(' ', '_')}.json").write_text(json.dumps(out, indent=2))
    for bb in BB:
        iso = stats[bb]["isotonic"]["fit_A_report_B"]
        line = " ".join(f"{n}={100*np.mean([oracle[bb][n][f]['saving'] for f in 'AB']):.1f}%"
                        for n in ["identity"] + MAPS)
        print(f"{args.task} {bb:<10} iso distinct {iso['distinct_raw']}->{iso['distinct_calibrated']} "
              f"largest run {iso['largest_tied_run']} tied {iso['share_tiles_in_ties_pp']:.1f}% "
              f"gap {iso['max_accept_gap_pp']:.2f}pp (raw {iso['max_accept_gap_raw_pp']:.2f}) | oracle {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
