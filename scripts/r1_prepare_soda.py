#!/usr/bin/env python3
"""Prepare the SODA-A validation split as an independent test set (R2.6a).

SODA-A (Cheng et al., 2023; 576 validation scenes of about 4800 x 2750 px, OBB,
CC BY-NC 4.0) is labelled by a different team under a different protocol from
DOTA. This script, with no images written to disk:

  * converts each scene's JSON to DOTA labelTxt with DOTA class names
    (airplane -> plane, small-vehicle, large-vehicle, ship, storage-tank,
    swimming-pool, helicopter; container and windmill have no DOTA class and
    are kept under their own names, which the evaluator skips);
  * writes SODA-A's class-agnostic "ignore" regions (instances above its size
    limit) to a separate file per scene. The evaluator drops detections centred
    in an ignore region, and tiles touching one are left out of the tile-label
    statistics (p+, TPR, FPR), because their class is unknown;
  * computes the 1024-px, 200-px-overlap tile windows with DOTA's tiler and a
    tiles.jsonl in the same layout as DOTA's (an object counts on a tile when at
    least 10 % of its area is inside).

    python scripts/r1_prepare_soda.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import shapely

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.datasets.tiling import tile_windows  # noqa: E402

RAW = REPO / "data/raw/SODA-A"
OUT = REPO / "data/processed/soda_a"
NAMES = {0: "plane", 1: "helicopter", 2: "small-vehicle", 3: "large-vehicle", 4: "ship", 5: "container",
         6: "storage-tank", 7: "swimming-pool", 8: "windmill"}
IGNORE = 9
MIN_FRAC = 0.1


def main() -> int:
    (OUT / "val/labelTxt").mkdir(parents=True, exist_ok=True)
    (OUT / "val/ignore").mkdir(parents=True, exist_ok=True)
    (OUT / "metadata").mkdir(parents=True, exist_ok=True)
    rows, counts, n_ign_tiles = [], Counter(), 0
    for f in sorted((RAW / "Annotations/val").glob("*.json")):
        d = json.load(open(f))
        stem = Path(d["images"]["file_name"]).stem
        W, H = int(d["images"]["width"]), int(d["images"]["height"])
        objs, ign = [], []
        for a in d["annotations"]:
            pts = np.array(a["poly"], float).reshape(-1, 2)
            if len(pts) != 4:
                continue
            (ign if a["category_id"] == IGNORE else objs).append((a["category_id"], pts))
        lines = ["imagesource:SODA-A", "gsd:null"]
        lines += [" ".join(f"{v:.1f}" for v in p.reshape(-1)) + f" {NAMES[c]} 0" for c, p in objs]
        (OUT / f"val/labelTxt/{stem}.txt").write_text("\n".join(lines) + "\n")
        (OUT / f"val/ignore/{stem}.txt").write_text(
            "".join(" ".join(f"{v:.1f}" for v in p.reshape(-1)) + "\n" for _, p in ign))
        counts.update(NAMES[c] for c, _ in objs)
        polys = shapely.convex_hull(shapely.polygons([p for _, p in objs])) if objs else np.array([])
        ipolys = shapely.convex_hull(shapely.polygons([p for _, p in ign])) if ign else np.array([])
        area = shapely.area(polys) if objs else np.zeros(0)
        iarea = shapely.area(ipolys) if ign else np.zeros(0)
        for k, w in enumerate(tile_windows(W, H, 1024, 200)):
            box = shapely.box(w.x, w.y, w.x + w.width, w.y + w.height)
            cc = Counter()
            if objs:
                frac = shapely.area(shapely.intersection(polys, box)) / np.maximum(area, 1e-9)
                for (c, _), fr in zip(objs, frac):
                    if fr >= MIN_FRAC:
                        cc[NAMES[c].replace("-", " ")] += 1
            has_ign = bool(ign) and bool((shapely.area(shapely.intersection(ipolys, box)) /
                                          np.maximum(iarea, 1e-9) >= MIN_FRAC).any())
            n_ign_tiles += has_ign
            rows.append({"tile_id": f"{stem}__x{w.x}_y{w.y}_w{w.width}_h{w.height}", "source_stem": stem,
                         "source_image": d["images"]["file_name"], "split": "val", "x": w.x, "y": w.y,
                         "width": w.width, "height": w.height, "imagesource": "SODA-A",
                         "class_counts": dict(cc), "has_ignore": has_ign, "tile_index": k})
    with open(OUT / "metadata/tiles.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summary = {"scenes": len({r["source_stem"] for r in rows}), "tiles": len(rows),
               "tiles_with_ignore": n_ign_tiles, "objects": dict(counts)}
    for task in ("ship", "plane", "small vehicle"):
        lab = [bool(r["class_counts"].get(task)) for r in rows if not r["has_ignore"]]
        summary[f"p_plus_{task.replace(' ', '_')}"] = float(np.mean(lab))
    (OUT / "metadata/summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
