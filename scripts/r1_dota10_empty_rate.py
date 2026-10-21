#!/usr/bin/env python3
"""Empty-tile rate of OAN's gating task under DOTA-1.0's annotation scope (F7).

OAN gates DOTA-1.0, whose labels omit two things DOTA-1.5 added to the same
images: the container-crane class and instances smaller than about 10 px. The
empty-tile rate measured on DOTA-1.5 labels is therefore a lower bound on the
DOTA-1.0 rate. This recomputes the rate on our tiling (1024 px, 200 px overlap,
an object counts on a tile when at least 10 % of its area is inside, as in
src/datasets/tiling.py) after dropping container cranes and objects whose
oriented box is smaller than 10 px, under two readings of "smaller": the longer
side, and the square root of the area. It approximates DOTA-1.0; it is not
DOTA-1.0's own labels.

Writes reports/r1/prior_work/dota10_empty_rate.json.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import shapely

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.r1_eval import to_polygons  # noqa: E402

META = REPO / "data/processed/dota_multiclass/metadata/tiles.jsonl"
MIN_FRAC = 0.1


def load_objects(split: str, stem: str):
    cls, pts = [], []
    for line in open(REPO / f"data/raw/DOTA/{split}/labelTxt/{stem}.txt"):
        v = line.split()
        if len(v) < 9 or ":" in line:
            continue
        pts.append(np.array(v[:8], float).reshape(4, 2)); cls.append(v[8])
    return np.array(cls), (np.stack(pts) if pts else np.zeros((0, 4, 2)))


def main() -> int:
    tiles = defaultdict(list)
    for line in open(META):
        r = json.loads(line)
        tiles[(r["split"], r["source_stem"])].append(r)
    out = {}
    for split in ("train", "val"):
        n = 0
        empty = {"dota15_all": 0, "no_crane": 0, "dota10_longside": 0, "dota10_sqrtarea": 0}
        for (s, stem), rows in tiles.items():
            if s != split:
                continue
            cls, pts = load_objects(split, stem)
            polys = to_polygons(pts) if len(pts) else np.array([])
            area = shapely.area(polys) if len(pts) else np.zeros(0)
            side = np.linalg.norm(np.roll(pts, -1, axis=1) - pts, axis=2).max(1) if len(pts) else np.zeros(0)
            keep = {"dota15_all": np.ones(len(cls), bool), "no_crane": cls != "container-crane"}
            keep["dota10_longside"] = keep["no_crane"] & (side >= 10)
            keep["dota10_sqrtarea"] = keep["no_crane"] & (np.sqrt(area) >= 10)
            for r in rows:
                n += 1
                if len(pts):
                    box = shapely.box(r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"])
                    frac = shapely.area(shapely.intersection(polys, box)) / np.maximum(area, 1e-9)
                    on = frac >= MIN_FRAC
                else:
                    on = np.zeros(0, bool)
                for k in empty:
                    empty[k] += int(not (on & keep[k]).any())
        out[split] = {"n_tiles": n, **{f"empty_rate_{k}": v / n for k, v in empty.items()}}
        print(split, {k: round(v, 4) for k, v in out[split].items()})
    dest = REPO / "reports/r1/prior_work"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "dota10_empty_rate.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
