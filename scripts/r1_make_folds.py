#!/usr/bin/env python3
"""Split DOTA-1.5 val into two folds of whole source images for cross-fitting.

Calibration maps, the gate backbone and the operating point are chosen on one
fold and reported on the other, then the folds swap (reviewer concerns R1.4 and
R2.5: the first submission fitted and reported on the same tiles). Splitting by
source image keeps the overlapping tiles of a scene together. Within each sensor
source the images are assigned greedily, largest first, to whichever fold is
furthest below half of the remaining budget, where an image's weight is its tile
count plus its positive tiles for each gating task, so both folds see each
sensor and roughly half of every task's positives.

Writes configs/splits/r1_val_2fold.yaml.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
META = REPO / "data/processed/dota_multiclass/metadata/tiles.jsonl"
OUT = REPO / "configs/splits/r1_val_2fold.yaml"
SEED = 67


def main() -> int:
    tiles = defaultdict(int)
    pos = defaultdict(lambda: defaultdict(int))
    source = {}
    for line in open(META):
        r = json.loads(line)
        if r["split"] != "val":
            continue
        img = r["source_stem"]
        tiles[img] += 1
        source[img] = r.get("imagesource") or "unknown"
        for c in r.get("class_counts", {}):
            pos[img][c] += 1
        if r.get("class_counts"):
            pos[img]["any"] += 1
    keys = sorted({c for p in pos.values() for c in p})
    totals = {k: sum(pos[i][k] for i in tiles) for k in keys}

    rng = random.Random(SEED)
    folds = {"A": [], "B": []}
    load = {f: defaultdict(float) for f in folds}
    for src in sorted(set(source.values())):
        imgs = [i for i in tiles if source[i] == src]
        rng.shuffle(imgs)
        imgs.sort(key=lambda i: -(tiles[i] + sum(pos[i].values())))
        for img in imgs:
            # Normalised deficit: tiles and each task's positives count equally.
            def deficit(f):
                d = (sum(tiles[i] for i in tiles) / 2 - load[f]["tiles"]) / max(sum(tiles.values()), 1)
                d += sum((totals[k] / 2 - load[f][k]) / max(totals[k], 1) for k in keys)
                return d
            f = max(folds, key=deficit)
            folds[f].append(img)
            load[f]["tiles"] += tiles[img]
            for k in keys:
                load[f][k] += pos[img][k]

    # The greedy pass leaves the lumpy tasks unbalanced (a single harbour scene can
    # hold dozens of ship tiles), so polish it with a local search over moves and
    # swaps of whole images within a sensor source, minimising the worst relative
    # imbalance over tiles and every task with at least 20 positive tiles.
    keys_bal = [k for k in keys if totals[k] >= 20] + ["tiles"]
    tot = dict(totals, tiles=sum(tiles.values()))
    weight = {i: dict({k: pos[i][k] for k in keys}, tiles=tiles[i]) for i in tiles}
    side = {i: f for f, v in folds.items() for i in v}

    def objective() -> float:
        a = defaultdict(float)
        for i, f in side.items():
            if f == "A":
                for k in keys_bal:
                    a[k] += weight[i][k]
        return max(abs(a[k] / tot[k] - 0.5) for k in keys_bal)

    best = objective()
    by_src = defaultdict(list)
    for i in tiles:
        by_src[source[i]].append(i)
    for _ in range(40000):
        src = rng.choice(sorted(by_src))
        i = rng.choice(by_src[src])
        if rng.random() < 0.5:
            j = None
            side[i] = "B" if side[i] == "A" else "A"
        else:
            j = rng.choice(by_src[src])
            if side[i] == side[j]:
                continue
            side[i], side[j] = side[j], side[i]
        val = objective()
        if val <= best:
            best = val
        else:  # undo
            if j is None:
                side[i] = "B" if side[i] == "A" else "A"
            else:
                side[i], side[j] = side[j], side[i]
    folds = {f: sorted(i for i in side if side[i] == f) for f in ("A", "B")}
    load = {f: defaultdict(float) for f in folds}
    for f, v in folds.items():
        for i in v:
            load[f]["tiles"] += tiles[i]
            for k in keys:
                load[f][k] += pos[i][k]
    print(f"worst relative imbalance over {len(keys_bal)} balanced quantities: {best:.3f}")

    summary = {f: {"images": len(v), "tiles": int(load[f]["tiles"]),
                   "positives": {k: int(load[f][k]) for k in keys},
                   "sources": {s: sum(1 for i in v if source[i] == s) for s in sorted(set(source.values()))}}
               for f, v in folds.items()}
    OUT.write_text(yaml.safe_dump({"seed": SEED, "split_by": "source image, stratified by imagesource",
                                   "summary": summary,
                                   "A": sorted(folds["A"]), "B": sorted(folds["B"])}, sort_keys=False))
    for f, s in summary.items():
        print(f, s["images"], "images", s["tiles"], "tiles", s["sources"])
        print("   positives:", {k: s["positives"][k] for k in keys})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
