#!/usr/bin/env python3
"""Teaser: what a tile gate can and cannot save, on one real scene.

Left panel shows the scene with its ground-truth ships, which is the sparsity
the cascade exploits. Right panel shows the same scene tiled, with every tile
coloured by what the gate did with it, using the same four categories as
reports/figures/savings_provenance.pdf so the reader meets the decomposition
here first and recognises it later.

The point the panel has to make in one look: the green tiles are the only
compute a lossless gate can save, and the yellow tiles are savings bought by
throwing detections away.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from figstyle import INK, INK_MUTED, PALETTE, TEXT_WIDTH, save, use_paper_style

import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None

STEM = "P1386"
SPLIT = "val"
CLASS = "ship"
GATE_SCORES = "reports/gate_scores/gate_mbv3large_val.jsonl"
# Operating point of the mbv3large gate at the 3 pp mAP tolerance, read off
# reports/cascade/gate_mbv3large_identity.json.
THRESHOLD = 0.95
RAW = Path("data/raw/DOTA") / SPLIT
METADATA = "data/processed/dota_ships/metadata/tiles.jsonl"
DISPLAY_W = 1600


def load_scene():
    img = Image.open(RAW / "images" / f"{STEM}.png").convert("RGB")
    w, h = img.size
    scale = DISPLAY_W / w
    img = img.resize((DISPLAY_W, int(h * scale)), Image.LANCZOS)
    return np.asarray(img), scale, (w, h)


def load_gt_polygons():
    polys = []
    path = RAW / "labelTxt" / f"{STEM}.txt"
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 10:
            continue  # imagesource: / gsd: header lines
        if parts[8] != CLASS:
            continue
        coords = [float(v) for v in parts[:8]]
        polys.append(np.array(coords).reshape(4, 2))
    return polys


def load_tiles():
    tiles = []
    scores = {}
    with open(GATE_SCORES) as fh:
        for line in fh:
            row = json.loads(line)
            scores[row["tile_id"]] = row["prob"]
    with open(METADATA) as fh:
        for line in fh:
            row = json.loads(line)
            if row["source_stem"] != STEM or row["split"] != SPLIT:
                continue
            prob = scores.get(row["tile_id"])
            if prob is None:
                continue
            tiles.append({
                "x": row["x"], "y": row["y"],
                "w": row["width"], "h": row["height"],
                "positive": CLASS in row.get("class_counts", {}),
                "accepted": prob >= THRESHOLD,
            })
    return tiles


def main() -> int:
    use_paper_style()
    scene, scale, (w0, h0) = load_scene()
    polys = load_gt_polygons()
    tiles = load_tiles()

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, TEXT_WIDTH * h0 / w0 / 2 + 0.55))

    # ---- left: the scene and its objects --------------------------------
    axl.imshow(scene)
    for p in polys:
        axl.add_patch(mpatches.Polygon(p * scale, closed=True, fill=False,
                                       edgecolor=PALETTE[1], linewidth=0.9, zorder=3))
    axl.set_title(f"(a) {len(polys)} ships in a {w0}$\\times${h0} scene", loc="left")

    # ---- right: what the gate did ---------------------------------------
    axr.imshow(scene, alpha=0.45)
    categories = {
        (True, True): (PALETTE[0], "detector runs, tile has ships"),
        (True, False): (PALETTE[3], "detector runs, tile empty (leak)"),
        (False, True): (PALETTE[1], "skipped, tile had ships (recall lost)"),
        (False, False): (PALETTE[2], "skipped, tile empty (true saving)"),
    }
    counts = {k: 0 for k in categories}
    # Tiles overlap by 200 px, so drawing them at full extent stacks the fills
    # into unreadable bands. Inset by half the overlap: each drawn cell is then
    # the tile's own stride footprint and the grid reads cleanly.
    inset = 100 * scale
    for t in tiles:
        key = (t["accepted"], t["positive"])
        counts[key] += 1
        colour, _ = categories[key]
        axr.add_patch(mpatches.Rectangle(
            (t["x"] * scale + inset, t["y"] * scale + inset),
            t["w"] * scale - 2 * inset, t["h"] * scale - 2 * inset,
            facecolor=colour, alpha=0.30, edgecolor=colour, linewidth=1.1, zorder=3))

    saved = counts[(False, True)] + counts[(False, False)]
    axr.set_title(f"(b) gate skips {saved}/{len(tiles)} tiles", loc="left")

    handles = [mpatches.Patch(facecolor=c, alpha=0.55, edgecolor=c,
                              label=f"{lab}  ({counts[k]})")
               for k, (c, lab) in categories.items()]
    axr.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, fontsize=6, handlelength=1.4)

    for ax in (axl, axr):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(wspace=0.04)
    save(fig, "teaser")
    print(f"  scene {STEM}: {len(tiles)} tiles, {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
