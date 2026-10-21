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

The gate decisions are the held-out operating point of the MobileNetV3-large
ship gate (reports/r1/cascade/sweep/ship_masks.npz, each tile decided by the
threshold chosen on the other fold). The scene is chosen to be representative
rather than flattering: among validation scenes with at least 20 tiles, 4
ship-bearing tiles and 50 ships, the one whose share of rejected ship-bearing
tiles is closest to the share over all of validation. Its counts go to
reports/r1/teaser_facts.json for the caption.
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

SPLIT = "val"
CLASS = "ship"
MASKS = "reports/r1/cascade/sweep/ship_masks.npz"
RAW = Path("data/raw/DOTA") / SPLIT
METADATA = "data/processed/dota_ships/metadata/tiles.jsonl"
DISPLAY_W = 1600


def choose_scene() -> tuple[str, dict, dict]:
    z = np.load(MASKS)
    acc = dict(zip(z["tiles"].tolist(), z["risk_pooled"].tolist()))
    rows = [json.loads(l) for l in open(METADATA)]
    rows = [r for r in rows if r["split"] == SPLIT and r["tile_id"] in acc]
    pos_all = [r for r in rows if CLASS in r.get("class_counts", {})]
    rate_all = sum(not acc[r["tile_id"]] for r in pos_all) / len(pos_all)
    by = {}
    for r in rows:
        by.setdefault(r["source_stem"], []).append(r)
    best = None
    for stem, rs in by.items():
        pos = [r for r in rs if CLASS in r.get("class_counts", {})]
        ships = sum(1 for _ in load_gt_polygons(stem))
        if len(rs) < 20 or len(pos) < 4 or ships < 50:
            continue
        rate = sum(not acc[r["tile_id"]] for r in pos) / len(pos)
        key = (abs(rate - rate_all), -ships)
        if best is None or key < best[0]:
            best = (key, stem, rate)
    return best[1], acc, {"overall_rejected_positive_share": rate_all, "scene_rejected_positive_share": best[2]}


def load_scene():
    img = Image.open(RAW / "images" / f"{STEM}.png").convert("RGB")
    w, h = img.size
    scale = DISPLAY_W / w
    img = img.resize((DISPLAY_W, int(h * scale)), Image.LANCZOS)
    return np.asarray(img), scale, (w, h)


def load_gt_polygons(stem=None):
    polys = []
    path = RAW / "labelTxt" / f"{stem or STEM}.txt"
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 10:
            continue  # imagesource: / gsd: header lines
        if parts[8] != CLASS:
            continue
        coords = [float(v) for v in parts[:8]]
        polys.append(np.array(coords).reshape(4, 2))
    return polys


def load_tiles(acc):
    tiles = []
    with open(METADATA) as fh:
        for line in fh:
            row = json.loads(line)
            if row["source_stem"] != STEM or row["split"] != SPLIT:
                continue
            if row["tile_id"] not in acc:
                continue
            tiles.append({
                "x": row["x"], "y": row["y"],
                "w": row["width"], "h": row["height"],
                "positive": CLASS in row.get("class_counts", {}),
                "accepted": bool(acc[row["tile_id"]]),
            })
    return tiles


def main() -> int:
    global STEM
    use_paper_style()
    STEM, acc, rates = choose_scene()
    scene, scale, (w0, h0) = load_scene()
    polys = load_gt_polygons()
    tiles = load_tiles(acc)

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
    save(fig, "teaser_r1", outdir="reports/r1/figures")
    facts = {"scene": STEM, "ships": len(polys), "tiles": len(tiles), "skipped": saved,
             "positive_tiles": counts[(True, True)] + counts[(False, True)],
             "rejected_positive_tiles": counts[(False, True)], **rates, "masks": MASKS}
    Path("reports/r1/teaser_facts.json").write_text(json.dumps(facts, indent=2))
    print(f"  scene {STEM}: {len(tiles)} tiles, {counts}, {rates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
