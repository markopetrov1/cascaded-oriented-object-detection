#!/usr/bin/env python3
"""Check whether SODA-A validation scenes duplicate DOTA-1.5 scenes (R2.6a).

SODA-A and most of DOTA are both Google Earth imagery, so the same place could
appear in both. Each scene is reduced to a perceptual hash (DCT of a 32 x 32
grey thumbnail, 8 x 8 low-frequency block against its median, 64 bits) for the
whole scene and for its central half, and every SODA-A scene is compared with
every DOTA-1.5 train and val scene. Pairs within a small Hamming distance are
listed for visual inspection. This finds re-used scenes and large shared
crops; it cannot rule out small partial overlaps.

Writes reports/r1/soda_dota_overlap.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.fft import dctn

REPO = Path(__file__).resolve().parents[1]
THRESH = 10


def phash(img: np.ndarray) -> np.ndarray:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(float)
    d = dctn(g, norm="ortho")[:8, :8].ravel()
    return d > np.median(d[1:])


def hashes(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        return None
    h, w = img.shape[:2]
    centre = img[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    return phash(img), phash(centre)


def main() -> int:
    soda = sorted((REPO / "data/raw/SODA-A/Images/val").glob("*.jpg"))
    dota = sorted((REPO / "data/raw/DOTA/train/images").glob("*.png")) + sorted((REPO / "data/raw/DOTA/val/images").glob("*.png"))
    hs = {p.stem: hashes(p) for p in soda}
    hd = {}
    for i, p in enumerate(dota):
        hd[f"{p.parent.parent.name}/{p.stem}"] = hashes(p)
        if i % 500 == 0:
            print(f"  DOTA {i}/{len(dota)}", flush=True)
    S = np.stack([np.concatenate(v) for v in hs.values() if v is not None])
    D = np.stack([np.concatenate(v) for v in hd.values() if v is not None])
    s_names = [k for k, v in hs.items() if v is not None]
    d_names = [k for k, v in hd.items() if v is not None]
    full = (S[:, None, :64] != D[None, :, :64]).sum(-1)
    cen = (S[:, None, 64:] != D[None, :, 64:]).sum(-1)
    dist = np.minimum(full, cen)
    pairs = [(s_names[i], d_names[j], int(full[i, j]), int(cen[i, j]))
             for i, j in zip(*np.nonzero(dist <= THRESH))]
    out = {"soda_scenes": len(s_names), "dota_scenes": len(d_names), "threshold_bits": THRESH,
           "min_distance_per_soda_scene_median": float(np.median(dist.min(1))),
           "min_distance_overall": int(dist.min()), "pairs": pairs}
    (REPO / "reports/r1/soda_dota_overlap.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "pairs"}), "pairs:", len(pairs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
