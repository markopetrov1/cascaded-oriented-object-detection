"""Cheap per-tile features used by context-adaptive calibration (Pillar 2,
method 5) and learned-threshold MLP (method 6).

Constraint: features must be O(milliseconds per tile) to compute. The whole
point is that they cost ~nothing relative to a neural-network gate, so they can
inform threshold selection without paying real compute. We extract:

- mean & std of each RGB channel
- brightness percentiles (p10, p50, p90)
- Sobel-edge density (mean abs gradient magnitude)
- entropy of the grayscale histogram (texture proxy)

The output is a fixed-shape ``(N, 9)`` matrix, which is enough to feed an MLP
or a per-bucket classifier without overfitting on typical val-set sizes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

FEATURE_NAMES = [
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std",
    "brightness_p10",
    "brightness_p50",
    "brightness_p90",
    "sobel_edge_density",
    "gray_entropy",
]


def compute_tile_features(image_bgr: np.ndarray) -> np.ndarray:
    """Compute the 9-dim feature vector for a single BGR tile."""
    import cv2

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR image, got shape {image_bgr.shape}")
    img = image_bgr.astype(np.float32) / 255.0
    rgb_mean = img.mean(axis=(0, 1))[::-1]  # B,G,R -> R,G,B
    rgb_std = float(img.std())

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    p10, p50, p90 = (float(np.percentile(gray, q)) / 255.0 for q in (10, 50, 90))

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    edge_density = float(edge_mag.mean()) / 255.0

    hist, _ = np.histogram(gray, bins=32, range=(0, 256), density=True)
    hist = hist + 1e-12
    hist = hist / hist.sum()
    entropy = float(-(hist * np.log(hist)).sum())

    return np.asarray(
        [
            float(rgb_mean[0]),
            float(rgb_mean[1]),
            float(rgb_mean[2]),
            rgb_std,
            p10,
            p50,
            p90,
            edge_density,
            entropy,
        ],
        dtype=np.float32,
    )


def extract_features_for_split(
    data_root: str | Path,
    split: str,
    out_jsonl: str | Path,
    image_suffix: str = ".png",
) -> Path:
    """Walk ``data/processed/<run>/images/<split>/`` and emit one row per tile.

    Output JSONL row::

        {"tile_id": "...", "split": "val", "features": {"rgb_mean_r": ..., ...}}
    """
    import cv2

    images_dir = Path(data_root) / "images" / split
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for img_path in sorted(images_dir.glob(f"*{image_suffix}")):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[tile_features] WARNING: skip unreadable {img_path}")
                continue
            vec = compute_tile_features(img)
            handle.write(
                json.dumps(
                    {
                        "tile_id": img_path.stem,
                        "split": split,
                        "features": {name: float(value) for name, value in zip(FEATURE_NAMES, vec)},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            n += 1
    print(f"[tile_features] wrote {n} rows to {out_path}")
    return out_path


def load_features_jsonl(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Read a features JSONL into ``(tile_ids, X)`` where ``X`` is ``(N, 9)``."""
    ids: list[str] = []
    rows: list[list[float]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        ids.append(str(item["tile_id"]))
        feats = item.get("features", {})
        rows.append([float(feats.get(name, 0.0)) for name in FEATURE_NAMES])
    return ids, np.asarray(rows, dtype=np.float32)


def join_features_to_metadata(
    features_jsonl: str | Path,
    metadata_jsonl: str | Path,
) -> dict[str, dict]:
    """Build a tile_id -> {features..., gsd, imagesource, is_positive} map.

    Useful for context-adaptive calibration which keys thresholds on
    ``imagesource`` / ``gsd``.
    """
    out: dict[str, dict] = {}
    for line in Path(features_jsonl).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["tile_id"])] = {"features": row.get("features", {})}
    for line in Path(metadata_jsonl).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tid = str(row["tile_id"])
        if tid not in out:
            continue
        out[tid].update(
            imagesource=row.get("imagesource"),
            gsd=row.get("gsd"),
            is_positive=int(row.get("is_positive", 0)),
            split=row.get("split"),
        )
    return out


def gsd_bucket(gsd: float | str | None) -> str:
    if gsd is None:
        return "unknown"
    try:
        v = float(gsd)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(v):
        return "unknown"
    if v < 0.15:
        return "very_high"  # < 15 cm
    if v < 0.30:
        return "high"
    if v < 0.60:
        return "medium"
    if v < 1.50:
        return "low"
    return "very_low"


def features_to_matrix(
    tile_ids: Iterable[str],
    feature_map: dict[str, dict],
    drop_missing: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Project a list of tile IDs onto the feature matrix; drop ones not present."""
    kept_ids: list[str] = []
    rows: list[list[float]] = []
    for tid in tile_ids:
        entry = feature_map.get(tid)
        if entry is None or "features" not in entry:
            if drop_missing:
                continue
            rows.append([0.0] * len(FEATURE_NAMES))
            kept_ids.append(tid)
            continue
        feats = entry["features"]
        rows.append([float(feats.get(name, 0.0)) for name in FEATURE_NAMES])
        kept_ids.append(tid)
    return kept_ids, np.asarray(rows, dtype=np.float32)
