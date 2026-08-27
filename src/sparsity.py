"""Tile-level sparsity statistics for cascade research.

The cascade thesis turns on object sparsity, so the per-class positive rate and
the distributions of objects-per-tile / object-size / GSD / sensor are first-class
artefacts (not just supporting numbers). This module reads ``metadata/tiles.jsonl``
emitted by :func:`src.datasets.dota.prepare_dota` and produces a JSON + CSV report.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.utils.paths import ensure_dir, write_json

SIZE_BUCKETS = [
    ("<32px", 0, 32),
    ("32-96px", 32, 96),
    ("96-256px", 96, 256),
    (">256px", 256, float("inf")),
]


def _bucket_size(size_px: float) -> str:
    for label, lo, hi in SIZE_BUCKETS:
        if lo <= size_px < hi:
            return label
    return SIZE_BUCKETS[-1][0]


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _summarize(tiles: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(tiles)
    if n == 0:
        return {"n_tiles": 0, "positive_rate": 0.0}
    positives = [t for t in tiles if t.get("is_positive")]
    n_pos = len(positives)
    obj_per_pos = [t["num_positives"] for t in positives if t.get("num_positives", 0) > 0]
    sizes = [t["max_obj_size_px"] for t in positives if t.get("max_obj_size_px") is not None]
    bucket_counts = Counter(_bucket_size(s) for s in sizes)

    summary: dict[str, Any] = {
        "n_tiles": n,
        "n_positive_tiles": n_pos,
        "positive_rate": _safe_div(n_pos, n),
        "mean_objects_per_positive_tile": round(mean(obj_per_pos), 3) if obj_per_pos else 0.0,
        "median_objects_per_positive_tile": float(median(obj_per_pos)) if obj_per_pos else 0.0,
        "max_obj_size_buckets": dict(bucket_counts),
    }
    return summary


def compute_sparsity_stats(metadata_jsonl: str | Path, out_path: str | Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(metadata_jsonl).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty or missing metadata at {metadata_jsonl}")

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_imagesource: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        by_split[row.get("split", "unknown")].append(row)
        src = row.get("imagesource") or "unknown"
        by_imagesource[str(src)].append(row)
        for cls in (row.get("class_counts") or {}).keys():
            by_class[cls].append(row)

    gsds = [r["gsd"] for r in rows if isinstance(r.get("gsd"), (int, float))]
    gsd_summary: dict[str, Any] = {}
    if gsds:
        gsd_summary = {
            "min": float(min(gsds)),
            "max": float(max(gsds)),
            "mean": round(mean(gsds), 4),
            "median": float(median(gsds)),
            "n_with_gsd": len(gsds),
        }

    report: dict[str, Any] = {
        "global": _summarize(rows),
        "per_split": {split: _summarize(items) for split, items in sorted(by_split.items())},
        "per_imagesource": {src: _summarize(items) for src, items in sorted(by_imagesource.items())},
        "per_class": {cls: _summarize(items) for cls, items in sorted(by_class.items())},
        "gsd_distribution": gsd_summary,
        "n_source_images": len({r["source_stem"] for r in rows}),
    }

    write_json(report, out_path)

    csv_path = Path(out_path).with_suffix(".csv")
    ensure_dir(csv_path.parent)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scope", "key", "n_tiles", "n_positive_tiles", "positive_rate"])
        writer.writerow(
            ["global", "all", report["global"]["n_tiles"], report["global"]["n_positive_tiles"], report["global"]["positive_rate"]]
        )
        for scope_name, scope in (
            ("split", report["per_split"]),
            ("imagesource", report["per_imagesource"]),
            ("class", report["per_class"]),
        ):
            for key, summary in scope.items():
                writer.writerow(
                    [scope_name, key, summary["n_tiles"], summary.get("n_positive_tiles", 0), summary["positive_rate"]]
                )
    return report
