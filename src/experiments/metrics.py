from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.utils.paths import ensure_dir, write_json


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float | str | int | None]:
    flat: dict[str, float | str | int | None] = {}
    for key, value in metrics.items():
        clean_key = key.replace("metrics/", "").replace("(", "").replace(")", "").replace("/", "_")
        if isinstance(value, (int, float, str)) or value is None:
            flat[clean_key] = value
    return flat


def f1_from_precision_recall(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def save_metric_files(row: dict[str, Any], json_path: str | Path, csv_path: str | Path) -> None:
    write_json(row, json_path)
    ensure_dir(Path(csv_path).parent)
    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def extract_ultralytics_metrics(result: Any) -> dict[str, Any]:
    raw = getattr(result, "results_dict", None)
    if raw is None and hasattr(result, "trainer"):
        raw = getattr(result.trainer, "metrics", None)
    raw = raw or {}
    metrics = flatten_metrics(dict(raw))

    box = getattr(result, "box", None)
    if box is not None:
        for attr, key in (
            ("map50", "map50"),
            ("map", "map50_95"),
            ("mp", "precision"),
            ("mr", "recall"),
        ):
            value = getattr(box, attr, None)
            if value is not None:
                try:
                    metrics[key] = float(value)
                except TypeError:
                    pass
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    if isinstance(precision, (int, float)) and isinstance(recall, (int, float)):
        metrics["f1"] = f1_from_precision_recall(float(precision), float(recall))
    return metrics


def extract_per_class_ap(result: Any, names: dict[int, str] | list[str] | None = None) -> list[dict[str, Any]]:
    box = getattr(result, "box", None)
    if box is None:
        return []
    ap = getattr(box, "ap", None)
    ap50 = getattr(box, "ap50", None)
    class_indices = getattr(box, "ap_class_index", None)
    if ap is None:
        return []
    rows: list[dict[str, Any]] = []
    for idx, value in enumerate(ap):
        class_id = int(class_indices[idx]) if class_indices is not None and idx < len(class_indices) else idx
        if isinstance(names, dict):
            class_name = names.get(class_id, str(class_id))
        elif isinstance(names, list) and class_id < len(names):
            class_name = names[class_id]
        else:
            class_name = str(class_id)
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "ap50_95": float(value),
                "ap50": float(ap50[idx]) if ap50 is not None and idx < len(ap50) else None,
            }
        )
    return rows
