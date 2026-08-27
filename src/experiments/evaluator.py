from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.experiments.metrics import (
    extract_per_class_ap,
    extract_ultralytics_metrics,
    save_metric_files,
)
from src.utils.hardware import collect_environment
from src.utils.paths import ensure_dir, iter_images, project_root, read_yaml, write_json
from src.utils.reproducibility import build_repro_metadata, write_repro_metadata
from src.utils.ultralytics_env import configure_ultralytics_settings


def _resolve_dataset_split(data_yaml: str | Path, split: str) -> list[Path]:
    data_path = Path(data_yaml)
    config = read_yaml(data_path)
    root = Path(config.get("path", "."))
    if not root.is_absolute():
        root = (project_root() / root).resolve()
    split_value = config.get(split) or config.get("val")
    if split_value is None:
        return []
    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = root / split_path
    if split_path.is_file():
        return [Path(line.strip()) for line in split_path.read_text().splitlines() if line.strip()]
    return iter_images(split_path)


def _names_from_yaml(data_yaml: str | Path) -> dict[int, str]:
    names = read_yaml(data_yaml).get("names", {})
    if isinstance(names, list):
        return {idx: name for idx, name in enumerate(names)}
    return {int(idx): str(name) for idx, name in names.items()}


def write_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(
    weights: str | Path,
    data_yaml: str | Path,
    task: str = "detect",
    split: str = "val",
    imgsz: int = 640,
    batch: int = 16,
    device: str | int | None = None,
    output_dir: str | Path = "results/metrics",
    run_name: str | None = None,
    sample_count: int = 12,
    precision: str | None = None,
    exported_model: str | Path | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    configure_ultralytics_settings()
    from ultralytics import YOLO

    weights_path = Path(weights)
    run_name = run_name or weights_path.parent.parent.name or weights_path.stem
    out_dir = ensure_dir(Path(output_dir) / run_name)
    model = YOLO(str(weights), task=task)
    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "plots": True,
        "save_json": True,
        "project": str(out_dir / "val"),
        "name": split,
        "exist_ok": True,
    }
    if device not in (None, "auto", ""):
        kwargs["device"] = device
    result = model.val(**kwargs)
    metrics = extract_ultralytics_metrics(result)
    row = {
        "run_name": run_name,
        "weights": str(weights),
        "data": str(data_yaml),
        "task": task,
        "split": split,
        **metrics,
    }
    if precision or exported_model:
        row["export_precision"] = precision or "fp32"
        row["exported_model"] = str(exported_model) if exported_model is not None else ""
    save_metric_files(row, out_dir / "metrics.json", out_dir / "metrics.csv")
    per_class_rows = extract_per_class_ap(result, _names_from_yaml(data_yaml))
    for item in per_class_rows:
        item["run_name"] = run_name
        item["task"] = task
        item["split"] = split
    write_rows(per_class_rows, out_dir / "per_class_ap.csv")
    write_json(collect_environment("."), out_dir / "environment.json")
    if precision or exported_model:
        write_repro_metadata(
            build_repro_metadata(
                weights=weights,
                exported_model=exported_model,
                data_yaml=data_yaml,
                task=task,
                split=split,
                imgsz=imgsz,
                batch=batch,
                device=device,
                precision=precision or "fp32",
                command=command,
            ),
            out_dir / "evaluation_metadata.json",
        )

    samples = _resolve_dataset_split(data_yaml, split)[:sample_count]
    if samples:
        pred_dir = ensure_dir(Path("results/predictions") / run_name / split)
        for image in samples:
            model.predict(
                source=str(image),
                imgsz=imgsz,
                project=str(pred_dir),
                name="visualizations",
                save=True,
                exist_ok=True,
                verbose=False,
            )
    return row
