from __future__ import annotations

import importlib.metadata
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.hardware import collect_environment
from src.utils.paths import write_json


def command_string(argv: list[str] | None = None) -> str:
    return " ".join(argv or sys.argv)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def tensorrt_version() -> str:
    try:
        import tensorrt as trt

        return str(getattr(trt, "__version__", "unknown"))
    except Exception:
        return package_version("tensorrt")


def build_repro_metadata(
    *,
    weights: str | Path | None = None,
    exported_model: str | Path | None = None,
    data_yaml: str | Path | None = None,
    task: str | None = None,
    split: str | None = None,
    imgsz: int | None = None,
    batch: int | float | None = None,
    device: str | int | None = None,
    precision: str | None = None,
    command: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env = collect_environment(".")
    torch_info = env.get("torch", {}) if isinstance(env.get("torch"), dict) else {}
    devices = torch_info.get("devices") if isinstance(torch_info.get("devices"), list) else []
    packages = env.get("packages", {}) if isinstance(env.get("packages"), dict) else {}
    metadata: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights) if weights is not None else None,
        "exported_model": str(exported_model) if exported_model is not None else None,
        "dataset_yaml": str(data_yaml) if data_yaml is not None else None,
        "task": task,
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "precision": precision,
        "command": command or command_string(),
        "gpu_name": devices[0].get("name") if devices else None,
        "cuda_version": torch_info.get("cuda_version"),
        "pytorch_version": packages.get("torch", package_version("torch")),
        "ultralytics_version": packages.get("ultralytics", package_version("ultralytics")),
        "tensorrt_version": tensorrt_version(),
        "environment": env,
    }
    if extra:
        metadata.update(extra)
    return metadata


def write_repro_metadata(metadata: dict[str, Any], path: str | Path) -> None:
    write_json(metadata, path)
