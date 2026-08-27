from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from src.utils.hardware import collect_environment
from src.utils.paths import ensure_dir, read_yaml, write_json
from src.utils.seed import set_seed
from src.utils.ultralytics_env import configure_ultralytics_settings


def _device_arg(device: str | int | None) -> str | int | None:
    if device in (None, "auto", ""):
        return None
    return device


def _batch_arg(batch: int | float) -> int | float:
    if isinstance(batch, float) and batch.is_integer():
        return int(batch)
    return batch


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config = read_yaml(path)
    if "models" not in config and "weights" in config:
        config["models"] = [{"name": Path(str(config["weights"])).stem, "weights": config["weights"]}]
    if "models" not in config:
        raise ValueError(f"Experiment config must define models: {path}")
    return config


def train_one(
    weights: str,
    dataset: str,
    task: str,
    name: str,
    project: str = "results/raw",
    imgsz: int = 640,
    epochs: int = 100,
    batch: int | float = 16,
    device: str | int | None = None,
    seed: int = 67,
    extra_args: dict[str, Any] | None = None,
) -> Path:
    configure_ultralytics_settings()
    from ultralytics import YOLO

    batch = _batch_arg(batch)
    set_seed(seed)
    run_dir = ensure_dir(Path(project) / name)
    full_config = {
        "weights": weights,
        "dataset": dataset,
        "task": task,
        "name": name,
        "project": project,
        "imgsz": imgsz,
        "epochs": epochs,
        "batch": batch,
        "device": device,
        "seed": seed,
        "extra_args": extra_args or {},
    }
    write_json(full_config, run_dir / "config_used.json")
    write_json(collect_environment("."), run_dir / "environment.json")

    model = YOLO(weights, task=task)
    kwargs: dict[str, Any] = {
        "data": dataset,
        "imgsz": imgsz,
        "epochs": epochs,
        "batch": batch,
        "project": project,
        "name": name,
        "seed": seed,
        "plots": True,
        "exist_ok": True,
    }
    selected_device = _device_arg(device)
    if selected_device is not None:
        kwargs["device"] = selected_device
    kwargs.update(extra_args or {})
    result = model.train(**kwargs)

    actual_dir = Path(getattr(getattr(result, "trainer", None), "save_dir", run_dir))
    if actual_dir != run_dir and actual_dir.exists():
        ensure_dir(run_dir)
        for sidecar in ("config_used.json", "environment.json"):
            src = run_dir / sidecar
            if src.exists():
                shutil.copy2(src, actual_dir / sidecar)
        return actual_dir
    return run_dir


def train_from_config(
    config_path: str | Path,
    only_model: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> list[Path]:
    config = load_experiment_config(config_path)
    overrides = overrides or {}
    extra_args = {**config.get("train_args", {}), **overrides.get("train_args", {})}
    run_dirs: list[Path] = []
    for model_config in config["models"]:
        if only_model and only_model not in {model_config.get("name"), model_config.get("weights")}:
            continue
        run_dirs.append(
            train_one(
                weights=model_config["weights"],
                dataset=overrides.get("dataset", config["dataset"]),
                task=overrides.get("task", config.get("task", "detect")),
                name=overrides.get("name", model_config["name"]),
                project=overrides.get("project", config.get("project", "results/raw")),
                imgsz=int(overrides.get("imgsz", config.get("imgsz", 640))),
                epochs=int(overrides.get("epochs", config.get("epochs", 100))),
                batch=overrides.get("batch", config.get("batch", 16)),
                device=overrides.get("device", config.get("device")),
                seed=int(overrides.get("seed", config.get("seed", 67))),
                extra_args=extra_args,
            )
        )
    return run_dirs
