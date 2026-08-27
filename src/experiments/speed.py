from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.experiments.trainer import load_experiment_config
from src.utils.paths import ensure_dir, iter_images, project_root, read_yaml, write_json
from src.utils.reproducibility import build_repro_metadata, write_repro_metadata
from src.utils.ultralytics_env import configure_ultralytics_settings


def model_file_size_mb(weights: str | Path) -> float | None:
    path = Path(weights)
    if not path.exists():
        return None
    return path.stat().st_size / (1024 * 1024)


def _dataset_images(data_yaml: str | Path, split: str, limit: int) -> list[Path]:
    data_path = Path(data_yaml)
    config = read_yaml(data_path)
    root = Path(config.get("path", "."))
    if not root.is_absolute():
        root = (project_root() / root).resolve()
    split_value = config.get(split) or config.get("val")
    if split_value is None:
        raise ValueError(f"No split '{split}' or val split found in {data_yaml}")
    source = Path(split_value)
    if not source.is_absolute():
        source = root / source
    images = iter_images(source)
    if not images:
        raise FileNotFoundError(f"No benchmark images found for split '{split}' at {source}")
    return images[:limit]


def _model_stats(model: Any) -> dict[str, Any]:
    stats = {"parameters": None, "flops": None}
    try:
        stats["parameters"] = int(sum(p.numel() for p in model.model.parameters()))
    except Exception:
        pass
    try:
        info = model.info(detailed=False, verbose=False)
        if isinstance(info, tuple) and len(info) >= 4:
            stats["flops"] = float(info[3])
    except Exception:
        pass
    return stats


def benchmark_model(
    weights: str,
    data_yaml: str,
    task: str,
    imgsz: int,
    split: str = "val",
    output_dir: str | Path = "results/metrics/speed",
    run_name: str | None = None,
    device: str = "auto",
    batch: int = 1,
    half: bool = False,
    image_count: int = 50,
    warmup: int = 10,
    iterations: int = 100,
    precision: str | None = None,
    exported_model: str | Path | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    configure_ultralytics_settings()
    from ultralytics import YOLO

    model = YOLO(weights, task=task)
    images = _dataset_images(data_yaml, split, image_count)
    run_name = run_name or Path(weights).stem

    stats = _model_stats(model)
    rows: list[float] = []
    predict_kwargs: dict[str, Any] = {
        "imgsz": imgsz,
        "batch": batch,
        "half": half,
        "verbose": False,
        "save": False,
    }
    if device not in (None, "auto", ""):
        predict_kwargs["device"] = device

    for image in images[: max(1, min(warmup, len(images)))]:
        model.predict(source=str(image), **predict_kwargs)

    sequence = [images[idx % len(images)] for idx in range(iterations * max(1, batch))]
    batches = [sequence[idx : idx + max(1, batch)] for idx in range(0, len(sequence), max(1, batch))]
    for batch_images in batches:
        start = time.perf_counter()
        model.predict(source=[str(image) for image in batch_images], **predict_kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        rows.append(elapsed_ms / max(1, len(batch_images)))

    mean_ms = statistics.mean(rows)
    std_ms = statistics.pstdev(rows) if len(rows) > 1 else 0.0
    result = {
        "run_name": run_name,
        "weights": weights,
        "task": task,
        "data": data_yaml,
        "split": split,
        "imgsz": imgsz,
        "device": device,
        "batch": batch,
        "half": half,
        "image_count": len(images),
        "warmup": warmup,
        "iterations": iterations,
        "model_size_mb": model_file_size_mb(weights),
        "parameters": stats["parameters"],
        "flops": stats["flops"],
        "latency_mean_ms": mean_ms,
        "latency_std_ms": std_ms,
        "latency_p50_ms": float(np.percentile(rows, 50)),
        "latency_p95_ms": float(np.percentile(rows, 95)),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else None,
    }
    if precision or exported_model:
        result["export_precision"] = precision or ("fp16" if half else "fp32")
        result["exported_model"] = str(exported_model) if exported_model is not None else ""
    out_dir = ensure_dir(Path(output_dir))
    write_json(result, out_dir / f"{run_name}.json")
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
                precision=precision or ("fp16" if half else "fp32"),
                command=command,
            ),
            out_dir / f"{run_name}_metadata.json",
        )
    csv_path = out_dir / f"{run_name}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)
    return result


def benchmark_model_cpu_and_device(
    weights: str,
    data_yaml: str,
    task: str,
    imgsz: int,
    split: str = "val",
    output_dir: str | Path = "results/metrics/speed",
    run_name: str | None = None,
    device: str = "auto",
    batch: int = 1,
    half: bool = False,
    image_count: int = 50,
    warmup: int = 10,
    iterations: int = 100,
    precision: str | None = None,
    exported_model: str | Path | None = None,
    command: str | None = None,
) -> list[dict[str, Any]]:
    if device in (None, "auto", ""):
        requested = []
        try:
            import torch

            if torch.cuda.is_available():
                requested = ["0"]
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                requested = ["mps"]
        except Exception:
            requested = []
    else:
        requested = [str(device)]
    devices = requested + ["cpu"]
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for dev in devices:
        if dev in seen:
            continue
        seen.add(dev)
        suffix = "cpu" if dev == "cpu" else "gpu"
        rows.append(
            benchmark_model(
                weights=weights,
                data_yaml=data_yaml,
                task=task,
                imgsz=imgsz,
                split=split,
                output_dir=output_dir,
                run_name=f"{run_name or Path(weights).stem}_{suffix}",
                device=dev,
                batch=batch,
                half=half,
                image_count=image_count,
                warmup=warmup,
                iterations=iterations,
                precision=precision,
                exported_model=exported_model,
                command=command,
            )
        )
    return rows


def benchmark_from_config(
    config_path: str | Path,
    split: str = "val",
    output_dir: str | Path = "results/metrics/speed",
    device: str | None = None,
    batch: int = 1,
    half: bool = False,
    image_count: int = 50,
    warmup: int = 10,
    iterations: int = 100,
    precision: str | None = None,
    command: str | None = None,
) -> list[dict[str, Any]]:
    config = load_experiment_config(config_path)
    results = []
    for model_config in config["models"]:
        results.extend(
            benchmark_model_cpu_and_device(
                weights=model_config["weights"],
                data_yaml=config["dataset"],
                task=config.get("task", "detect"),
                imgsz=int(config.get("imgsz", 640)),
                split=split,
                output_dir=output_dir,
                run_name=model_config["name"],
                device=device or config.get("device", "auto"),
                batch=batch,
                half=half,
                image_count=image_count,
                warmup=warmup,
                iterations=iterations,
                precision=precision,
                command=command,
            )
        )
    return results
