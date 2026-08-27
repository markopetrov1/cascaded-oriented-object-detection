from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def git_commit(root: str | Path = ".") -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def package_versions(names: list[str] | None = None) -> dict[str, str]:
    names = names or [
        "ultralytics",
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "Pillow",
        "PyYAML",
        "pandas",
        "matplotlib",
    ]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def collect_environment(root: str | Path = ".") -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_commit": git_commit(root),
        "packages": package_versions(),
    }
    try:
        import torch

        env["torch"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
            "devices": [
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "capability": torch.cuda.get_device_capability(idx),
                    "memory_total": torch.cuda.get_device_properties(idx).total_memory,
                }
                for idx in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        env["torch"] = {"error": str(exc)}
    return env
