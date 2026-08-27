from __future__ import annotations

import os
from pathlib import Path

from src.utils.paths import ensure_dir, project_root


def configure_ultralytics_settings() -> Path:
    config_dir = Path(os.environ.get("YOLO_CONFIG_DIR", project_root() / ".ultralytics"))
    ensure_dir(config_dir)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    return config_dir
