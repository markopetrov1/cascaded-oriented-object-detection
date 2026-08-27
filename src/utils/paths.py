from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base or project_root() / candidate).resolve()


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def write_json(data: Any, path: str | Path, indent: int = 2) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, sort_keys=True)
        handle.write("\n")


def append_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def iter_images(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(p for p in root_path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def link_or_copy(src: str | Path, dst: str | Path, copy: bool = False) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    ensure_dir(dst_path.parent)
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    if copy:
        shutil.copy2(src_path, dst_path)
    else:
        dst_path.symlink_to(src_path.resolve())


def require_exists(path: str | Path, what: str) -> Path:
    checked = Path(path)
    if not checked.exists():
        raise FileNotFoundError(f"{what} not found: {checked}")
    return checked
