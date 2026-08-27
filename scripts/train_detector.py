#!/usr/bin/env python3
"""Train a YOLO-OBB detector on the cascade dataset.

Thin wrapper over :mod:`src.experiments.trainer`. The only logic on top of the
vendored trainer: if the experiment config carries a ``splits_yaml`` key, we
read the geographic split, materialize per-split text files of absolute tile
paths, derive a new dataset yaml pointing to those files, and pass that to the
trainer via ``overrides``. Otherwise the trainer runs unmodified on whatever
the dataset yaml says.

Usage::

    python scripts/train_detector.py --config configs/experiments/baseline_yolo11m_obb.yaml --device 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.trainer import load_experiment_config, train_from_config  # noqa: E402
from src.splits import materialize_split_filelists  # noqa: E402
from src.utils.paths import ensure_dir, read_yaml, write_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", required=True, help="Path to an experiment yaml under configs/experiments/")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", default=None, help="GPU index (e.g. 0) or 'cpu'")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--name", default=None, help="Override run name")
    p.add_argument(
        "--splits-yaml",
        default=None,
        help="Override geographic split yaml (otherwise read from experiment config).",
    )
    p.add_argument("--only-model", default=None, help="If config has multiple models, train only this one")
    return p.parse_args()


def _maybe_derive_split_yaml(
    experiment_config_path: Path, splits_yaml: str | None
) -> str | None:
    """If a splits_yaml is in play, write a derived dataset yaml whose train/val/test
    point to the materialized file-list text files. Returns the path to the
    derived dataset yaml, or ``None`` if no override is needed.
    """
    config = load_experiment_config(experiment_config_path)
    splits_path = splits_yaml or config.get("splits_yaml")
    if not splits_path:
        return None

    dataset_yaml_path = Path(config["dataset"])
    dataset_data = read_yaml(dataset_yaml_path)
    images_root = Path(dataset_data["path"]) / "images"
    if not images_root.exists():
        # Tiles haven't been written yet; let the trainer fail loudly with the
        # default yaml so the user sees the missing-data error rather than a
        # mysterious empty file-list.
        return None

    derived_dir = ensure_dir(dataset_yaml_path.parent / "_derived" / experiment_config_path.stem)
    image_suffix = next((p.suffix for p in images_root.rglob("*") if p.is_file()), ".png")
    file_lists = materialize_split_filelists(
        splits_yaml=splits_path,
        images_dir=images_root,
        out_dir=derived_dir,
        image_suffix=image_suffix,
    )

    derived = dict(dataset_data)
    derived["path"] = str(Path(dataset_data["path"]).resolve())
    derived["train"] = str(file_lists["train"].resolve())
    derived["val"] = str(file_lists["val"].resolve())
    derived["test"] = str(file_lists["test"].resolve())
    derived_yaml = derived_dir / dataset_yaml_path.name
    write_yaml(derived, derived_yaml)
    print(f"[train] derived split-aware dataset yaml -> {derived_yaml}")
    print(
        f"[train] tiles per split: train={_count_lines(file_lists['train'])} "
        f"val={_count_lines(file_lists['val'])} test={_count_lines(file_lists['test'])}"
    )
    return str(derived_yaml)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    overrides: dict = {}
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.device is not None:
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.name is not None:
        overrides["name"] = args.name

    derived_yaml = _maybe_derive_split_yaml(cfg_path, args.splits_yaml)
    if derived_yaml:
        overrides["dataset"] = derived_yaml

    run_dirs = train_from_config(cfg_path, only_model=args.only_model, overrides=overrides)
    for d in run_dirs:
        print(f"[train] run directory: {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
