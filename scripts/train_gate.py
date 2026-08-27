#!/usr/bin/env python3
"""Train a Stage-1 binary gate from a YAML config.

Usage::

    python scripts/train_gate.py --config configs/experiments/gate_resnet18.yaml --device 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gate import GateTrainConfig, train_gate  # noqa: E402
from src.utils.paths import read_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--device", default=None, help="GPU index (e.g. 0) or 'cpu'")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--name", default=None, help="Override run name")
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def _resolve_split_tile_ids(splits_yaml: str | None) -> tuple[list[str] | None, list[str] | None]:
    """If a splits_yaml is supplied, return (train_tile_filter, val_tile_filter)
    where each filter is a list of source-stem prefixes the GateDataset should
    accept. Returns (None, None) if no split is configured (use all tiles).
    """
    if not splits_yaml or not Path(splits_yaml).exists():
        return None, None
    spec = read_yaml(splits_yaml)
    train_stems = set(spec.get("train_stems", []))
    val_stems = set(spec.get("val_stems", []))
    return _stems_to_tile_filter(train_stems), _stems_to_tile_filter(val_stems)


def _stems_to_tile_filter(stems: set[str]) -> list[str] | None:
    """The GateDataset filters by tile_id (the file stem). prepare_dota writes
    tile stems as ``{source_stem}__x..._y..._w..._h...``. We can't enumerate every
    tile_id without scanning the disk, but GateDataset accepts a list and matches
    exactly. Return None to mean 'no filter'; otherwise the tile_ids list is
    expanded by the caller after loading the dataset.

    This implementation defers to a stem-based filter inside the dataloader by
    returning a sentinel list; we adapt by reading directories from disk in the
    train script directly. See ``_expand_tile_ids``.
    """
    if not stems:
        return []
    # We tag with the stems set; the train script will expand to actual tile_ids.
    return ["__STEM_FILTER__:" + ",".join(sorted(stems))]


def _expand_tile_ids(filter_spec: list[str] | None, data_root: Path, split: str) -> list[str] | None:
    """Resolve a stem-prefix filter into the actual tile-id list."""
    if filter_spec is None:
        return None
    if not filter_spec:
        return []  # split intentionally empty
    if filter_spec[0].startswith("__STEM_FILTER__:"):
        stems = set(filter_spec[0].split(":", 1)[1].split(","))
        images_dir = data_root / "images" / split
        if not images_dir.exists():
            raise FileNotFoundError(f"Missing images dir for split {split}: {images_dir}")
        ids = []
        for path in sorted(images_dir.iterdir()):
            stem = path.stem
            source = stem.split("__", 1)[0]
            if source in stems:
                ids.append(stem)
        return ids
    return list(filter_spec)


def main() -> int:
    args = parse_args()
    raw = read_yaml(args.config)
    overrides: dict = {}
    if args.device is not None:
        overrides["device"] = (
            f"cuda:{args.device}" if args.device.isdigit() else args.device
        )
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.name is not None:
        overrides["name"] = args.name
    if args.use_wandb:
        overrides["use_wandb"] = True
    if args.no_wandb:
        overrides["use_wandb"] = False

    splits_yaml = raw.pop("splits_yaml", None)
    raw.update(overrides)

    train_filter, val_filter = _resolve_split_tile_ids(splits_yaml)
    raw["train_tile_ids"] = _expand_tile_ids(train_filter, Path(raw["data_root"]), raw["train_split"])
    raw["val_tile_ids"] = _expand_tile_ids(val_filter, Path(raw["data_root"]), raw["val_split"])

    cfg = GateTrainConfig(**raw)
    print(f"[train_gate] backbone={cfg.backbone} loss={cfg.loss} sampler={cfg.sampler} epochs={cfg.epochs}")
    out = train_gate(cfg)
    print(f"[train_gate] done. best val PR-AUC = {out['best_pr_auc']:.4f}; run_dir = {out['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
