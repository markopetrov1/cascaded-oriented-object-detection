#!/usr/bin/env python3
"""End-to-end DOTA preparation: tiling + binary gate labels + enriched metadata
+ sparsity statistics + geographic split.

Single-command entry point. Splits already-existing tile-level metadata so it
can be re-run cheaply when only stats or splits change.

Usage::

    python scripts/prepare_data.py \\
        --raw-dir data/raw/DOTA \\
        --out-dir data/processed/dota_ships \\
        --tile-size 1024 --overlap 200 \\
        --positive-classes ship \\
        --stats-out reports/sparsity_ships.json \\
        --splits-out configs/splits/geographic_v1.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/prepare_data.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.converters import DOTA_CLASSES, DOTA_V2_CLASSES  # noqa: E402
from src.datasets.dota import prepare_dota  # noqa: E402
from src.splits import build_geographic_split  # noqa: E402
from src.sparsity import compute_sparsity_stats  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw-dir", required=True, help="DOTA raw root with train/, val/, test/")
    p.add_argument("--out-dir", required=True, help="Output dir for tiled images + labels + metadata")
    p.add_argument("--dataset-yaml", default="configs/datasets/dota.yaml", help="Where to write the Ultralytics dataset yaml")
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--overlap", type=int, default=200)
    p.add_argument("--min-area-ratio", type=float, default=0.1)
    p.add_argument(
        "--positive-classes",
        nargs="*",
        default=["ship"],
        help="Classes that mark a tile as positive for the gate. Default: ship",
    )
    p.add_argument(
        "--detector-classes",
        nargs="*",
        default=None,
        help=(
            "Classes the detector should learn (OBB labels are filtered + remapped to these "
            "consecutive ids). Defaults to --positive-classes for ship-only training; pass "
            "explicitly to train a multi-class detector that differs from the gate set."
        ),
    )
    p.add_argument(
        "--class-set",
        choices=("v1", "v2"),
        default="v1",
        help="Which DOTA class list to use. v1 = 15 classes (DOTA-v1.x), v2 = +container_crane,airport,helipad",
    )
    p.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    p.add_argument("--no-validate", action="store_true", help="Skip YOLO label validation")
    p.add_argument("--stats-out", default="reports/sparsity.json")
    p.add_argument("--splits-out", default="configs/splits/geographic_v1.yaml")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=67)
    p.add_argument(
        "--skip-split",
        action="store_true",
        help="Only emit tiles + metadata + sparsity. Useful for smoke runs on a single split.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="If existing metadata is found, skip already-processed source images and continue.",
    )
    p.add_argument(
        "--labels-only",
        action="store_true",
        help="Skip writing tile images. Use this to regenerate labels for a different positive_classes set when tiles already exist (e.g. via symlink). The caller is responsible for providing the images dir.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    classes = DOTA_V2_CLASSES if args.class_set == "v2" else DOTA_CLASSES
    detector_classes = args.detector_classes or args.positive_classes

    print(f"[prepare] tiling DOTA from {args.raw_dir} -> {args.out_dir}")
    print(f"[prepare] gate-positive classes: {args.positive_classes}; detector classes: {detector_classes}")
    summary = prepare_dota(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        dataset_yaml=args.dataset_yaml,
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_area_ratio=args.min_area_ratio,
        classes=classes,
        positive_classes=args.positive_classes,
        detector_classes=detector_classes,
        splits=tuple(args.splits),
        validate=not args.no_validate,
        resume=args.resume,
        labels_only=args.labels_only,
    )
    print(
        f"[prepare] tiles: train={summary.train_tiles} ({summary.train_positive_tiles} pos) "
        f"val={summary.val_tiles} ({summary.val_positive_tiles} pos) "
        f"test={summary.test_tiles} ({summary.test_positive_tiles} pos)"
    )

    metadata_jsonl = Path(args.out_dir) / "metadata" / "tiles.jsonl"
    if metadata_jsonl.exists() and metadata_jsonl.stat().st_size > 0:
        print(f"[prepare] computing sparsity stats -> {args.stats_out}")
        report = compute_sparsity_stats(metadata_jsonl, args.stats_out)
        print(f"[prepare] global positive_rate = {report['global'].get('positive_rate', 0):.4f}")
    else:
        print("[prepare] no metadata produced; skipping sparsity stats")
        return 1

    if args.skip_split:
        print("[prepare] --skip-split set; not building geographic split")
        return 0

    print(f"[prepare] building geographic split -> {args.splits_out}")
    split_payload = build_geographic_split(
        metadata_jsonl=metadata_jsonl,
        out_yaml=args.splits_out,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    print(
        f"[prepare] split: {split_payload['n_train_stems']}/"
        f"{split_payload['n_val_stems']}/{split_payload['n_test_stems']} "
        f"train/val/test stems from {split_payload['n_source_images']} images"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
