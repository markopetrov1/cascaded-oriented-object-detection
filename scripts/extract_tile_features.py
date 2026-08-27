#!/usr/bin/env python3
"""Extract cheap per-tile features used by context-adaptive calibration and
the learned-threshold MLP.

Outputs a JSONL at ``--out`` with one row per tile::

    {"tile_id": "...", "split": "val", "features": {"rgb_mean_r": 0.5, ...}}

Usage::

    python scripts/extract_tile_features.py \\
        --data-root data/processed/dota_ships --split val \\
        --out reports/tile_features/val.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tile_features import extract_features_for_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--out", required=True)
    p.add_argument("--image-suffix", default=".png")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    extract_features_for_split(
        data_root=args.data_root,
        split=args.split,
        out_jsonl=args.out,
        image_suffix=args.image_suffix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
