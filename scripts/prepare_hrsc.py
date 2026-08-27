#!/usr/bin/env python3
"""Convert HRSC2016 to the cascade pipeline's directory layout.

Usage::

    python scripts/prepare_hrsc.py --raw-dir data/raw/HRSC2016 --out-dir data/processed/hrsc2016
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.hrsc2016 import prepare_hrsc2016  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset-yaml", default="configs/datasets/hrsc2016_ships.yaml")
    p.add_argument("--splits", nargs="+", default=["test"], help="Which HRSC splits to convert")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    counts = prepare_hrsc2016(
        raw_dir=args.raw_dir, out_dir=args.out_dir, dataset_yaml=args.dataset_yaml, splits=tuple(args.splits)
    )
    for split, n in counts.items():
        print(f"[prepare_hrsc] {split}: {n} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
