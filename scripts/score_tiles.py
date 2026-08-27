#!/usr/bin/env python3
"""Run a trained gate over a split and emit a per-tile (logit, prob) JSONL.

Usage::

    python scripts/score_tiles.py \\
        --weights runs/gate_resnet18/best.pt \\
        --data-root data/processed/dota_ships \\
        --split val \\
        --out reports/gate_scores/resnet18_val.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gate import score_tiles  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--weights", required=True, help="Path to a gate checkpoint (best.pt)")
    p.add_argument("--data-root", default="data/processed/dota_ships")
    p.add_argument("--split", default="val")
    p.add_argument("--out", required=True, help="Where to write the scored-tiles JSONL")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--backbone",
        default=None,
        help="Override the backbone name (otherwise inferred from the checkpoint config).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    score_tiles(
        checkpoint_path=args.weights,
        data_root=args.data_root,
        split=args.split,
        out_jsonl=args.out,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        amp=not args.no_amp,
        backbone=args.backbone,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
