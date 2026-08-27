#!/usr/bin/env python3
"""Build per-class gate-label trees without re-tiling anything.

A gate task is fully specified by which classes count as positive, and that is
already recorded per tile in metadata/tiles.jsonl. So a new gate task needs
only a directory of 0/1 files -- seconds of work -- rather than another pass of
prepare_data.py over the raw imagery, which takes about 45 minutes per class.

Layout produced, matching what src.gate.GateDataset expects:

    data/processed/gate_<task>/images/<split>      -> symlink to the shared tiles
    data/processed/gate_<task>/gate_labels/<split>/<tile_id>.txt
    data/processed/gate_<task>/metadata/tiles.jsonl -> symlink

GateDataset derives the label path from the dataset root rather than from the
image path, so a directory symlink is safe here. It is *not* safe for the
Ultralytics detector, which resolves image paths and would silently read
labels from the symlink's target -- that tree uses hard links instead.

Usage:
    python3 scripts/make_gate_labels.py --task ship
    python3 scripts/make_gate_labels.py --task any
    python3 scripts/make_gate_labels.py --all --min-positive-val 100
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SOURCE = Path("data/processed/dota_ships")
METADATA = SOURCE / "metadata" / "tiles.jsonl"
OUT_ROOT = Path("data/processed")
SPLITS = ("train", "val")


def load_rows() -> list[dict]:
    return [json.loads(line) for line in METADATA.read_text().splitlines() if line.strip()]


def task_slug(task: str) -> str:
    return "gate_" + task.replace(" ", "_")


def build(task: str, rows: list[dict], force: bool = False) -> dict:
    """Write one gate-label tree. `task` is a class name or 'any'."""
    root = OUT_ROOT / task_slug(task)
    counts = {}
    for split in SPLITS:
        img_src = (SOURCE / "images" / split).resolve()
        img_dst = root / "images" / split
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        if img_dst.is_symlink() or img_dst.exists():
            if force:
                img_dst.unlink()
                img_dst.symlink_to(img_src)
        else:
            img_dst.symlink_to(img_src)

        gate_dir = root / "gate_labels" / split
        gate_dir.mkdir(parents=True, exist_ok=True)

        n = n_pos = 0
        for row in rows:
            if row["split"] != split:
                continue
            present = set(row.get("class_counts", {}))
            positive = bool(present) if task == "any" else (task in present)
            (gate_dir / f"{row['tile_id']}.txt").write_text("1" if positive else "0")
            n += 1
            n_pos += int(positive)
        counts[split] = {"n_tiles": n, "n_positive": n_pos, "p_plus": n_pos / n if n else 0.0}

    meta_dst = root / "metadata"
    meta_dst.mkdir(parents=True, exist_ok=True)
    link = meta_dst / "tiles.jsonl"
    if not link.exists() and not link.is_symlink():
        link.symlink_to(METADATA.resolve())

    return {"task": task, "root": str(root), **counts}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="class name, or 'any' for the union of all classes")
    ap.add_argument("--all", action="store_true", help="build every viable class plus 'any'")
    ap.add_argument("--min-positive-val", type=int, default=100,
                    help="skip classes with fewer positive val tiles than this; a gate "
                         "trained on a handful of positives is not a measurement")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rows = load_rows()
    val_counts = Counter()
    for row in rows:
        if row["split"] == "val":
            val_counts.update(set(row.get("class_counts", {})))

    if args.all:
        viable = [c for c, n in val_counts.items() if n >= args.min_positive_val]
        skipped = [(c, n) for c, n in val_counts.items() if n < args.min_positive_val]
        tasks = sorted(viable, key=lambda c: val_counts[c]) + ["any"]
        for c, n in sorted(skipped, key=lambda x: x[1]):
            print(f"  skipping {c!r}: only {n} positive val tiles")
    elif args.task:
        tasks = [args.task]
    else:
        ap.error("pass --task or --all")

    print(f"\n{'task':<22}{'train p+':>10}{'val p+':>10}{'val pos':>10}")
    built = []
    for task in tasks:
        info = build(task, rows, force=args.force)
        built.append(info)
        print(f"{task:<22}{info['train']['p_plus']:>10.4f}"
              f"{info['val']['p_plus']:>10.4f}{info['val']['n_positive']:>10}")

    out = Path("reports/gate_tasks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(built, indent=2))
    print(f"\n  {len(built)} gate tasks -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
