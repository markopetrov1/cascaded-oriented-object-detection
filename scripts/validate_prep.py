#!/usr/bin/env python3
"""Post-tiling validation: count tiles, verify binary↔OBB consistency, print
sparsity summary, confirm geographic split disjointness. Run once after
prepare_data.py to catch issues before kicking off long training jobs.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml


def main(processed_root: str = "data/processed/dota_ships", split_yaml: str = "configs/splits/geographic_v1.yaml") -> int:
    root = Path(processed_root)
    metadata = root / "metadata" / "tiles.jsonl"
    if not metadata.exists():
        print(f"FAIL: missing {metadata}")
        return 2

    rows = [json.loads(l) for l in metadata.read_text().splitlines() if l.strip()]
    print(f"=== {processed_root} ===")
    print(f"total tiles: {len(rows):,} from {len({r['source_stem'] for r in rows}):,} source images")

    # Per-split tile counts.
    by_split = Counter(r["split"] for r in rows)
    by_split_pos = Counter(r["split"] for r in rows if r.get("is_positive"))
    for split, n in sorted(by_split.items()):
        n_pos = by_split_pos.get(split, 0)
        print(f"  {split}: {n:,} tiles  ({n_pos:,} positive = {n_pos/max(n,1):.1%})")

    # Binary ↔ OBB consistency: is_positive==1 iff num_positives>0.
    inconsistent = sum(1 for r in rows if (int(r.get("num_positives", 0)) > 0) != bool(r.get("is_positive")))
    if inconsistent:
        print(f"FAIL: binary/OBB inconsistency on {inconsistent} tiles")
        return 3
    print("binary ↔ OBB consistency: OK")

    # Header presence.
    n_with_source = sum(1 for r in rows if r.get("imagesource"))
    n_with_gsd = sum(1 for r in rows if isinstance(r.get("gsd"), (int, float)))
    print(f"tiles with imagesource: {n_with_source:,} ({n_with_source/len(rows):.0%})")
    print(f"tiles with gsd: {n_with_gsd:,} ({n_with_gsd/len(rows):.0%})")

    # Class distribution from class_counts.
    cls_counter: Counter[str] = Counter()
    for r in rows:
        for c, n in (r.get("class_counts") or {}).items():
            cls_counter[c] += int(n)
    print("top classes by total objects across positive tiles:")
    for cls, n in cls_counter.most_common(8):
        print(f"  {cls}: {n:,}")

    # Geographic split.
    split_path = Path(split_yaml)
    if split_path.exists():
        spec = yaml.safe_load(split_path.read_text())
        train, val, test = set(spec["train_stems"]), set(spec["val_stems"]), set(spec["test_stems"])
        if train & val or train & test or val & test:
            print("FAIL: geographic split not disjoint")
            return 4
        print(
            f"geographic split disjoint: {len(train)}/{len(val)}/{len(test)} "
            f"train/val/test stems (total {len(train|val|test)})"
        )
        # Cross-check tile coverage.
        tiles_in_split = {"train": 0, "val": 0, "test": 0}
        for r in rows:
            stem = r["source_stem"]
            if stem in train:
                tiles_in_split["train"] += 1
            elif stem in val:
                tiles_in_split["val"] += 1
            elif stem in test:
                tiles_in_split["test"] += 1
        print(f"tiles per geographic split: {tiles_in_split}")

    # Spot-check a single tile triplet on disk.
    sample = next((r for r in rows if r.get("is_positive")), rows[0])
    img = root / "images" / sample["split"] / f"{sample['tile_id']}{Path(sample['source_image']).suffix}"
    if not img.exists():
        # Try .png; we standardize on the source suffix in dota.py.
        img = root / "images" / sample["split"] / f"{sample['tile_id']}.png"
    obb = root / "labels" / sample["split"] / f"{sample['tile_id']}.txt"
    gate = root / "gate_labels" / sample["split"] / f"{sample['tile_id']}.txt"
    print(f"\nspot-check tile: {sample['tile_id']}")
    for label, p in [("image", img), ("obb_label", obb), ("gate_label", gate)]:
        ok = p.exists()
        size = p.stat().st_size if ok else 0
        print(f"  {label}: {'OK' if ok else 'MISSING'} ({size}B) {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
