#!/usr/bin/env python3
"""Object-loss rate: fraction of GT *objects* dropped by the gate (not just tiles).

For each (gate, threshold) point on the ships Pareto curve, compute:
  - n_gt_objects_total = sum of objects across all positive tiles
  - n_gt_objects_lost  = sum of objects on tiles the gate dropped
  - object_loss_rate   = n_gt_objects_lost / n_gt_objects_total

This is more interpretable than ``gate_recall_on_positive_tiles`` because a
single dropped tile can hide multiple objects.
"""
from __future__ import annotations
import json, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cascade import load_tile_scores  # noqa: E402


def per_tile_object_count(metadata_jsonl: str, split: str) -> dict[str, int]:
    """Return tile_id -> num_positives, but ONLY for tiles that are positive
    (is_positive=1). Negative tiles have no GT objects of the positive class
    even if their metadata's num_positives field has a stale class-agnostic
    count."""
    out: dict[str, int] = {}
    for line in Path(metadata_jsonl).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != split:
            continue
        if int(row.get("is_positive", 0)) != 1:
            continue
        out[row["tile_id"]] = int(row.get("num_positives", 0))
    return out


def object_loss_at_threshold(scores_path: str, n_objects: dict[str, int], threshold: float) -> tuple[int, int]:
    """Return (objects_lost, objects_total) where objects_total = total GT
    objects on positive tiles, and objects_lost = objects on positive tiles
    the gate dropped (prob < threshold)."""
    scores = load_tile_scores(scores_path)
    total = sum(n_objects.values())
    lost = sum(n_objects[s.tile_id] for s in scores if s.tile_id in n_objects and s.prob < threshold)
    return lost, total


def main() -> int:
    cases = [
        ("DOTA-ships",         "reports/gate_scores/gate_mbv3large_val.jsonl",
         "data/processed/dota_ships/metadata/tiles.jsonl", "val"),
        ("DOTA-planes",        "reports/planes/gate_scores/gate_planes_resnet50_val.jsonl",
         "data/processed/dota_planes/metadata/tiles.jsonl", "val"),
        ("DOTA-small_vehicle", "reports/small_vehicle/gate_scores/gate_small_vehicle_resnet50_val.jsonl",
         "data/processed/dota_small_vehicle/metadata/tiles.jsonl", "val"),
    ]
    print(f"{'dataset':<22} {'gate':<22} {'t':>6} {'objs_total':>11} {'objs_lost':>10} {'loss_rate':>10}")
    rows = []
    for name, scores_path, meta, split in cases:
        n_objs = per_tile_object_count(meta, split)
        for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
            lost, total = object_loss_at_threshold(scores_path, n_objs, t)
            loss_rate = lost / total if total else 0.0
            gate = Path(scores_path).stem.replace("_val", "")
            rows.append({"dataset": name, "gate": gate, "threshold": t,
                         "objects_total": total, "objects_lost": lost, "loss_rate": loss_rate})
            print(f"  {name:<20} {gate:<22} {t:>6.2f} {total:>11} {lost:>10} {loss_rate:>10.4f}")
    out = Path("reports/object_loss_rate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n  Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
