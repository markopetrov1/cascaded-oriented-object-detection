#!/usr/bin/env python3
"""How much the distillation soft targets say beyond the tile labels (A-F2).

The soft target of a training tile is the highest confidence among the
detector's predictions on it, zero where it predicted nothing
(src/codesign.py soft_target_from_detector, as used by scripts/train_distill.py).
Writes reports/r1/distill_targets.json with the targets' PR-AUC against the
binary tile labels and their mean on positive and on empty tiles.

    python scripts/r1_distill_targets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.codesign import soft_target_from_detector  # noqa: E402

PRED = REPO / "runs/obb/runs/baseline_yolo11m_dota_ships_obb_p50/predict_train_c001"
META = REPO / "data/processed/dota_ships/metadata/tiles.jsonl"
IMAGES = REPO / "data/processed/dota_ships/images/train"
OUT = REPO / "reports/r1/distill_targets.json"


def main() -> int:
    scores = {}
    for txt in sorted((PRED / "labels").glob("*.txt")):
        s = [float(p[9]) for p in (l.split() for l in txt.read_text().splitlines()) if len(p) >= 10]
        scores[txt.stem] = np.asarray(s, dtype=np.float32)
    train = {p.stem for p in IMAGES.iterdir()}
    rows = [json.loads(l) for l in open(META)]
    rows = [r for r in rows if r["tile_id"] in train]
    ids = [r["tile_id"] for r in rows]
    y = np.array([int(r["is_positive"]) for r in rows])
    t = soft_target_from_detector(scores, ids)
    out = {"predictions": str(PRED.relative_to(REPO)), "n_tiles": len(ids), "n_positive": int(y.sum()),
           "prauc_vs_labels": float(average_precision_score(y, t)),
           "mean_positive": float(t[y == 1].mean()), "mean_empty": float(t[y == 0].mean()),
           "share_empty_above_half": float((t[y == 0] >= 0.5).mean()),
           "share_positive_below_half": float((t[y == 1] < 0.5).mean())}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
