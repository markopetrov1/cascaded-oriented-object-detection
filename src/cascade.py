"""Cascade runtime composition: gate → conditional detector → aggregated outputs.

Two views of the cascade live here:

1. ``CascadeRuntime`` — operationally accurate. Used for latency benchmarking
   and live inference. Calls the gate per tile, applies a threshold (or
   calibrated mapping), runs the detector only on accepted tiles.
2. ``cached_cascade_eval`` — research-time helper. Assumes you have already run
   the *full* detector on every tile (e.g. via Ultralytics ``predict``) and
   the gate has scored every tile. Composing the cascade is then just masking:
   detections from rejected tiles are dropped. This is dramatically cheaper for
   Pareto sweeps where the same (gate_scores, detector_outputs) pair is
   replayed with hundreds of thresholds and calibrators.

Co-design variants (shared backbone, early-exit, distillation,
end-to-end relaxed gating) live in this module starting in Phase 3. For now
the "independently-trained" cascade in (2) is enough.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass
class TileScore:
    tile_id: str
    label: int  # ground-truth is_positive
    prob: float  # gate probability (calibrated or raw)


@dataclass
class TileDetections:
    """Per-tile detections in tile-local coordinates.

    The detector's outputs are stored verbatim for later evaluation; the cascade
    layer never re-runs the detector. ``polygons`` is one (4,2) array per
    detection; ``scores`` and ``classes`` are aligned 1-D arrays.
    """

    tile_id: str
    polygons: np.ndarray  # (N, 4, 2) tile-local
    scores: np.ndarray  # (N,)
    classes: np.ndarray  # (N,) int


@dataclass
class CascadeOutput:
    """Aggregated cascade output for a tile, after gate filtering."""

    tile_id: str
    accepted: bool
    detections: TileDetections | None
    gate_prob: float
    threshold: float


# ---------- Cached evaluation (research-time, fast) ------------------------


def apply_threshold(
    tile_scores: list[TileScore],
    detections: dict[str, TileDetections],
    threshold: float,
) -> list[CascadeOutput]:
    """Compose gate scores + cached detector outputs at a fixed threshold.

    Tiles whose gate prob is below ``threshold`` are *rejected* — their
    detections are discarded entirely (the detector would not have run).
    Rejected tiles' ``detections`` are replaced with ``None`` so downstream mAP
    aggregators see them as empty.
    """
    out: list[CascadeOutput] = []
    for ts in tile_scores:
        if ts.prob >= threshold:
            out.append(
                CascadeOutput(
                    tile_id=ts.tile_id,
                    accepted=True,
                    detections=detections.get(ts.tile_id),
                    gate_prob=ts.prob,
                    threshold=threshold,
                )
            )
        else:
            out.append(
                CascadeOutput(
                    tile_id=ts.tile_id,
                    accepted=False,
                    detections=None,
                    gate_prob=ts.prob,
                    threshold=threshold,
                )
            )
    return out


def filter_rate(outputs: list[CascadeOutput]) -> float:
    if not outputs:
        return 0.0
    return float(np.mean([0.0 if o.accepted else 1.0 for o in outputs]))


def gate_recall(outputs: list[CascadeOutput], tile_labels: dict[str, int]) -> float:
    pos = [o for o in outputs if tile_labels.get(o.tile_id, 0) == 1]
    if not pos:
        return float("nan")
    return float(np.mean([1.0 if o.accepted else 0.0 for o in pos]))


# ---------- Loading helpers ------------------------------------------------


def load_tile_scores(path: str | Path) -> list[TileScore]:
    rows: list[TileScore] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(TileScore(tile_id=row["tile_id"], label=int(row["label"]), prob=float(row["prob"])))
    return rows


def load_tile_labels_from_metadata(metadata_jsonl: str | Path) -> dict[str, int]:
    """Build a tile_id -> is_positive mapping from prepare_dota's tiles.jsonl.
    Useful when scoring tiles via a separate gate run that didn't carry labels.
    """
    out: dict[str, int] = {}
    for line in Path(metadata_jsonl).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["tile_id"])] = int(row.get("is_positive", 0))
    return out


# ---------- Detector outputs adapter ---------------------------------------


def load_detector_outputs_yolo_obb(
    runs_dir: str | Path,
    image_root: str | Path,
) -> dict[str, TileDetections]:
    """Load Ultralytics OBB prediction labels from a `predict` run.

    Ultralytics writes one .txt per image under ``runs_dir/labels/`` with rows::
        <class_id> x1 y1 x2 y2 x3 y3 x4 y4 [score]

    Returns a dict keyed by tile_id (image stem) so cascade composition can
    look up detections without re-running inference.
    """
    runs_path = Path(runs_dir)
    labels_dir = runs_path / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Expected Ultralytics predict labels at {labels_dir}")

    import re

    def _tile_dims(stem: str) -> tuple[int, int]:
        m = re.search(r"_w(\d+)_h(\d+)", stem)
        return (int(m.group(1)), int(m.group(2))) if m else (1024, 1024)

    out: dict[str, TileDetections] = {}
    for txt in sorted(labels_dir.glob("*.txt")):
        w, h = _tile_dims(txt.stem)
        scale = np.array([w, h], dtype=np.float32)
        polys: list[np.ndarray] = []
        scores: list[float] = []
        classes: list[int] = []
        for line in txt.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            cls = int(parts[0])
            pts = np.asarray([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)
            pts = pts * scale  # normalized -> pixel coords
            score = float(parts[9]) if len(parts) >= 10 else 1.0
            polys.append(pts)
            scores.append(score)
            classes.append(cls)
        out[txt.stem] = TileDetections(
            tile_id=txt.stem,
            polygons=np.stack(polys, axis=0) if polys else np.zeros((0, 4, 2), dtype=np.float32),
            scores=np.asarray(scores, dtype=np.float32),
            classes=np.asarray(classes, dtype=np.int64),
        )
    return out


# ---------- Compose with calibrated probabilities --------------------------


def calibrated_threshold_sweep(
    probs: np.ndarray,
    labels: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    n_thresholds: int = 101,
) -> list[dict[str, float]]:
    """Helper: apply a probability transform (e.g. temperature scaling), then
    sweep thresholds, returning per-threshold (recall, filter_rate). Drops
    Pareto-relevant numbers in one pass.
    """
    p = transform(probs) if transform is not None else probs
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    rows: list[dict[str, float]] = []
    n_pos = max(int(labels.sum()), 1)
    for t in thresholds:
        kept = p >= t
        tp = int(((kept) & (labels == 1)).sum())
        rows.append(
            {
                "threshold": float(t),
                "recall": float(tp / n_pos),
                "filter_rate": float(1.0 - kept.mean()),
                "kept_positives": int(tp),
                "kept_negatives": int(((kept) & (labels == 0)).sum()),
            }
        )
    return rows


# ---------- Stubs for Phase-3 co-design ------------------------------------


@dataclass
class CoDesignConfig:
    """Configuration for shared-backbone / early-exit / distilled cascade.
    Filled out in Phase 3; defined here so the imports settle now.
    """

    name: str
    backbone: str = "resnet18"
    binary_head_position: int = -1  # -1 means at the final pooled feature
    detector_head: str = "obb_smooth_l1"
    distill_from_detector: bool = False
    relaxed_gating: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


def build_codesign(cfg: CoDesignConfig):  # pragma: no cover - Phase 3 stub
    raise NotImplementedError(
        "Co-design models (shared-backbone / early-exit / distilled / relaxed-gating) "
        "are Phase 3 work. Use the independently-trained cascade for now."
    )
