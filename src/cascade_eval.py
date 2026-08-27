"""End-to-end cascade evaluation: mAP, GFLOPs, latency, all together.

The headline figure of the paper is mAP-vs-compute (research plan §10).
This module produces the rows that go into that figure for any
``(gate_scores, detector_outputs, ground_truth, threshold, [calibrator])``
tuple. The split-of-concerns:

- ``OBBEvaluator``: rotated-IoU mAP@0.5 and mAP@0.5:0.95. We use the polygon
  IoU primitive from shapely (already a vendored dependency through
  ultralytics), not OpenCV's ``rotatedRectangleIntersection`` — shapely is
  numerically robust and clear.
- ``ComputeAccountant``: bookkeeping for FLOPs and latency. Per-stage costs
  are passed in as constants (measured once via ``src.experiments.speed``);
  per-image cost is then ``gate_flops * n_tiles + detector_flops * n_accepted``.
- ``cascade_pareto_row``: produces a single row of the Pareto table.
- ``oracle_gate``: synthesizes a perfect gate from ground-truth tile labels —
  the upper bound research plan §7.1 demands.

We deliberately *cache* detector outputs upstream and pass them in here;
re-running the detector for every threshold sweep is what wastes GPU-days.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.cascade import (
    CascadeOutput,
    TileDetections,
    TileScore,
    apply_threshold,
    filter_rate,
    gate_recall,
    load_tile_labels_from_metadata,
    load_tile_scores,
)


# ---------- Ground truth ---------------------------------------------------


@dataclass
class TileGroundTruth:
    """Per-tile OBB ground truth. ``polygons`` are tile-local in pixel coords."""

    tile_id: str
    polygons: np.ndarray  # (M, 4, 2)
    classes: np.ndarray  # (M,)


def _tile_dims_from_stem(stem: str, default: int = 1024) -> tuple[int, int]:
    """Extract (width, height) from tile stem, e.g. P0006__x0_y0_w1024_h1024 -> (1024, 1024)."""
    import re
    m = re.search(r"_w(\d+)_h(\d+)", stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    return default, default


def load_yolo_obb_ground_truth(
    labels_dir: str | Path,
    tile_size: int = 1024,
    keep_classes: set[int] | None = None,
) -> dict[str, TileGroundTruth]:
    """Read YOLO-OBB labels back into pixel-coord polygons.

    YOLO OBB labels are normalized to [0,1]. Tile dimensions are parsed from
    the tile stem (e.g. ``_w558_h1024``) so edge tiles are handled correctly.

    ``keep_classes`` restricts the ground truth to those class ids. mAP is
    averaged over the classes present in the ground truth, so passing a single
    id turns the result into that class's AP -- which is what the sparsity
    sweep needs, where a gate trained for one class is paired with the
    multi-class detector. Detections of other classes are then simply never
    scored. Filtering here avoids materialising a per-class copy of every
    label file.
    """
    out: dict[str, TileGroundTruth] = {}
    labels_path = Path(labels_dir)
    for txt in sorted(labels_path.glob("*.txt")):
        w, h = _tile_dims_from_stem(txt.stem, default=tile_size)
        scale = np.array([w, h], dtype=np.float32)
        polys: list[np.ndarray] = []
        cls: list[int] = []
        for line in txt.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            cid = int(parts[0])
            if keep_classes is not None and cid not in keep_classes:
                continue
            cls.append(cid)
            pts = np.asarray([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)
            polys.append(pts * scale)
        out[txt.stem] = TileGroundTruth(
            tile_id=txt.stem,
            polygons=np.stack(polys, axis=0) if polys else np.zeros((0, 4, 2), dtype=np.float32),
            classes=np.asarray(cls, dtype=np.int64),
        )
    return out


# ---------- Polygon IoU + mAP (precomputed for speed) ----------------------


def _polygon_iou(p1: np.ndarray, p2: np.ndarray) -> float:
    """IoU of two convex polygons via shapely."""
    from shapely.geometry import Polygon
    a = Polygon(p1.tolist())
    b = Polygon(p2.tolist())
    if not (a.is_valid and b.is_valid) or a.area <= 0 or b.area <= 0:
        return 0.0
    inter = a.intersection(b).area
    union = a.area + b.area - inter
    return float(inter / union) if union > 0 else 0.0


def _pairwise_iou(detections: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """Full IoU matrix [N_det, N_gt]. Shapely called once per pair."""
    if len(detections) == 0 or len(gts) == 0:
        return np.zeros((len(detections), len(gts)), dtype=np.float32)
    iou = np.zeros((len(detections), len(gts)), dtype=np.float32)
    for i, p in enumerate(detections):
        for j, g in enumerate(gts):
            iou[i, j] = _polygon_iou(p, g)
    return iou


@dataclass
class _PrecomputedTile:
    """Per-tile precomputed data for fast AP sweeps."""
    tile_id: str
    # Per class: (det_indices_global, iou_matrix[n_det_cls, n_gt_cls], n_gt_cls)
    # det_indices_global indexes into the flat global detection array
    det_scores: np.ndarray        # (N_det,)  all detections in this tile
    det_classes: np.ndarray       # (N_det,)
    gt_classes: np.ndarray        # (N_gt,)
    iou_matrix: np.ndarray        # (N_det, N_gt) - may be empty


def precompute_iou(
    detections: dict[str, "TileDetections"],
    ground_truth: dict[str, TileGroundTruth],
) -> dict[str, _PrecomputedTile]:
    """Precompute per-tile IoU matrices once. This is the only place shapely runs.

    Call this once before the threshold sweep; pass the result to
    ``evaluate_obb_map_precomputed`` for each threshold.
    """
    out: dict[str, _PrecomputedTile] = {}
    all_tile_ids = set(detections) | set(ground_truth)
    for tile_id in all_tile_ids:
        det = detections.get(tile_id)
        gt = ground_truth.get(tile_id)
        det_polys = det.polygons if det is not None else np.zeros((0, 4, 2), dtype=np.float32)
        det_scores = det.scores if det is not None else np.zeros(0, dtype=np.float32)
        det_classes = det.classes if det is not None else np.zeros(0, dtype=np.int64)
        gt_polys = gt.polygons if gt is not None else np.zeros((0, 4, 2), dtype=np.float32)
        gt_classes = gt.classes if gt is not None else np.zeros(0, dtype=np.int64)
        iou = _pairwise_iou(det_polys, gt_polys)
        out[tile_id] = _PrecomputedTile(
            tile_id=tile_id,
            det_scores=det_scores,
            det_classes=det_classes,
            gt_classes=gt_classes,
            iou_matrix=iou,
        )
    return out


def _ap_from_precomputed(
    tile_index: dict[str, _PrecomputedTile],
    accepted_set: set[str],
    cls: int,
    n_gt_total: int,
    iou_threshold: float,
) -> float:
    """AP for one class at one IoU threshold using precomputed IoU matrices."""
    if n_gt_total == 0:
        return 0.0
    entries: list[tuple[float, str, int]] = []  # (score, tile_id, det_local_idx)
    for tile_id in accepted_set:
        td = tile_index.get(tile_id)
        if td is None or len(td.det_classes) == 0:
            continue
        mask = td.det_classes == cls
        idxs = np.where(mask)[0]
        for i in idxs:
            entries.append((float(td.det_scores[i]), tile_id, int(i)))
    entries.sort(key=lambda x: -x[0])
    tp = np.zeros(len(entries), dtype=np.int64)
    fp = np.zeros(len(entries), dtype=np.int64)
    matched: dict[str, set[int]] = {}
    for d_idx, (_, tile_id, local_i) in enumerate(entries):
        td = tile_index[tile_id]
        gt_mask = td.gt_classes == cls
        gt_idxs = np.where(gt_mask)[0]
        if len(gt_idxs) == 0:
            fp[d_idx] = 1
            continue
        tile_matched = matched.setdefault(tile_id, set())
        iou_row = td.iou_matrix[local_i, gt_idxs]
        best_j_local = int(np.argmax(iou_row))
        best_iou = float(iou_row[best_j_local])
        best_gt_global = int(gt_idxs[best_j_local])
        if best_iou >= iou_threshold and best_gt_global not in tile_matched:
            tp[d_idx] = 1
            tile_matched.add(best_gt_global)
        else:
            fp[d_idx] = 1
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / max(n_gt_total, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    interpolated = np.zeros(101, dtype=np.float64)
    for i, r_t in enumerate(recall_thresholds):
        mask2 = recall >= r_t
        interpolated[i] = float(precision[mask2].max()) if mask2.any() else 0.0
    return float(interpolated.mean())


def evaluate_obb_map(
    cascade_outputs: list[CascadeOutput],
    ground_truth: dict[str, TileGroundTruth],
    iou_thresholds: tuple[float, ...] = (0.5,),
    _precomputed: dict[str, _PrecomputedTile] | None = None,
) -> dict[str, float]:
    """mAP given cascade outputs and per-tile GT.

    Pass ``_precomputed`` (from ``precompute_iou``) to avoid recomputing shapely
    IoU on repeated calls (threshold sweeps). If omitted, computed on the fly.
    """
    from src.cascade import TileDetections
    if _precomputed is None:
        det_map = {o.tile_id: o.detections for o in cascade_outputs if o.detections is not None}
        _precomputed = precompute_iou(det_map, ground_truth)

    accepted_set = {o.tile_id for o in cascade_outputs if o.accepted}
    classes = sorted({int(c) for td in _precomputed.values() for c in td.gt_classes.tolist()})
    if not classes:
        return {"mAP@0.50": 0.0, "mAP@0.5:0.95": 0.0, "n_classes": 0}

    out: dict[str, float] = {}
    per_iou_aps: list[float] = []
    for iou_t in iou_thresholds:
        class_aps = []
        for cls in classes:
            # GT denominator = all tiles (rejected tiles' objects become unrecoverable FN).
            n_gt = int(sum(int(np.sum(td.gt_classes == cls)) for td in _precomputed.values()))
            class_aps.append(_ap_from_precomputed(_precomputed, accepted_set, cls, n_gt, iou_t))
        mean_ap = float(np.mean(class_aps))
        out[f"mAP@{iou_t:.2f}"] = mean_ap
        per_iou_aps.append(mean_ap)
    out["mAP@0.5:0.95"] = float(np.mean(per_iou_aps))
    out["n_classes"] = len(classes)
    return out


# ---------- Compute bookkeeping --------------------------------------------


@dataclass
class StageCost:
    """Measured per-tile cost of one stage of the cascade.

    Latency is the median per-tile cost on the reference hardware (RTX 8000);
    flops is GFLOPs/tile (so scale appropriately when summing over N tiles).
    Both numbers come from ``src.experiments.speed.benchmark`` runs.
    """

    name: str
    flops_g: float
    latency_ms: float


@dataclass
class ComputeAccountant:
    """Sum gate + detector costs for an aggregate cascade run."""

    gate: StageCost
    detector: StageCost

    def total(self, n_tiles: int, n_accepted: int) -> dict[str, float]:
        gate_g = self.gate.flops_g * n_tiles
        det_g = self.detector.flops_g * n_accepted
        gate_ms = self.gate.latency_ms * n_tiles
        det_ms = self.detector.latency_ms * n_accepted
        return {
            "gate_gflops": gate_g,
            "detector_gflops": det_g,
            "total_gflops": gate_g + det_g,
            "gate_latency_ms": gate_ms,
            "detector_latency_ms": det_ms,
            "total_latency_ms": gate_ms + det_ms,
            "n_tiles": int(n_tiles),
            "n_accepted": int(n_accepted),
            "filter_rate": float(1.0 - n_accepted / max(n_tiles, 1)),
        }


# ---------- One-row Pareto helper ------------------------------------------


def cascade_pareto_row(
    tile_scores: list[TileScore],
    detections: dict[str, TileDetections],
    ground_truth: dict[str, TileGroundTruth],
    threshold: float,
    accountant: ComputeAccountant,
    label: str,
    _precomputed: dict[str, _PrecomputedTile] | None = None,
) -> dict[str, Any]:
    outputs = apply_threshold(tile_scores, detections, threshold)
    n = len(outputs)
    n_acc = sum(1 for o in outputs if o.accepted)
    map_metrics = evaluate_obb_map(
        outputs,
        ground_truth,
        iou_thresholds=tuple(float(f"{v:.2f}") for v in np.linspace(0.5, 0.95, 10)),
        _precomputed=_precomputed,
    )
    compute = accountant.total(n, n_acc)
    tile_label_map = {ts.tile_id: ts.label for ts in tile_scores}
    return {
        "label": label,
        "threshold": float(threshold),
        "filter_rate": filter_rate(outputs),
        "gate_recall_on_positive_tiles": gate_recall(outputs, tile_label_map),
        **map_metrics,
        **compute,
    }


def cascade_pareto_sweep(
    tile_scores: list[TileScore],
    detections: dict[str, TileDetections],
    ground_truth: dict[str, TileGroundTruth],
    accountant: ComputeAccountant,
    label: str,
    n_thresholds: int = 21,
    calibration: str = "identity",
) -> list[dict[str, Any]]:
    """Sweep thresholds and return all Pareto rows, precomputing IoU once.

    This is the fast entry point for threshold sweeps — shapely is called only
    once (in ``precompute_iou``) rather than once per threshold.
    """
    print(f"[eval_cascade] precomputing IoU matrices for {label} ...", flush=True)
    precomputed = precompute_iou(detections, ground_truth)
    print(f"[eval_cascade] IoU precomputed for {len(precomputed)} tiles; sweeping {n_thresholds} thresholds ...", flush=True)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    rows = []
    for t in thresholds:
        row = cascade_pareto_row(tile_scores, detections, ground_truth, float(t), accountant, label, _precomputed=precomputed)
        row["calibration"] = calibration
        rows.append(row)
        print(f"  t={t:.3f} filter={row['filter_rate']:.3f} gate_rec={row['gate_recall_on_positive_tiles']:.3f} mAP@0.5={row.get('mAP@0.50', 0):.4f} GFLOPs={row['total_gflops']:.1f}", flush=True)
    return rows


# ---------- Oracle gate ----------------------------------------------------


def oracle_tile_scores(
    metadata_jsonl: str | Path,
    split_filter: Iterable[str] | None = None,
) -> list[TileScore]:
    """Synthesize a perfect gate from the prepare_dota metadata. Probability
    is 1.0 on positives, 0.0 on negatives. Threshold any value in (0, 1) to
    accept exactly the positives.
    """
    label_map = load_tile_labels_from_metadata(metadata_jsonl)
    out: list[TileScore] = []
    splits = set(split_filter) if split_filter else None
    if splits is not None:
        for line in Path(metadata_jsonl).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") not in splits:
                continue
            tid = row["tile_id"]
            lbl = int(row.get("is_positive", 0))
            out.append(TileScore(tile_id=tid, label=lbl, prob=float(lbl)))
    else:
        for tid, lbl in label_map.items():
            out.append(TileScore(tile_id=tid, label=int(lbl), prob=float(lbl)))
    return out


# ---------- Convenience loader -------------------------------------------


def assemble_for_evaluation(
    score_jsonl: str | Path,
    detector_runs_dir: str | Path,
    gt_labels_dir: str | Path,
    detector_image_root: str | Path,
    tile_size: int = 1024,
    keep_classes: set[int] | None = None,
):
    """One-call helper: load tile scores, detector outputs, and ground truth.

    Returns a tuple ``(tile_scores, detections, ground_truth)`` ready to feed
    into ``cascade_pareto_row``.
    """
    from src.cascade import load_detector_outputs_yolo_obb

    tile_scores = load_tile_scores(score_jsonl)
    detections = load_detector_outputs_yolo_obb(detector_runs_dir, detector_image_root)
    ground_truth = load_yolo_obb_ground_truth(
        gt_labels_dir, tile_size=tile_size, keep_classes=keep_classes
    )
    return tile_scores, detections, ground_truth


# ---------- Stratified evaluation -----------------------------------------


def stratify_tiles_by(
    metadata_jsonl: str | Path,
    stratum: str,
    size_buckets: tuple[tuple[str, float, float], ...] = (
        ("<32px", 0.0, 32.0),
        ("32-96px", 32.0, 96.0),
        ("96-256px", 96.0, 256.0),
        (">256px", 256.0, float("inf")),
    ),
    gsd_buckets: tuple[tuple[str, float, float], ...] = (
        ("very_high", 0.0, 0.15),
        ("high", 0.15, 0.30),
        ("medium", 0.30, 0.60),
        ("low", 0.60, 1.50),
        ("very_low", 1.50, float("inf")),
    ),
) -> dict[str, set[str]]:
    """Group tile_ids by a stratum drawn from prepare_dota's tiles.jsonl.

    Strata supported:
      - ``imagesource``: bucket = the imagesource string ('GoogleEarth', 'GF-2', ...)
      - ``gsd_bucket``: discretized GSD buckets
      - ``size_bucket``: by max object size on tile (negatives get '<32px' too)
      - ``boundary``: 'interior' (no objects within 16px of the tile edge) vs
        'boundary' (at least one such object). Approximated from polygon
        bounding boxes in tiles.jsonl by checking whether ``max_obj_size_px``
        is large enough that the object likely touches an edge — a heuristic
        proxy; refine in Phase 4 if needed.
      - ``split``: train/val/test
    """
    out: dict[str, set[str]] = {}
    for line in Path(metadata_jsonl).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tid = str(row["tile_id"])
        bucket: str
        if stratum == "imagesource":
            bucket = str(row.get("imagesource") or "unknown")
        elif stratum == "gsd_bucket":
            gsd = row.get("gsd")
            try:
                gsd_v = float(gsd)
            except (TypeError, ValueError):
                gsd_v = None
            bucket = "unknown"
            if gsd_v is not None:
                for name, lo, hi in gsd_buckets:
                    if lo <= gsd_v < hi:
                        bucket = name
                        break
        elif stratum == "size_bucket":
            size = float(row.get("max_obj_size_px", 0.0))
            bucket = size_buckets[0][0]
            for name, lo, hi in size_buckets:
                if lo <= size < hi:
                    bucket = name
                    break
        elif stratum == "boundary":
            # Approximate: tiles whose largest object spans more than ~70% of
            # the tile dimension are likely tile-boundary cases. The exact
            # boundary metric needs polygon coords (we have only the size in
            # the JSONL row); refine when needed.
            size = float(row.get("max_obj_size_px", 0.0))
            tile_w = float(row.get("width", 1024)) or 1024.0
            bucket = "boundary" if size >= 0.7 * tile_w else "interior"
        elif stratum == "split":
            bucket = str(row.get("split", "unknown"))
        else:
            raise ValueError(f"Unknown stratum: {stratum}")
        out.setdefault(bucket, set()).add(tid)
    return out


def evaluate_stratified(
    cascade_outputs: list[CascadeOutput],
    ground_truth: dict[str, TileGroundTruth],
    metadata_jsonl: str | Path,
    stratum: str,
    iou_thresholds: tuple[float, ...] | None = None,
    tile_label_map: dict[str, int] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute mAP + gating metrics within each bucket of the stratum.

    Restricts both the cascade outputs and the ground truth to tile_ids in
    each bucket, then runs ``evaluate_obb_map`` plus per-bucket filter_rate
    and gate_recall. Used to populate the supplementary stratified Pareto
    tables (research plan §7.4 / §11).
    """
    if iou_thresholds is None:
        iou_thresholds = tuple(float(f"{v:.2f}") for v in np.linspace(0.5, 0.95, 10))
    buckets = stratify_tiles_by(metadata_jsonl, stratum)
    results: dict[str, dict[str, float]] = {}
    for bucket_name, tile_ids in sorted(buckets.items()):
        sub_outputs = [o for o in cascade_outputs if o.tile_id in tile_ids]
        sub_gt = {k: v for k, v in ground_truth.items() if k in tile_ids}
        n_tiles = len(sub_outputs)
        n_gt = len(sub_gt)
        if not sub_outputs or not sub_gt:
            results[bucket_name] = {"n_tiles": n_tiles, "n_gt_tiles": n_gt, "skipped": 1.0}
            continue
        m = evaluate_obb_map(sub_outputs, sub_gt, iou_thresholds=iou_thresholds)
        m["n_tiles"] = n_tiles
        m["n_gt_tiles"] = n_gt
        m["filter_rate"] = float(filter_rate(sub_outputs))
        if tile_label_map is not None:
            sub_label_map = {tid: tile_label_map[tid] for tid in tile_ids if tid in tile_label_map}
            if sub_label_map:
                m["gate_recall_on_positive_tiles"] = float(gate_recall(sub_outputs, sub_label_map))
        results[bucket_name] = m
    return results
