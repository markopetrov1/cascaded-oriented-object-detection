"""Image-level cascade evaluation for the round-1 revision.

The first submission scored every operating point at tile level: a tile's
detections were matched against that tile's ground truth, objects in the
200-pixel overlaps counted once per tile, and nothing was merged. Reviewers
asked for object-level recall *after post-processing*, so this module evaluates
the way a deployed cascade would be scored:

1. detections from the accepted tiles of an image are shifted into image
   coordinates and merged per class with polygon NMS (IoU 0.3);
2. the merged detections are matched against the image's DOTA ground truth with
   the DOTA devkit rule: each detection takes the ground-truth polygon it
   overlaps most; above the IoU threshold it is a true positive if that object
   is not difficult and not yet claimed, a false positive if already claimed,
   and ignored if the object is difficult;
3. AP is computed per class (VOC07 11-point as in the DOTA devkit, and the
   101-point COCO interpolation), averaged over the classes present.

Everything that does not depend on the operating point is computed once and
cached: polygon IoUs between detections and ground truth, and the NMS
suppression graph between detections. Because the devkit rule picks the
best-overlapping object regardless of whether it is already claimed, each
detection's candidate object and IoU are fixed, so matching at an operating
point reduces to "the first kept detection per object wins", which is
vectorised. Only the NMS pass is a loop, and it visits only detections that can
suppress another one.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import shapely

CLASSES = [
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge", "large vehicle",
    "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool",
    "container crane",
]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
STEM_RE = re.compile(r"^(?P<src>.+)__x(?P<x>\d+)_y(?P<y>\d+)_w(?P<w>\d+)_h(?P<h>\d+)$")
IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * k, 2) for k in range(10))


def parse_stem(stem: str) -> tuple[str, int, int, int, int]:
    """`P0003__x123_y0_w1024_h1023` -> ("P0003", 123, 0, 1024, 1023). Untiled
    images (HRSC2016) map to themselves at offset zero with unknown size."""
    m = STEM_RE.match(stem)
    if not m:
        return stem, 0, 0, 0, 0
    return m["src"], int(m["x"]), int(m["y"]), int(m["w"]), int(m["h"])


def to_polygons(corners: np.ndarray) -> np.ndarray:
    """(N, 4, 2) corner arrays -> shapely polygons. Convex hulls, because a few
    DOTA ground-truth quadrilaterals are self-intersecting in their vertex order
    and the devkit's polygon IoU treats every box as a convex quadrilateral."""
    if len(corners) == 0:
        return np.empty(0, dtype=object)
    closed = np.concatenate([corners, corners[:, :1]], axis=1)
    return shapely.convex_hull(shapely.polygons(closed))


def pairwise_iou(a: np.ndarray, b: np.ndarray, min_iou: float = 0.0
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index pairs (i, j) and IoU for every intersecting a[i], b[j] with IoU > min_iou."""
    if len(a) == 0 or len(b) == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e, np.empty(0)
    tree = shapely.STRtree(b)
    ia, ib = tree.query(a, predicate="intersects")
    if len(ia) == 0:
        return ia, ib, np.empty(0)
    inter = shapely.area(shapely.intersection(a[ia], b[ib]))
    union = shapely.area(a[ia]) + shapely.area(b[ib]) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    keep = iou > min_iou
    return ia[keep], ib[keep], iou[keep]


# ---------------------------------------------------------------- loading ----

def load_tile_predictions(pred_dir: str | Path, tile_ids: Iterable[str],
                          classes: set[int] | None = None,
                          sizes: dict[str, tuple[int, int]] | None = None
                          ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """tile_id -> (cls, score, corners in tile pixels). Missing file = no detections.

    Tile size is read from the stem; untiled images (HRSC2016) need `sizes`.
    """
    labels = Path(pred_dir) / "labels"
    out = {}
    for tid in tile_ids:
        f = labels / f"{tid}.txt"
        _, _, _, w, h = parse_stem(tid)
        if w == 0:
            w, h = sizes[tid]
        rows = np.loadtxt(f, ndmin=2) if f.exists() and f.stat().st_size else np.empty((0, 10))
        if rows.shape[1] < 10:
            rows = np.empty((0, 10))
        if classes is not None and len(rows):
            rows = rows[np.isin(rows[:, 0].astype(int), list(classes))]
        corners = rows[:, 1:9].reshape(-1, 4, 2) * np.array([w, h], dtype=float)
        out[tid] = (rows[:, 0].astype(np.int64), rows[:, 9].astype(float), corners)
    return out


def load_dota_image_gt(label_dir: str | Path, images: Iterable[str]
                       ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """image -> (cls, corners, difficult) from raw DOTA labelTxt files."""
    out = {}
    for img in images:
        cls, corners, diff = [], [], []
        f = Path(label_dir) / f"{img}.txt"
        for line in f.read_text().splitlines() if f.exists() else []:
            parts = line.split()
            if len(parts) < 9:
                continue
            name = parts[8].replace("-", " ")
            if name not in CLASS_ID:
                continue
            cls.append(CLASS_ID[name])
            corners.append(np.array(parts[:8], dtype=float).reshape(4, 2))
            diff.append(len(parts) > 9 and parts[9] == "1")
        out[img] = (np.array(cls, dtype=np.int64),
                    np.array(corners).reshape(-1, 4, 2) if corners else np.empty((0, 4, 2)),
                    np.array(diff, dtype=bool))
    return out


def load_yolo_gt(label_dir: str | Path, items: Iterable[str], sizes: dict[str, tuple[int, int]]
                 ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """item -> (cls, corners px, difficult=False) from normalised YOLO-OBB labels."""
    out = {}
    for it in items:
        f = Path(label_dir) / f"{it}.txt"
        rows = np.loadtxt(f, ndmin=2) if f.exists() and f.stat().st_size else np.empty((0, 9))
        w, h = sizes[it]
        corners = rows[:, 1:9].reshape(-1, 4, 2) * np.array([w, h], dtype=float)
        out[it] = (rows[:, 0].astype(np.int64), corners, np.zeros(len(rows), dtype=bool))
    return out


# ------------------------------------------------------------------ cache ----

@dataclass
class EvalCache:
    """Everything about one detector on one image set that no operating point changes.

    Detections are global arrays sorted by (image, score descending). `tile`
    indexes into `tiles` (the gate's unit of decision). `best_gt` is the global
    index of the ground-truth object a detection overlaps most among same-class
    objects in its image (-1 if none) and `best_iou` that overlap.
    `sup_ptr`/`sup_idx` is a CSR list of the detections each detection would
    suppress in NMS (same class, same image, lower score, IoU > nms_iou).
    """
    tiles: list[str]
    tile_image: np.ndarray          # tile -> image index
    images: list[str]
    det_image: np.ndarray
    det_tile: np.ndarray
    det_cls: np.ndarray
    det_score: np.ndarray
    det_corners: np.ndarray         # image pixel coordinates, (N, 4, 2)
    best_gt: np.ndarray
    best_iou: np.ndarray
    gt_image: np.ndarray
    gt_cls: np.ndarray
    gt_difficult: np.ndarray
    gt_corners: np.ndarray
    gt_tiles_ptr: np.ndarray        # CSR: tiles that contain each object (>= 10 % of its area)
    gt_tiles_idx: np.ndarray
    sup_ptr: np.ndarray
    sup_idx: np.ndarray
    suppressors: np.ndarray         # detections with at least one out-edge, in global score order
    nms_iou: float = 0.3
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        arrays = {k: v for k, v in self.__dict__.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(path, **arrays,
                            _tiles=np.array(self.tiles), _images=np.array(self.images),
                            _meta=np.array(json.dumps(dict(self.meta, nms_iou=self.nms_iou))))

    @classmethod
    def load(cls, path: str | Path) -> "EvalCache":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["_meta"]))
        kw = {k: z[k] for k in z.files if not k.startswith("_")}
        return cls(tiles=[str(t) for t in z["_tiles"]], images=[str(i) for i in z["_images"]],
                   nms_iou=meta.pop("nms_iou"), meta=meta, **kw)


def build_cache(tile_preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
                image_gt: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
                tile_windows: dict[str, tuple[str, int, int, int, int]],
                nms_iou: float = 0.3, min_contained: float = 0.1, meta: dict | None = None) -> EvalCache:
    """tile_windows: tile_id -> (image, x, y, w, h) in image pixels."""
    tiles = sorted(tile_windows)
    images = sorted({tile_windows[t][0] for t in tiles})
    img_index = {im: k for k, im in enumerate(images)}
    tile_image = np.array([img_index[tile_windows[t][0]] for t in tiles], dtype=np.int64)

    d_img, d_tile, d_cls, d_score, d_corn = [], [], [], [], []
    for ti, t in enumerate(tiles):
        c, s, corners = tile_preds.get(t, (np.empty(0, np.int64), np.empty(0), np.empty((0, 4, 2))))
        if len(c) == 0:
            continue
        _, x, y, _, _ = tile_windows[t]
        d_img.append(np.full(len(c), tile_image[ti])); d_tile.append(np.full(len(c), ti))
        d_cls.append(c); d_score.append(s); d_corn.append(corners + np.array([x, y], dtype=float))
    det_image = np.concatenate(d_img) if d_img else np.empty(0, np.int64)
    det_tile = np.concatenate(d_tile) if d_tile else np.empty(0, np.int64)
    det_cls = np.concatenate(d_cls) if d_cls else np.empty(0, np.int64)
    det_score = np.concatenate(d_score) if d_score else np.empty(0)
    det_corners = np.concatenate(d_corn) if d_corn else np.empty((0, 4, 2))
    order = np.lexsort((-det_score, det_image))
    det_image, det_tile, det_cls, det_score, det_corners = (
        det_image[order], det_tile[order], det_cls[order], det_score[order], det_corners[order])

    g_img, g_cls, g_diff, g_corn = [], [], [], []
    for im in images:
        c, corners, diff = image_gt.get(im, (np.empty(0, np.int64), np.empty((0, 4, 2)), np.empty(0, bool)))
        g_img.append(np.full(len(c), img_index[im])); g_cls.append(c); g_diff.append(diff); g_corn.append(corners)
    gt_image = np.concatenate(g_img).astype(np.int64)
    gt_cls = np.concatenate(g_cls).astype(np.int64)
    gt_difficult = np.concatenate(g_diff).astype(bool)
    gt_corners = np.concatenate(g_corn).reshape(-1, 4, 2) if len(gt_cls) else np.empty((0, 4, 2))

    best_gt = np.full(len(det_score), -1, dtype=np.int64)
    best_iou = np.zeros(len(det_score))
    edges_hi, edges_lo = [], []
    gt_tiles = [[] for _ in range(len(gt_cls))]
    det_bounds = np.searchsorted(det_image, np.arange(len(images) + 1))
    gt_bounds = np.searchsorted(gt_image, np.arange(len(images) + 1)) if len(gt_image) else np.zeros(len(images) + 1, int)
    tile_bounds = {}
    for ti, t in enumerate(tiles):
        tile_bounds.setdefault(tile_image[ti], []).append(ti)

    for k in range(len(images)):
        d0, d1 = det_bounds[k], det_bounds[k + 1]
        g0, g1 = gt_bounds[k], gt_bounds[k + 1]
        dpoly = to_polygons(det_corners[d0:d1])
        gpoly = to_polygons(gt_corners[g0:g1])
        # objects -> containing tiles (the tiling rule: >= 10 % of the polygon inside)
        if g1 > g0:
            garea = shapely.area(gpoly)
            for ti in tile_bounds.get(k, []):
                _, x, y, w, h = tile_windows[tiles[ti]]
                if w == 0:   # untiled: the image is the tile
                    for gi in range(g1 - g0):
                        gt_tiles[g0 + gi].append(ti)
                    continue
                inside = shapely.area(shapely.intersection(gpoly, shapely.box(x, y, x + w, y + h)))
                for gi in np.nonzero(inside >= min_contained * np.maximum(garea, 1e-9))[0]:
                    gt_tiles[g0 + gi].append(ti)
        if d1 == d0:
            continue
        # detection -> best same-class object
        ia, ib, iou = pairwise_iou(dpoly, gpoly)
        same = det_cls[d0 + ia] == gt_cls[g0 + ib]
        ia, ib, iou = ia[same], ib[same], iou[same]
        if len(ia):
            o = np.lexsort((-iou, ia))
            first = np.r_[True, ia[o][1:] != ia[o][:-1]]
            best_gt[d0 + ia[o][first]] = g0 + ib[o][first]
            best_iou[d0 + ia[o][first]] = iou[o][first]
        # NMS suppression graph (within the image and class; higher score suppresses lower)
        ja, jb, jiou = pairwise_iou(dpoly, dpoly, min_iou=nms_iou)
        sel = (ja < jb) & (det_cls[d0 + ja] == det_cls[d0 + jb])   # sorted by score: lower index = higher score
        edges_hi.append(d0 + ja[sel]); edges_lo.append(d0 + jb[sel])

    hi = np.concatenate(edges_hi) if edges_hi else np.empty(0, np.int64)
    lo = np.concatenate(edges_lo) if edges_lo else np.empty(0, np.int64)
    o = np.argsort(hi, kind="stable")
    hi, lo = hi[o], lo[o]
    sup_ptr = np.searchsorted(hi, np.arange(len(det_score) + 1)).astype(np.int64)
    has_out = np.nonzero(np.diff(sup_ptr) > 0)[0]
    suppressors = has_out[np.argsort(-det_score[has_out], kind="stable")]
    gt_tiles_ptr = np.zeros(len(gt_cls) + 1, dtype=np.int64)
    gt_tiles_ptr[1:] = np.cumsum([len(x) for x in gt_tiles])
    gt_tiles_idx = np.array([t for x in gt_tiles for t in x], dtype=np.int64)
    return EvalCache(tiles=tiles, tile_image=tile_image, images=images, det_image=det_image,
                     det_tile=det_tile, det_cls=det_cls, det_score=det_score, det_corners=det_corners,
                     best_gt=best_gt, best_iou=best_iou, gt_image=gt_image, gt_cls=gt_cls,
                     gt_difficult=gt_difficult, gt_corners=gt_corners, gt_tiles_ptr=gt_tiles_ptr,
                     gt_tiles_idx=gt_tiles_idx, sup_ptr=sup_ptr, sup_idx=lo, suppressors=suppressors,
                     nms_iou=nms_iou, meta=meta or {})


# ------------------------------------------------------------- evaluation ----

def merged_keep(cache: EvalCache, accepted_tiles: np.ndarray, det_mask: np.ndarray | None = None) -> np.ndarray:
    """Detections surviving image-level NMS when only `accepted_tiles` were run."""
    alive = accepted_tiles[cache.det_tile].copy()
    if det_mask is not None:
        alive &= det_mask
    ptr, idx = cache.sup_ptr, cache.sup_idx
    for d in cache.suppressors:
        if alive[d]:
            alive[idx[ptr[d]:ptr[d + 1]]] = False
    return alive


def _ap_from_flags(scores: np.ndarray, tp: np.ndarray, valid: np.ndarray, n_pos: int
                   ) -> tuple[float, float]:
    """(VOC07 11-point AP, COCO 101-point AP) from per-detection TP flags.
    `valid` False marks detections matched to difficult objects (ignored)."""
    if n_pos == 0:
        return float("nan"), float("nan")
    s, t = scores[valid], tp[valid]
    if len(s) == 0:
        return 0.0, 0.0
    o = np.argsort(-s, kind="stable")
    t = t[o]
    ctp = np.cumsum(t); cfp = np.cumsum(~t)
    rec = ctp / n_pos
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    ap11 = np.mean([prec[rec >= r].max() if (rec >= r).any() else 0.0 for r in np.linspace(0, 1, 11)])
    ap101 = np.mean([prec[rec >= r].max() if (rec >= r).any() else 0.0 for r in np.linspace(0, 1, 101)])
    return float(ap11), float(ap101)


def match(cache: EvalCache, keep: np.ndarray, iou_thr: float, classes: Iterable[int],
          images: np.ndarray | None = None) -> dict:
    """DOTA-devkit matching of the kept detections. Returns per-detection flags and per-object hits.

    `images`: optional boolean mask over images restricting the evaluation (folds, bootstrap).
    """
    classes = np.asarray(list(classes))
    dsel = keep & np.isin(cache.det_cls, classes)
    gsel = np.isin(cache.gt_cls, classes)
    if images is not None:
        dsel &= images[cache.det_image]
        gsel &= images[cache.gt_image]
    d = np.nonzero(dsel)[0]
    bg = cache.best_gt[d]
    over = (bg >= 0) & (cache.best_iou[d] >= iou_thr)
    difficult = np.zeros(len(d), bool)
    difficult[over] = cache.gt_difficult[bg[over]]
    valid = ~(over & difficult)
    cand = over & ~difficult
    tp = np.zeros(len(d), bool)
    if cand.any():
        ci = np.nonzero(cand)[0]
        o = ci[np.argsort(-cache.det_score[d[ci]], kind="stable")]
        _, first = np.unique(bg[o], return_index=True)
        tp[o[first]] = True
    hit = np.zeros(len(cache.gt_cls), bool)
    hit[bg[tp]] = True
    return {"det": d, "tp": tp, "valid": valid, "gt_hit": hit, "gt_sel": gsel & ~cache.gt_difficult}


def evaluate(cache: EvalCache, accepted_tiles: np.ndarray, classes: Iterable[int],
             images: np.ndarray | None = None, iou_thresholds: Iterable[float] = (0.5,),
             det_mask: np.ndarray | None = None) -> dict:
    """mAP (VOC07 and COCO-101) per IoU threshold, averaged over classes with objects."""
    keep = merged_keep(cache, accepted_tiles, det_mask)
    classes = list(classes)
    iou_thresholds = list(iou_thresholds)
    out = {}
    for thr in iou_thresholds:
        m = match(cache, keep, thr, classes, images)
        ap11, ap101 = [], []
        for c in classes:
            dc = cache.det_cls[m["det"]] == c
            n_pos = int((m["gt_sel"] & (cache.gt_cls == c)).sum())
            if n_pos == 0:
                continue
            a, b = _ap_from_flags(cache.det_score[m["det"]][dc], m["tp"][dc], m["valid"][dc], n_pos)
            ap11.append(a); ap101.append(b)
        out[f"mAP{int(round(thr * 100))}_voc07"] = float(np.mean(ap11)) if ap11 else float("nan")
        out[f"mAP{int(round(thr * 100))}_coco"] = float(np.mean(ap101)) if ap101 else float("nan")
    if len(iou_thresholds) == 10:
        out["mAP50_95_coco"] = float(np.mean([out[f"mAP{int(round(t * 100))}_coco"] for t in iou_thresholds]))
    return out


def object_recall(cache: EvalCache, accepted_tiles: np.ndarray, classes: Iterable[int],
                  conf: float, iou_thr: float = 0.5, images: np.ndarray | None = None) -> np.ndarray:
    """Boolean per object: detected at confidence >= conf after merging the accepted tiles."""
    return match(cache, merged_keep(cache, accepted_tiles, cache.det_score >= conf), iou_thr,
                 classes, images)["gt_hit"]


def objects_all_tiles_rejected(cache: EvalCache, accepted_tiles: np.ndarray) -> np.ndarray:
    """True for objects every containing tile of which was rejected."""
    n_tiles = np.diff(cache.gt_tiles_ptr)
    owner = np.repeat(np.arange(len(n_tiles)), n_tiles)
    n_acc = np.bincount(owner, weights=accepted_tiles[cache.gt_tiles_idx].astype(float),
                        minlength=len(n_tiles))
    return (n_tiles > 0) & (n_acc == 0)
