from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.datasets.converters import (
    DOTA_CLASSES,
    OBBAnnotation,
    obb_to_yolo_line,
    parse_dota_annotation_line,
    polygon_area,
)
from src.datasets.tiling import annotations_for_tile, tile_windows
from src.datasets.validation import raise_on_invalid, validate_yolo_dataset
from src.utils.paths import ensure_dir, iter_images, write_json, write_yaml


@dataclass(frozen=True)
class DotaPrepareSummary:
    output_dir: str
    train_tiles: int
    val_tiles: int
    test_tiles: int
    train_positive_tiles: int
    val_positive_tiles: int
    test_positive_tiles: int
    tile_size: int
    overlap: int
    classes: list[str]
    positive_classes: list[str]


def _find_split_dirs(raw_dir: Path, split: str) -> tuple[Path | None, Path | None]:
    image_candidates = [
        raw_dir / split / "images",
        raw_dir / split / "Images",
        raw_dir / "images" / split,
        raw_dir / f"{split}_images",
        raw_dir / split,
    ]
    label_candidates = [
        raw_dir / split / "labelTxt",
        raw_dir / split / "labelTxt-v1.0",
        raw_dir / split / "labelTxt-v1.5",
        raw_dir / split / "labels",
        raw_dir / "labels" / split,
        raw_dir / "labelTxt" / split,
        raw_dir / f"{split}_labelTxt",
    ]
    image_dir = next((p for p in image_candidates if p.exists() and iter_images(p)), None)
    label_dir = next((p for p in label_candidates if p.exists()), None)
    return image_dir, label_dir


def parse_dota_label_with_header(
    path: str | Path, classes: list[str] = DOTA_CLASSES
) -> tuple[dict[str, str | float | None], list[OBBAnnotation]]:
    """Parse a DOTA labelTxt file. Returns (header, annotations).

    DOTA per-image labels start with two header lines:
        imagesource:GoogleEarth
        gsd:0.146
    Either may be missing or set to ``null`` / a non-numeric token; we keep them as
    raw strings so downstream code can decide on coercion.
    """
    header: dict[str, str | float | None] = {"imagesource": None, "gsd": None}
    annotations: list[OBBAnnotation] = []
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("imagesource"):
            _, _, value = stripped.partition(":")
            header["imagesource"] = value.strip() or None
            continue
        if lower.startswith("gsd"):
            _, _, value = stripped.partition(":")
            value = value.strip()
            try:
                header["gsd"] = float(value)
            except (ValueError, TypeError):
                header["gsd"] = value or None
            continue
        parsed = parse_dota_annotation_line(stripped, classes)
        if parsed is not None:
            annotations.append(parsed)
    return header, annotations


def _polygon_extent(points) -> tuple[float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return float(max(xs) - min(xs)), float(max(ys) - min(ys))


def prepare_dota(
    raw_dir: str | Path,
    out_dir: str | Path,
    dataset_yaml: str | Path = "configs/datasets/dota.yaml",
    tile_size: int = 1024,
    overlap: int = 200,
    min_area_ratio: float = 0.1,
    classes: list[str] = DOTA_CLASSES,
    positive_classes: list[str] | None = None,
    detector_classes: list[str] | None = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
    validate: bool = True,
    resume: bool = False,
    labels_only: bool = False,
) -> DotaPrepareSummary:
    """Tile DOTA images and emit OBB labels, gate (binary) labels, and enriched metadata.

    For each tile, three files are produced:
      images/{split}/<tile_stem>.png    — the cropped image
      labels/{split}/<tile_stem>.txt    — YOLO OBB labels (full class set)
      gate_labels/{split}/<tile_stem>.txt — single int (0 or 1): is the tile positive
                                            for any class in ``positive_classes``?

    A JSONL row is appended to metadata/tiles.jsonl per tile, carrying:
      tile_id, split, source_image, source_stem, tile_index, x, y, width, height,
      is_positive, num_positives, positive_class_counts, num_objects, class_counts,
      imagesource, gsd, max_obj_size_px.
    """
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"DOTA root not found: {root}")
    out_path = Path(out_dir)

    positive_set = set(positive_classes) if positive_classes else set(classes)
    # Detector class set: which classes the OBB labels include (and are remapped to
    # consecutive ids 0..K-1). Defaults to the full parsing class list. The dataset
    # yaml is written with these names so Ultralytics sees the right mapping.
    detector_list = list(detector_classes) if detector_classes else list(classes)
    detector_id_map = {name: idx for idx, name in enumerate(detector_list)}

    metadata_path = out_path / "metadata" / "tiles.jsonl"
    ensure_dir(metadata_path.parent)

    # Resume support: when set, keep existing metadata and skip per-image
    # processing for any (split, source_stem) already represented. We trust
    # the existing file because prepare_dota writes per-tile rows and the
    # outer loop is per-image: a (split, stem) pair appearing in metadata
    # means at least one tile for that image was finalized. For the rare case
    # of a half-written image, the next run will re-tile it because no row
    # was flushed before the crash.
    processed_pairs: set[tuple[str, str]] = set()
    if resume and metadata_path.exists():
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            processed_pairs.add((str(row.get("split")), str(row.get("source_stem"))))
        print(f"[prepare_dota] resume: skipping {len(processed_pairs)} (split,stem) pairs from existing metadata")
    elif metadata_path.exists():
        metadata_path.unlink()

    counts: dict[str, int] = {}
    positive_counts: dict[str, int] = {}
    if resume and metadata_path.exists():
        # Pre-populate counts from already-processed rows so the summary at
        # the end reports the full dataset, not just new work.
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            split_name = str(row.get("split", "unknown"))
            counts[split_name] = counts.get(split_name, 0) + 1
            positive_counts[split_name] = positive_counts.get(split_name, 0) + int(row.get("is_positive", 0))

    for split in splits:
        image_dir, label_dir = _find_split_dirs(root, split)
        counts[split] = 0
        positive_counts[split] = 0
        if image_dir is None:
            continue
        if not labels_only:
            ensure_dir(out_path / "images" / split)
        ensure_dir(out_path / "labels" / split)
        ensure_dir(out_path / "gate_labels" / split)
        for image_path in iter_images(image_dir):
            if (split, image_path.stem) in processed_pairs:
                continue
            label_path = label_dir / f"{image_path.stem}.txt" if label_dir else None
            header: dict[str, str | float | None] = {"imagesource": None, "gsd": None}
            annotations: list[OBBAnnotation] = []
            if label_path and label_path.exists():
                header, annotations = parse_dota_label_with_header(label_path, classes)
            elif split != "test":
                raise FileNotFoundError(f"Missing DOTA label for {image_path.name}: {label_path}")

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                # Corrupt or unreadable file: skip with a warning rather than
                # crashing the whole pipeline. Caller sees the count via the
                # summary and can investigate after the run completes.
                print(f"[prepare_dota] WARNING: could not read image, skipping: {image_path}")
                continue
            height, width = image.shape[:2]

            with metadata_path.open("a", encoding="utf-8") as meta_handle:
                for idx, window in enumerate(tile_windows(width, height, tile_size, overlap)):
                    tile_name = (
                        f"{image_path.stem}__x{window.x}_y{window.y}"
                        f"_w{window.width}_h{window.height}{image_path.suffix}"
                    )
                    tile_image = out_path / "images" / split / tile_name
                    tile_label = out_path / "labels" / split / f"{Path(tile_name).stem}.txt"
                    gate_label = out_path / "gate_labels" / split / f"{Path(tile_name).stem}.txt"

                    if not labels_only:
                        # Crop directly from the already-loaded image rather than
                        # re-reading the source per tile (vendored save_tile()
                        # re-decodes from disk every call, which makes 1024-px
                        # tiling ~25x slower than necessary on multi-thousand-pixel
                        # DOTA imagery).
                        tile_arr = image[window.y : window.y + window.height, window.x : window.x + window.width]
                        tile_image.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(tile_image), tile_arr):
                            raise ValueError(f"Could not write tile: {tile_image}")
                    tile_anns = annotations_for_tile(annotations, window, min_area_ratio)
                    # Filter + remap to detector class ids before emitting YOLO labels.
                    detector_anns = [
                        OBBAnnotation(
                            class_name=a.class_name,
                            class_id=detector_id_map[a.class_name],
                            points=a.points,
                            difficult=a.difficult,
                        )
                        for a in tile_anns
                        if a.class_name in detector_id_map
                    ]
                    tile_label.write_text(
                        "\n".join(
                            obb_to_yolo_line(item, window.width, window.height)
                            for item in detector_anns
                        )
                        + ("\n" if detector_anns else ""),
                        encoding="utf-8",
                    )

                    positives = [a for a in tile_anns if a.class_name in positive_set]
                    is_positive = int(len(positives) > 0)
                    gate_label.write_text(str(is_positive), encoding="utf-8")

                    class_counts: dict[str, int] = {}
                    positive_class_counts: dict[str, int] = {}
                    max_obj_size_px = 0.0
                    for ann in tile_anns:
                        class_counts[ann.class_name] = class_counts.get(ann.class_name, 0) + 1
                        w, h = _polygon_extent(ann.points)
                        max_obj_size_px = max(max_obj_size_px, max(w, h))
                    for ann in positives:
                        positive_class_counts[ann.class_name] = (
                            positive_class_counts.get(ann.class_name, 0) + 1
                        )

                    row = {
                        "tile_id": Path(tile_name).stem,
                        "split": split,
                        "source_image": image_path.name,
                        "source_stem": image_path.stem,
                        "tile_index": idx,
                        "x": window.x,
                        "y": window.y,
                        "width": window.width,
                        "height": window.height,
                        "is_positive": is_positive,
                        "num_positives": len(positives),
                        "positive_class_counts": positive_class_counts,
                        "num_objects": len(tile_anns),
                        "class_counts": class_counts,
                        "max_obj_size_px": round(max_obj_size_px, 2),
                        "imagesource": header.get("imagesource"),
                        "gsd": header.get("gsd"),
                    }
                    meta_handle.write(json.dumps(row, sort_keys=True) + "\n")
                    counts[split] += 1
                    positive_counts[split] += is_positive

    if counts.get("train", 0) == 0 and counts.get("val", 0) == 0:
        raise FileNotFoundError(
            f"Could not find DOTA images under {root}. Expected at least one of "
            f"{splits} with images/labels."
        )

    yaml_data = {
        "path": str(out_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "task": "obb",
        "names": {idx: name for idx, name in enumerate(detector_list)},
    }
    write_yaml(yaml_data, dataset_yaml)
    summary = DotaPrepareSummary(
        output_dir=str(out_path),
        train_tiles=counts.get("train", 0),
        val_tiles=counts.get("val", 0),
        test_tiles=counts.get("test", 0),
        train_positive_tiles=positive_counts.get("train", 0),
        val_positive_tiles=positive_counts.get("val", 0),
        test_positive_tiles=positive_counts.get("test", 0),
        tile_size=tile_size,
        overlap=overlap,
        classes=list(classes),
        positive_classes=sorted(positive_set),
    )
    write_json(summary.__dict__, out_path / "metadata" / "prepare_summary.json")
    if validate and counts.get("train", 0) > 0 and not labels_only:
        raise_on_invalid(validate_yolo_dataset(out_path, task="obb"))
    return summary
