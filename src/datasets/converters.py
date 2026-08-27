from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DIOR_CLASSES = [
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "Expressway-Service-area",
    "Expressway-toll-station",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
]

DOTA_CLASSES = [
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
]

# DOTA-v2.0 adds three classes on top of v1.0/v1.5. Verify exact spelling against
# the v2.0 release on download; correct here if the release uses different tokens.
DOTA_V2_CLASSES = DOTA_CLASSES + [
    "container crane",
    "airport",
    "helipad",
]


@dataclass(frozen=True)
class HBBAnnotation:
    class_name: str
    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class OBBAnnotation:
    class_name: str
    class_id: int
    points: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
    difficult: int = 0


def class_mapping(names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(names)}


def voc_box_to_yolo(
    x_min: float, y_min: float, x_max: float, y_max: float, width: int, height: int
) -> tuple[float, float, float, float]:
    x_min = max(0.0, min(float(width), x_min))
    x_max = max(0.0, min(float(width), x_max))
    y_min = max(0.0, min(float(height), y_min))
    y_max = max(0.0, min(float(height), y_max))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid horizontal box with non-positive width or height")
    x_center = ((x_min + x_max) / 2.0) / width
    y_center = ((y_min + y_max) / 2.0) / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    return x_center, y_center, box_width, box_height


def parse_dior_xml(xml_path: str | Path, names: list[str] | None = None) -> tuple[int, int, list[HBBAnnotation]]:
    names = names or DIOR_CLASSES
    name_to_id = class_mapping(names)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")
    width = int(float(size.findtext("width", "0")))
    height = int(float(size.findtext("height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}: {width}x{height}")

    annotations: list[HBBAnnotation] = []
    for obj in root.findall("object"):
        class_name = (obj.findtext("name") or "").strip()
        if class_name not in name_to_id:
            raise ValueError(f"Unknown DIOR class '{class_name}' in {xml_path}")
        box = obj.find("bndbox")
        if box is None:
            continue
        annotations.append(
            HBBAnnotation(
                class_name=class_name,
                class_id=name_to_id[class_name],
                x_min=float(box.findtext("xmin", "0")),
                y_min=float(box.findtext("ymin", "0")),
                x_max=float(box.findtext("xmax", "0")),
                y_max=float(box.findtext("ymax", "0")),
            )
        )
    return width, height, annotations


def hbb_to_yolo_line(annotation: HBBAnnotation, width: int, height: int) -> str:
    xywh = voc_box_to_yolo(
        annotation.x_min, annotation.y_min, annotation.x_max, annotation.y_max, width, height
    )
    return f"{annotation.class_id} " + " ".join(f"{value:.6f}" for value in xywh)


def parse_dota_annotation_line(
    line: str,
    names: list[str] | None = None,
    strict_classes: bool = False,
) -> OBBAnnotation | None:
    """Parse a DOTA polygon annotation line.

    With ``strict_classes=True`` (legacy behavior) raise ``ValueError`` on an
    unknown class. With ``strict_classes=False`` (default since v0.2) return
    ``None`` so the caller can skip silently — the cascade pipeline only cares
    about the positive_classes filter, so unknown classes from a newer DOTA
    revision should not crash a v1.0/v1.5 run.
    """
    names = names or DOTA_CLASSES
    stripped = line.strip()
    if not stripped or stripped.startswith("imagesource") or stripped.startswith("gsd"):
        return None
    parts = stripped.split()
    if len(parts) < 9:
        return None
    coords = [float(value) for value in parts[:8]]
    class_name = " ".join(parts[8:-1]) if len(parts) > 9 else parts[8]
    difficult = int(float(parts[-1])) if len(parts) > 9 and parts[-1].replace(".", "", 1).isdigit() else 0
    name_to_id = class_mapping(names)
    if class_name not in name_to_id and class_name.replace("-", " ") in name_to_id:
        class_name = class_name.replace("-", " ")
    if class_name not in name_to_id:
        if strict_classes:
            raise ValueError(f"Unknown DOTA class '{class_name}' in line: {line}")
        return None
    points = tuple((coords[i], coords[i + 1]) for i in range(0, 8, 2))
    return OBBAnnotation(class_name, name_to_id[class_name], points, difficult)


def polygon_area(points: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def normalize_obb_points(
    points: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]],
    width: int,
    height: int,
) -> list[float]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    normalized: list[float] = []
    for x, y in points:
        nx = x / width
        ny = y / height
        if not (-1e-6 <= nx <= 1 + 1e-6 and -1e-6 <= ny <= 1 + 1e-6):
            raise ValueError(f"OBB point outside image bounds after normalization: {(nx, ny)}")
        normalized.extend([min(1.0, max(0.0, nx)), min(1.0, max(0.0, ny))])
    if polygon_area([(normalized[i], normalized[i + 1]) for i in range(0, 8, 2)]) <= 0:
        raise ValueError("Degenerate OBB polygon")
    return normalized


def obb_to_yolo_line(annotation: OBBAnnotation, width: int, height: int) -> str:
    normalized = normalize_obb_points(annotation.points, width, height)
    return f"{annotation.class_id} " + " ".join(f"{value:.6f}" for value in normalized)


def rect_from_polygon(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...] | None:
    if len(points) < 3 or polygon_area(points) <= 1.0:
        return None
    try:
        import cv2

        arr = np.array(points, dtype=np.float32)
        rect = cv2.minAreaRect(arr)
        box = cv2.boxPoints(rect)
        return tuple((float(x), float(y)) for x, y in box)
    except Exception:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if math.isclose(max(xs), min(xs)) or math.isclose(max(ys), min(ys)):
            return None
        return (
            (min(xs), min(ys)),
            (max(xs), min(ys)),
            (max(xs), max(ys)),
            (min(xs), max(ys)),
        )
