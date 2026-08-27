from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from src.datasets.converters import OBBAnnotation, polygon_area, rect_from_polygon


@dataclass(frozen=True)
class TileWindow:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


def tile_windows(width: int, height: int, tile_size: int, overlap: int) -> list[TileWindow]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be >= 0 and smaller than tile_size")
    stride = tile_size - overlap

    xs = list(range(0, max(width - tile_size, 0) + 1, stride))
    ys = list(range(0, max(height - tile_size, 0) + 1, stride))
    if not xs or xs[-1] + tile_size < width:
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] + tile_size < height:
        ys.append(max(0, height - tile_size))

    windows: list[TileWindow] = []
    seen: set[tuple[int, int]] = set()
    for y in ys:
        for x in xs:
            if (x, y) in seen:
                continue
            seen.add((x, y))
            windows.append(TileWindow(x, y, min(tile_size, width - x), min(tile_size, height - y)))
    return windows


def _inside(point: tuple[float, float], edge: str, value: float) -> bool:
    x, y = point
    if edge == "left":
        return x >= value
    if edge == "right":
        return x <= value
    if edge == "top":
        return y >= value
    if edge == "bottom":
        return y <= value
    raise ValueError(edge)


def _intersect(
    p1: tuple[float, float], p2: tuple[float, float], edge: str, value: float
) -> tuple[float, float]:
    x1, y1 = p1
    x2, y2 = p2
    if edge in {"left", "right"}:
        if x1 == x2:
            return value, y1
        t = (value - x1) / (x2 - x1)
        return value, y1 + t * (y2 - y1)
    if y1 == y2:
        return x1, value
    t = (value - y1) / (y2 - y1)
    return x1 + t * (x2 - x1), value


def clip_polygon_to_window(
    points: list[tuple[float, float]] | tuple[tuple[float, float], ...], window: TileWindow
) -> list[tuple[float, float]]:
    output = list(points)
    for edge, value in (
        ("left", window.x),
        ("right", window.x2),
        ("top", window.y),
        ("bottom", window.y2),
    ):
        if not output:
            break
        input_points = output
        output = []
        previous = input_points[-1]
        for current in input_points:
            current_inside = _inside(current, edge, value)
            previous_inside = _inside(previous, edge, value)
            if current_inside:
                if not previous_inside:
                    output.append(_intersect(previous, current, edge, value))
                output.append(current)
            elif previous_inside:
                output.append(_intersect(previous, current, edge, value))
            previous = current
    return output


def annotations_for_tile(
    annotations: list[OBBAnnotation], window: TileWindow, min_area_ratio: float = 0.1
) -> list[OBBAnnotation]:
    tile_annotations: list[OBBAnnotation] = []
    for annotation in annotations:
        original_area = polygon_area(annotation.points)
        clipped = clip_polygon_to_window(annotation.points, window)
        clipped_area = polygon_area(clipped)
        if original_area <= 0 or clipped_area / original_area < min_area_ratio:
            continue
        rect = rect_from_polygon([(x - window.x, y - window.y) for x, y in clipped])
        if rect is None:
            continue
        # cv2.minAreaRect of a clipped polygon can overshoot the tile bounds by
        # sub-pixel amounts (the rotated rect must contain all polygon vertices
        # but may extend past the clip box). Clamp so downstream YOLO
        # normalization stays in [0, 1].
        clamped = tuple(
            (
                max(0.0, min(float(window.width), float(x))),
                max(0.0, min(float(window.height), float(y))),
            )
            for x, y in rect
        )
        tile_annotations.append(
            OBBAnnotation(annotation.class_name, annotation.class_id, clamped, annotation.difficult)
        )
    return tile_annotations


def save_tile(image_path: str | Path, output_path: str | Path, window: TileWindow) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    tile = image[window.y : window.y2, window.x : window.x2]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), tile):
        raise ValueError(f"Could not write tile: {output_path}")
