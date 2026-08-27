"""HRSC2016 ship-detection dataset adapter.

HRSC2016 ships in PASCAL VOC style XML with both axis-aligned and rotated box
annotations. We parse the rotated boxes (``robndbox``) and emit the same
on-disk layout as our DOTA-Ships pipeline so the cascade can be evaluated
zero-shot:

  data/processed/hrsc2016/images/test/<id>.png
  data/processed/hrsc2016/labels/test/<id>.txt   # YOLO-OBB (single class: ship)
  data/processed/hrsc2016/gate_labels/test/<id>.txt  # binary
  data/processed/hrsc2016/metadata/tiles.jsonl

Used for the cross-dataset robustness check in research plan §9.7. We do not
tile HRSC images; they are typically <2k pixels per side and ships are
prominent enough that direct evaluation matches the spirit of the test.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils.paths import ensure_dir, write_yaml


@dataclass(frozen=True)
class HRSC2016Box:
    cx: float
    cy: float
    w: float
    h: float
    angle: float  # radians
    class_id: int = 0


def _polygon_from_robndbox(box: HRSC2016Box) -> np.ndarray:
    """Convert (cx, cy, w, h, angle) to a 4x2 polygon in pixel coords."""
    cos_a, sin_a = math.cos(box.angle), math.sin(box.angle)
    half_w, half_h = box.w / 2.0, box.h / 2.0
    corners = np.asarray(
        [[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]],
        dtype=np.float32,
    )
    R = np.asarray([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    rotated = corners @ R.T
    rotated[:, 0] += box.cx
    rotated[:, 1] += box.cy
    return rotated


def _parse_xml(path: Path) -> tuple[int, int, list[HRSC2016Box]]:
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()
    width = int(float(root.findtext("Img_SizeWidth", "0") or root.findtext("size/width", "0")))
    height = int(float(root.findtext("Img_SizeHeight", "0") or root.findtext("size/height", "0")))
    boxes: list[HRSC2016Box] = []
    # HRSC2016 layout: <HRSC_Objects><HRSC_Object><box_xmin>...; the rotated boxes
    # use <mbox_cx>, <mbox_cy>, <mbox_w>, <mbox_h>, <mbox_ang>.
    for obj in root.iter("HRSC_Object"):
        cx = float(obj.findtext("mbox_cx", "0") or 0)
        cy = float(obj.findtext("mbox_cy", "0") or 0)
        w = float(obj.findtext("mbox_w", "0") or 0)
        h = float(obj.findtext("mbox_h", "0") or 0)
        ang = float(obj.findtext("mbox_ang", "0") or 0)
        if w > 0 and h > 0:
            boxes.append(HRSC2016Box(cx=cx, cy=cy, w=w, h=h, angle=ang, class_id=0))
    if not boxes:
        # Fallback: VOC-style robndbox under <annotation><object>.
        for obj in root.iter("object"):
            r = obj.find("robndbox")
            if r is None:
                continue
            cx = float(r.findtext("cx", "0") or 0)
            cy = float(r.findtext("cy", "0") or 0)
            w = float(r.findtext("w", "0") or 0)
            h = float(r.findtext("h", "0") or 0)
            ang = float(r.findtext("angle", "0") or 0)
            if w > 0 and h > 0:
                boxes.append(HRSC2016Box(cx=cx, cy=cy, w=w, h=h, angle=ang, class_id=0))
    return width, height, boxes


def prepare_hrsc2016(
    raw_dir: str | Path,
    out_dir: str | Path,
    dataset_yaml: str | Path = "configs/datasets/hrsc2016_ships.yaml",
    splits: tuple[str, ...] = ("test",),
) -> dict[str, int]:
    """Convert HRSC2016 to our cascade-ready layout.

    Expected raw layout::

        raw_dir/AllImages/<id>.bmp (or .jpg)
        raw_dir/Annotations/<id>.xml
        raw_dir/ImageSets/{train,val,trainval,test}.txt   # split membership

    For zero-shot cascade evaluation, only ``test`` is required; pass
    ``splits=("test",)``.
    """
    import cv2

    raw = Path(raw_dir)
    out = Path(out_dir)
    metadata_path = out / "metadata" / "tiles.jsonl"
    ensure_dir(metadata_path.parent)
    if metadata_path.exists():
        metadata_path.unlink()

    image_dir_candidates = [raw / "AllImages", raw / "images"]
    image_dir = next((p for p in image_dir_candidates if p.exists()), None)
    if image_dir is None:
        raise FileNotFoundError(f"Could not find HRSC images under {raw}")
    annotations_dir = raw / "Annotations"
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Missing HRSC annotations dir: {annotations_dir}")
    sets_dir = raw / "ImageSets"

    counts: dict[str, int] = {s: 0 for s in splits}
    with metadata_path.open("w", encoding="utf-8") as meta_handle:
        for split in splits:
            ensure_dir(out / "images" / split)
            ensure_dir(out / "labels" / split)
            ensure_dir(out / "gate_labels" / split)
            ids: list[str]
            split_file = sets_dir / f"{split}.txt"
            if split_file.exists():
                ids = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
            else:
                # If no splits file, take all images.
                ids = [p.stem for p in sorted(image_dir.iterdir()) if p.suffix.lower() in (".bmp", ".jpg", ".jpeg", ".png")]

            for image_id in ids:
                src_image = next(
                    (image_dir / f"{image_id}{ext}" for ext in (".bmp", ".jpg", ".jpeg", ".png") if (image_dir / f"{image_id}{ext}").exists()),
                    None,
                )
                if src_image is None:
                    print(f"[hrsc] WARNING: no image for {image_id}")
                    continue
                xml_path = annotations_dir / f"{image_id}.xml"
                if not xml_path.exists():
                    print(f"[hrsc] WARNING: no annotation for {image_id}")
                    continue
                width, height, boxes = _parse_xml(xml_path)
                img = cv2.imread(str(src_image), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"[hrsc] WARNING: unreadable image {src_image}")
                    continue
                if width <= 0 or height <= 0:
                    height, width = img.shape[:2]
                out_image = out / "images" / split / f"{image_id}.png"
                cv2.imwrite(str(out_image), img)
                out_label = out / "labels" / split / f"{image_id}.txt"
                lines: list[str] = []
                for box in boxes:
                    poly = _polygon_from_robndbox(box)
                    poly[:, 0] = np.clip(poly[:, 0], 0, width)
                    poly[:, 1] = np.clip(poly[:, 1], 0, height)
                    norm = poly.copy()
                    norm[:, 0] /= width
                    norm[:, 1] /= height
                    lines.append("0 " + " ".join(f"{v:.6f}" for v in norm.reshape(-1).tolist()))
                out_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                gate_label = out / "gate_labels" / split / f"{image_id}.txt"
                gate_label.write_text("1" if boxes else "0", encoding="utf-8")
                meta_handle.write(
                    json.dumps(
                        {
                            "tile_id": image_id,
                            "split": split,
                            "source_image": src_image.name,
                            "source_stem": image_id,
                            "tile_index": 0,
                            "x": 0,
                            "y": 0,
                            "width": width,
                            "height": height,
                            "is_positive": 1 if boxes else 0,
                            "num_positives": len(boxes),
                            "positive_class_counts": {"ship": len(boxes)} if boxes else {},
                            "num_objects": len(boxes),
                            "class_counts": {"ship": len(boxes)} if boxes else {},
                            "max_obj_size_px": float(max((max(b.w, b.h) for b in boxes), default=0.0)),
                            "imagesource": "HRSC2016",
                            "gsd": None,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                counts[split] += 1

    write_yaml(
        {
            "path": str(out.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "task": "obb",
            "names": {0: "ship"},
        },
        dataset_yaml,
    )
    return counts
