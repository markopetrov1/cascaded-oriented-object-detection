from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.paths import iter_images


@dataclass
class ValidationReport:
    images: int
    labels: int
    missing_labels: list[str]
    invalid_labels: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing_labels and not self.invalid_labels


def validate_yolo_dataset(root: str | Path, task: str = "detect") -> ValidationReport:
    root_path = Path(root)
    images = iter_images(root_path / "images")
    missing: list[str] = []
    invalid: list[str] = []
    label_count = 0
    expected_values = 5 if task == "detect" else 9

    for image in images:
        split = image.parent.name
        label = root_path / "labels" / split / f"{image.stem}.txt"
        if not label.exists():
            missing.append(str(image.relative_to(root_path)))
            continue
        label_count += 1
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != expected_values:
                invalid.append(f"{label}:{line_number}: expected {expected_values} values")
                continue
            try:
                class_id = int(parts[0])
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                invalid.append(f"{label}:{line_number}: non-numeric value")
                continue
            if class_id < 0:
                invalid.append(f"{label}:{line_number}: negative class id")
            if any(value < -1e-6 or value > 1 + 1e-6 for value in coords):
                invalid.append(f"{label}:{line_number}: coordinates outside [0, 1]")

    return ValidationReport(len(images), label_count, missing, invalid)


def raise_on_invalid(report: ValidationReport) -> None:
    if report.ok:
        return
    messages = []
    if report.missing_labels:
        preview = ", ".join(report.missing_labels[:5])
        messages.append(f"{len(report.missing_labels)} images missing labels: {preview}")
    if report.invalid_labels:
        preview = ", ".join(report.invalid_labels[:5])
        messages.append(f"{len(report.invalid_labels)} invalid labels: {preview}")
    raise ValueError("; ".join(messages))
