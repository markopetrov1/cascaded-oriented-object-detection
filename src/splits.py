"""Geographic splits for DOTA tiles.

Random tile splits are unsafe in remote sensing: neighboring tiles from the same
source image leak between train/val/test. This module partitions on
``source_stem`` (whole source images go to one split) and stratifies by
``imagesource`` so each split sees each sensor.

Caveat — the public DOTA distribution does not carry literal-region metadata;
``imagesource`` is sensor (GoogleEarth, GF-2, JL-1, ...). Stem-disjointness
prevents tile-level leakage; sensor stratification is a proxy for distribution
match. The paper must state this caveat.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.utils.paths import write_yaml


def _greedy_balance(
    items: list[tuple[str, int]],  # (source_stem, weight)
    fractions: dict[str, float],
    rng: random.Random,
) -> dict[str, list[str]]:
    """Greedy LPT-style assignment: sort by descending weight, place each item
    into the split currently most below its target. Stable for small ties via shuffle.
    """
    rng.shuffle(items)
    items.sort(key=lambda kv: -kv[1])
    total = sum(w for _, w in items)
    targets = {split: total * frac for split, frac in fractions.items()}
    current = {split: 0.0 for split in fractions}
    assignment: dict[str, list[str]] = {split: [] for split in fractions}
    for stem, weight in items:
        deficits = {s: targets[s] - current[s] for s in fractions}
        chosen = max(deficits, key=lambda s: deficits[s])
        assignment[chosen].append(stem)
        current[chosen] += weight
    return assignment


def build_geographic_split(
    metadata_jsonl: str | Path,
    out_yaml: str | Path,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 67,
    weight_by: str = "tiles",  # "tiles" | "positives"
) -> dict[str, Any]:
    """Build a stem-disjoint, imagesource-stratified split.

    The strategy: bucket source stems by their dominant imagesource; within each
    bucket, run a greedy weighted assignment to hit the target fractions. The
    union per split is the final assignment. Weights default to tile counts;
    pass ``weight_by="positives"`` to balance on positive-tile counts (useful
    when sparsity is extreme and we want each split to have similar absolute
    positive volumes).
    """
    test_frac = round(1.0 - train_frac - val_frac, 6)
    if test_frac < 0 or train_frac < 0 or val_frac < 0:
        raise ValueError("Fractions must be non-negative and sum <= 1")
    fractions = {"train": train_frac, "val": val_frac, "test": test_frac}

    rows = [json.loads(l) for l in Path(metadata_jsonl).read_text().splitlines() if l.strip()]
    if not rows:
        raise ValueError(f"No tile metadata at {metadata_jsonl}")

    # Aggregate per (source_stem, imagesource).
    stem_meta: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"imagesource": None, "tiles": 0, "positives": 0}
    )
    for row in rows:
        stem = row["source_stem"]
        meta = stem_meta[stem]
        meta["tiles"] += 1
        meta["positives"] += int(row.get("is_positive", 0))
        if meta["imagesource"] is None:
            meta["imagesource"] = row.get("imagesource") or "unknown"

    # Bucket stems by imagesource.
    by_source: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for stem, meta in stem_meta.items():
        weight = max(1, meta[weight_by])
        by_source[str(meta["imagesource"])].append((stem, weight))

    rng = random.Random(seed)
    assignment: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for source, items in sorted(by_source.items()):
        local = _greedy_balance(items, fractions, rng)
        for split in fractions:
            assignment[split].extend(local[split])

    for split in assignment:
        assignment[split].sort()

    # Sanity: stem-disjointness.
    train_set = set(assignment["train"])
    val_set = set(assignment["val"])
    test_set = set(assignment["test"])
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Geographic split produced overlapping stems")

    imagesource_distribution = {
        split: _imagesource_counts(assignment[split], stem_meta) for split in assignment
    }

    payload: dict[str, Any] = {
        "seed": seed,
        "weight_by": weight_by,
        "fractions": fractions,
        "n_source_images": len(stem_meta),
        "n_train_stems": len(assignment["train"]),
        "n_val_stems": len(assignment["val"]),
        "n_test_stems": len(assignment["test"]),
        "imagesource_distribution": imagesource_distribution,
        "train_stems": assignment["train"],
        "val_stems": assignment["val"],
        "test_stems": assignment["test"],
    }
    write_yaml(payload, out_yaml)
    return payload


def _imagesource_counts(
    stems: list[str], stem_meta: dict[str, dict[str, Any]]
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for stem in stems:
        counts[str(stem_meta[stem]["imagesource"])] += 1
    return dict(sorted(counts.items()))


def materialize_split_filelists(
    splits_yaml: str | Path,
    images_dir: str | Path,
    out_dir: str | Path,
    image_suffix: str = ".png",
) -> dict[str, Path]:
    """Write per-split text files of absolute tile paths for Ultralytics.

    Tiles are emitted by ``prepare_dota`` under ``images/{split}/<stem>__x..._y...{suffix}``.
    The geographic split is defined over source stems, so we filter tiles whose
    filename starts with a stem in the split. Returns the mapping
    ``{"train": Path, "val": Path, "test": Path}``.
    """
    import yaml

    spec = yaml.safe_load(Path(splits_yaml).read_text())
    images_root = Path(images_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    split_to_stems = {
        "train": set(spec.get("train_stems", [])),
        "val": set(spec.get("val_stems", [])),
        "test": set(spec.get("test_stems", [])),
    }

    tile_paths: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    # All tiles live somewhere under images/{split}/; we ignore the on-disk split
    # subdir and re-bucket by source_stem.
    for tile_path in sorted(images_root.rglob(f"*{image_suffix}")):
        stem_part = tile_path.stem.split("__", 1)[0]
        for split, stems in split_to_stems.items():
            if stem_part in stems:
                tile_paths[split].append(str(tile_path.resolve()))
                break

    out_paths: dict[str, Path] = {}
    for split, paths in tile_paths.items():
        out_path = out_root / f"{split}.txt"
        out_path.write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")
        out_paths[split] = out_path
    return out_paths
