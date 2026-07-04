"""Intergrowth descriptors derived from predicted ore segmentation masks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image

from ore_detection.descriptors.contacts import contact_lengths
from ore_detection.descriptors.morphology import component_stats, summarize_components

MaskRows = list[list[int]]


def _rows_from_image(mask: Image.Image, *, foreground_binary: bool) -> MaskRows:
    image = mask.convert("L")
    raw = list(image.tobytes())
    rows: MaskRows = []
    for row in range(image.height):
        start = row * image.width
        values = raw[start : start + image.width]
        if foreground_binary:
            rows.append([1 if int(value) > 0 else 0 for value in values])
        else:
            rows.append([int(value) for value in values])
    return rows


def _rows_from_sequence(mask: Sequence[Sequence[Any]], *, foreground_binary: bool) -> MaskRows:
    rows: MaskRows = []
    width = len(mask[0]) if mask else 0
    for row_values in mask:
        if len(row_values) != width:
            raise ValueError("mask rows must all have the same width")
        if foreground_binary:
            rows.append([1 if int(value) > 0 else 0 for value in row_values])
        else:
            rows.append([int(value) for value in row_values])
    return rows


def binary_mask_rows(mask: Image.Image | Sequence[Sequence[Any]]) -> MaskRows:
    """Return a 0/1 row mask where any non-zero pixel is ore."""
    if isinstance(mask, Image.Image):
        return _rows_from_image(mask, foreground_binary=True)
    return _rows_from_sequence(mask, foreground_binary=True)


def class_mask_rows(mask: Image.Image | Sequence[Sequence[Any]]) -> MaskRows:
    """Return class-index rows from a PIL or nested sequence mask."""
    if isinstance(mask, Image.Image):
        return _rows_from_image(mask, foreground_binary=False)
    return _rows_from_sequence(mask, foreground_binary=False)


def _height_width(rows: MaskRows) -> tuple[int, int]:
    height = len(rows)
    width = len(rows[0]) if height else 0
    for row in rows:
        if len(row) != width:
            raise ValueError("mask rows must all have the same width")
    return height, width


def _quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = fraction * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _component_area_distribution(binary_rows: MaskRows) -> dict[str, float]:
    stats = component_stats(binary_rows, foreground_values={1})
    areas = sorted(float(item["area"]) for item in stats)
    if not areas:
        return {
            "component_area_min": 0.0,
            "component_area_p50": 0.0,
            "component_area_p90": 0.0,
            "component_area_max": 0.0,
            "component_area_mean": 0.0,
        }
    return {
        "component_area_min": areas[0],
        "component_area_p50": _quantile(areas, 0.50),
        "component_area_p90": _quantile(areas, 0.90),
        "component_area_max": areas[-1],
        "component_area_mean": sum(areas) / len(areas),
    }


def _class_name(class_index: int, class_names: Sequence[str]) -> str:
    if 0 <= class_index < len(class_names):
        return str(class_names[class_index])
    return f"class_{class_index}"


def _class_counts(rows: MaskRows) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in rows:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    return counts


def summarize_intergrowth_prediction(
    binary_mask: Image.Image | Sequence[Sequence[Any]],
    *,
    multiclass_mask: Image.Image | Sequence[Sequence[Any]] | None = None,
    class_names: Sequence[str] = (),
    background_index: int = 0,
    small_area_threshold: int = 25,
) -> dict[str, float]:
    """Aggregate geometry and mineral-aware descriptors from prediction masks.

    Talc is intentionally excluded. Only reviewed UI talc masks should supply
    talc descriptors later.
    """
    binary_rows = binary_mask_rows(binary_mask)
    height, width = _height_width(binary_rows)
    pixel_count = height * width
    ore_area = sum(sum(row) for row in binary_rows)
    morphology = summarize_components(
        binary_rows,
        foreground_values={1},
        small_area_threshold=small_area_threshold,
    )
    component_perimeter = sum(item["perimeter"] for item in component_stats(binary_rows, foreground_values={1}))
    ore_background_contact = contact_lengths(binary_rows, classes={0, 1}).get((0, 1), 0)

    descriptors: dict[str, float] = {
        "pixel_count": float(pixel_count),
        "ore_area": float(ore_area),
        "ore_area_fraction": float(ore_area / pixel_count) if pixel_count else 0.0,
        "ore_background_contact_length": float(ore_background_contact),
        "ore_background_contact_density": float(ore_background_contact / pixel_count) if pixel_count else 0.0,
        "ore_perimeter": float(component_perimeter),
        "ore_perimeter_density": float(component_perimeter / pixel_count) if pixel_count else 0.0,
    }
    descriptors.update({f"ore_{key}": float(value) for key, value in morphology.items()})
    descriptors.update(_component_area_distribution(binary_rows))

    if multiclass_mask is None:
        return descriptors

    class_rows = class_mask_rows(multiclass_mask)
    class_height, class_width = _height_width(class_rows)
    if (class_height, class_width) != (height, width):
        raise ValueError("binary_mask and multiclass_mask must have the same shape")

    class_counts = _class_counts(class_rows)
    for class_index, count in sorted(class_counts.items()):
        name = _class_name(class_index, class_names)
        descriptors[f"class_area_{name}"] = float(count)
        descriptors[f"class_fraction_{name}"] = float(count / pixel_count) if pixel_count else 0.0

    mineral_classes = {class_index for class_index in class_counts if class_index != background_index}
    for (left, right), length in sorted(contact_lengths(class_rows, classes=mineral_classes).items()):
        left_name = _class_name(int(left), class_names)
        right_name = _class_name(int(right), class_names)
        descriptors[f"mineral_contact_{left_name}__{right_name}"] = float(length)

    return descriptors


def summarize_prediction_artifacts(sample_dir: str | Path, *, small_area_threshold: int = 25) -> dict[str, Any]:
    """Build one descriptor row from a saved trained-model prediction folder."""
    sample_dir = Path(sample_dir)
    metadata_path = sample_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    ore_checkpoint = metadata.get("ore_checkpoint") or {}
    class_names = tuple(str(name) for name in ore_checkpoint.get("class_names", ()))
    background_index = int(ore_checkpoint.get("background_index", 0))

    with Image.open(sample_dir / "ore_mask.png") as binary_image:
        multiclass_path = sample_dir / "ore_multiclass_mask.png"
        if multiclass_path.exists():
            with Image.open(multiclass_path) as class_image:
                descriptors = summarize_intergrowth_prediction(
                    binary_image,
                    multiclass_mask=class_image,
                    class_names=class_names,
                    background_index=background_index,
                    small_area_threshold=small_area_threshold,
                )
        else:
            descriptors = summarize_intergrowth_prediction(
                binary_image,
                small_area_threshold=small_area_threshold,
            )

    row: dict[str, Any] = {
        "sample_id": metadata.get("sample_id", sample_dir.name),
        "image_path": metadata.get("image_path", ""),
        "prediction_dir": str(sample_dir),
    }
    row.update(descriptors)
    return row


def write_descriptor_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write descriptor rows to CSV with stable unioned columns."""
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
