"""Dataset inventory helpers for baseline crops and source mask datasets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from PIL import Image

from ore_detection.data.color_mask import unique_rgb_colors

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_SUFFIXES = {".png", ".bmp", ".tif", ".tiff"}


def _image_size(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    return width, height, mode


def _iter_files(root: Path, suffixes: set[str]) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def inventory_baseline_images(root: str | Path) -> list[dict[str, object]]:
    """Inventory baseline weak-label crops.

    Expected layout: ``baseline/Part N/<label>/<image>``.
    """
    root = Path(root)
    records: list[dict[str, object]] = []
    for image_path in _iter_files(root, IMAGE_SUFFIXES):
        rel = image_path.relative_to(root)
        if "panoramas" in {part.lower() for part in rel.parts}:
            continue
        if len(rel.parts) < 3:
            continue
        part, label = rel.parts[0], rel.parts[1]
        width, height, mode = _image_size(image_path)
        records.append(
            {
                "dataset": "baseline",
                "path": str(image_path),
                "part": part,
                "label": label,
                "split": "weak_label",
                "width": width,
                "height": height,
                "mode": mode,
                "file_size_bytes": image_path.stat().st_size,
                "magnification": "10x",
            }
        )
    return records


def _find_image_for_mask(dataset_root: Path, split: str, stem: str) -> Path | None:
    image_dir = dataset_root / "imgs" / split
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
        candidate_upper = image_dir / f"{stem}{suffix.upper()}"
        if candidate_upper.exists():
            return candidate_upper
    matches = list(image_dir.glob(f"{stem}.*")) if image_dir.exists() else []
    return matches[0] if matches else None


def inventory_source_dataset(root: str | Path, *, dataset_name: str) -> list[dict[str, object]]:
    """Inventory one source segmentation dataset.

    Expected layout: ``set_N/imgs/<split>/<stem>.*`` and
    ``set_N/masks_colored/<split>/<stem>.png``. The ``masks`` and
    ``masks_human`` folders are intentionally ignored.
    """
    root = Path(root)
    records: list[dict[str, object]] = []
    masks_root = root / "masks_colored"
    for mask_path in _iter_files(masks_root, MASK_SUFFIXES):
        rel = mask_path.relative_to(masks_root)
        if len(rel.parts) < 2:
            continue
        split = rel.parts[0]
        image_path = _find_image_for_mask(root, split, mask_path.stem)
        if image_path is None:
            continue
        width, height, mode = _image_size(image_path)
        mask_width, mask_height, mask_mode = _image_size(mask_path)
        with Image.open(mask_path) as mask_image:
            colors = unique_rgb_colors(mask_image)
        records.append(
            {
                "dataset": dataset_name,
                "path": str(image_path),
                "mask_colored_path": str(mask_path),
                "split": split,
                "stem": mask_path.stem,
                "width": width,
                "height": height,
                "mode": mode,
                "mask_width": mask_width,
                "mask_height": mask_height,
                "mask_mode": mask_mode,
                "unique_mask_colors_count": len(colors),
                "unique_mask_colors": ";".join(
                    f"{r},{g},{b}:{count}" for (r, g, b), count in sorted(colors.items())
                ),
                "file_size_bytes": image_path.stat().st_size,
                "mask_file_size_bytes": mask_path.stat().st_size,
                "magnification": "50x",
                "shape_match": width == mask_width and height == mask_height,
            }
        )
    return records


def write_inventory_csv(records: list[dict[str, object]], path: str | Path) -> None:
    """Write inventory records to CSV with stable field ordering."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
