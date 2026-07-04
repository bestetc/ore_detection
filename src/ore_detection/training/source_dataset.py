"""Source segmentation dataset indexing.

The first segmentation stage trains from external source datasets. This module
pairs ``set_N/imgs/<split>`` OM images with binary masks generated from
``set_N/masks_colored/<split>``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class SourceSegmentationSample:
    dataset: str
    split: str
    stem: str
    image_path: Path
    mask_path: Path
    width: int
    height: int


def _find_image(image_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        for candidate in (image_dir / f"{stem}{suffix}", image_dir / f"{stem}{suffix.upper()}"):
            if candidate.exists():
                return candidate
    matches = sorted(image_dir.glob(f"{stem}.*")) if image_dir.exists() else []
    return matches[0] if matches else None


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def list_source_samples(
    *,
    datasets_root: str | Path = "datasets",
    binary_masks_root: str | Path = "data_work/binary_masks",
    datasets: Iterable[str] = ("set_1", "set_2", "set_3"),
    split: str | None = None,
) -> list[SourceSegmentationSample]:
    """List source segmentation samples paired with generated binary masks."""
    datasets_root = Path(datasets_root)
    binary_masks_root = Path(binary_masks_root)
    samples: list[SourceSegmentationSample] = []

    for dataset in datasets:
        mask_root = binary_masks_root / dataset
        if not mask_root.exists():
            continue
        for mask_path in sorted(mask_root.rglob("*.png")):
            rel = mask_path.relative_to(mask_root)
            if len(rel.parts) < 2:
                continue
            sample_split = rel.parts[0]
            if split is not None and sample_split != split:
                continue
            image_path = _find_image(datasets_root / dataset / "imgs" / sample_split, mask_path.stem)
            if image_path is None:
                continue
            width, height = _image_size(image_path)
            mask_width, mask_height = _image_size(mask_path)
            if (width, height) != (mask_width, mask_height):
                raise ValueError(f"image/mask shape mismatch for {image_path} and {mask_path}")
            samples.append(
                SourceSegmentationSample(
                    dataset=dataset,
                    split=sample_split,
                    stem=mask_path.stem,
                    image_path=image_path,
                    mask_path=mask_path,
                    width=width,
                    height=height,
                )
            )
    return samples


def load_sample_images(sample: SourceSegmentationSample) -> tuple[Image.Image, Image.Image]:
    """Load a sample as RGB image and single-channel 0/1 label mask."""
    image = Image.open(sample.image_path).convert("RGB")
    mask = Image.open(sample.mask_path).convert("L")
    if image.size != mask.size:
        raise ValueError(f"image/mask size mismatch for {sample.image_path} and {sample.mask_path}")
    return image, mask
