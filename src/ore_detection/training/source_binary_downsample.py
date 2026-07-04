"""Prepare downsampled source binary segmentation data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from ore_detection.training.source_dataset import SourceSegmentationSample, list_source_samples
from ore_detection.training.source_ore_downsample import (
    DEFAULT_DOWNSAMPLE_FACTOR,
    DEFAULT_SIZE_DIVISOR,
    calculate_rgb_mean_std_from_images,
    model_compatible_downsample_size,
)

DEFAULT_DATASETS = ("set_1", "set_2", "set_3")


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required to prepare downsampled binary masks.") from exc
    return torch


@dataclass(frozen=True)
class PreparedBinaryDownsampledSample:
    dataset: str
    split: str
    stem: str
    image_path: Path
    mask_path: Path
    width: int
    height: int


def _mask_image_to_tensor(mask: Image.Image):
    torch = _require_torch()
    mask = mask.convert("L")
    data = torch.tensor(list(mask.tobytes()), dtype=torch.uint8).view(mask.height, mask.width)
    return (data > 0).to(dtype=torch.uint8)


def save_downsampled_binary_sample(
    sample: SourceSegmentationSample,
    *,
    output_root: str | Path,
    factor: int = DEFAULT_DOWNSAMPLE_FACTOR,
    size_divisor: int = DEFAULT_SIZE_DIVISOR,
) -> PreparedBinaryDownsampledSample:
    """Downsample one binary image/mask pair and save it under ``output_root``."""
    torch = _require_torch()
    output_root = Path(output_root)
    target_size = model_compatible_downsample_size(
        sample.width,
        sample.height,
        factor=factor,
        size_divisor=size_divisor,
    )

    image_target = output_root / "images" / sample.dataset / sample.split / f"{sample.stem}.png"
    mask_target = output_root / "masks" / sample.dataset / sample.split / f"{sample.stem}.pt"
    image_target.parent.mkdir(parents=True, exist_ok=True)
    mask_target.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(sample.image_path) as opened:
        image = opened.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
        image.save(image_target)

    with Image.open(sample.mask_path) as opened:
        mask_image = opened.convert("L").resize(target_size, Image.Resampling.NEAREST)
        mask = _mask_image_to_tensor(mask_image)
    torch.save(mask, mask_target)

    return PreparedBinaryDownsampledSample(
        dataset=sample.dataset,
        split=sample.split,
        stem=sample.stem,
        image_path=image_target,
        mask_path=mask_target,
        width=target_size[0],
        height=target_size[1],
    )


def format_binary_train_stats_const(
    stats: dict[str, object],
    *,
    factor: int = DEFAULT_DOWNSAMPLE_FACTOR,
    size_divisor: int = DEFAULT_SIZE_DIVISOR,
) -> str:
    """Return Python source for generated binary training constants."""
    mean = tuple(float(value) for value in stats["mean"])  # type: ignore[index]
    std = tuple(float(value) for value in stats["std"])  # type: ignore[index]
    image_count = int(stats["image_count"])  # type: ignore[arg-type]
    pixel_count = int(stats["pixel_count"])  # type: ignore[arg-type]
    return (
        '"""Generated source binary segmentation training constants.\n\n'
        "This file is updated by scripts/prepare_source_binary_downsampled_dataset.py.\n"
        'Values are calculated from downsampled train images in the 0..1 tensor scale.\n'
        '"""\n\n'
        f"SOURCE_BINARY_TRAIN_RGB_MEAN = {mean!r}\n"
        f"SOURCE_BINARY_TRAIN_RGB_STD = {std!r}\n"
        f"SOURCE_BINARY_TRAIN_STATS_IMAGE_COUNT = {image_count}\n"
        f"SOURCE_BINARY_TRAIN_STATS_PIXEL_COUNT = {pixel_count}\n"
        f"SOURCE_BINARY_DOWNSAMPLE_FACTOR = {factor}\n"
        f"SOURCE_BINARY_DOWNSAMPLE_SIZE_DIVISOR = {size_divisor}\n"
    )


def write_binary_train_stats_const(
    stats: dict[str, object],
    *,
    target_path: str | Path,
    factor: int = DEFAULT_DOWNSAMPLE_FACTOR,
    size_divisor: int = DEFAULT_SIZE_DIVISOR,
) -> None:
    """Write train RGB statistics to a Python constants module."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        format_binary_train_stats_const(stats, factor=factor, size_divisor=size_divisor),
        encoding="utf-8",
    )


def prepare_downsampled_source_binary_dataset(
    *,
    datasets_root: str | Path = "datasets",
    binary_masks_root: str | Path = "data_work/binary_masks",
    output_root: str | Path = "data_work/source_binary_downsampled",
    datasets: Iterable[str] = DEFAULT_DATASETS,
    factor: int = DEFAULT_DOWNSAMPLE_FACTOR,
    size_divisor: int = DEFAULT_SIZE_DIVISOR,
    const_path: str | Path | None = "src/ore_detection/training/source_binary_const.py",
) -> dict[str, object]:
    """Prepare downsampled image/mask files and optional train statistics constants."""
    source_samples = list_source_samples(
        datasets_root=datasets_root,
        binary_masks_root=binary_masks_root,
        datasets=datasets,
    )
    prepared = [
        save_downsampled_binary_sample(
            sample,
            output_root=output_root,
            factor=factor,
            size_divisor=size_divisor,
        )
        for sample in source_samples
    ]
    train_images = [sample.image_path for sample in prepared if sample.split == "train"]
    stats = calculate_rgb_mean_std_from_images(train_images)
    if const_path is not None:
        write_binary_train_stats_const(stats, target_path=const_path, factor=factor, size_divisor=size_divisor)
    return {
        "sample_count": len(prepared),
        "train_count": sum(1 for sample in prepared if sample.split == "train"),
        "test_count": sum(1 for sample in prepared if sample.split == "test"),
        "stats": stats,
        "output_root": str(output_root),
    }
