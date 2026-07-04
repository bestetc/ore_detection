"""Prepare downsampled source ore segmentation data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from ore_detection.training.source_ore_dataset import SourceOreSegmentationSample, list_source_ore_samples

DEFAULT_DOWNSAMPLE_FACTOR = 4
DEFAULT_SIZE_DIVISOR = 4
DEFAULT_DATASETS = ("set_1", "set_2", "set_3")


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required to prepare downsampled ore masks.") from exc
    return torch


@dataclass(frozen=True)
class PreparedDownsampledSample:
    dataset: str
    split: str
    stem: str
    image_path: Path
    mask_path: Path
    width: int
    height: int


def model_compatible_downsample_size(
    width: int,
    height: int,
    *,
    factor: int = DEFAULT_DOWNSAMPLE_FACTOR,
    size_divisor: int = DEFAULT_SIZE_DIVISOR,
) -> tuple[int, int]:
    """Return a downsampled size whose dimensions are divisible by ``size_divisor``."""
    if factor < 1:
        raise ValueError(f"factor must be positive, got {factor}")
    if size_divisor < 1:
        raise ValueError(f"size_divisor must be positive, got {size_divisor}")

    target_width = max(size_divisor, width // factor)
    target_height = max(size_divisor, height // factor)
    target_width = max(size_divisor, (target_width // size_divisor) * size_divisor)
    target_height = max(size_divisor, (target_height // size_divisor) * size_divisor)
    return target_width, target_height


def load_one_hot_mask(path: str | Path):
    """Load a saved one-hot mask tensor from disk."""
    torch = _require_torch()
    try:
        mask = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        mask = torch.load(path, map_location="cpu")
    if mask.ndim != 3:
        raise ValueError(f"mask tensor must be shaped [C,H,W], got {tuple(mask.shape)} for {path}")
    return mask


def one_hot_to_class_index(mask):
    """Convert a one-hot mask ``[C,H,W]`` to a compact uint8 class-index mask."""
    torch = _require_torch()
    if mask.ndim != 3:
        raise ValueError(f"mask tensor must be shaped [C,H,W], got {tuple(mask.shape)}")
    class_index = mask.argmax(dim=0)
    if int(class_index.max().item()) > 255:
        raise ValueError("class-index masks require class ids <= 255")
    return class_index.to(dtype=torch.uint8)


def resize_class_index_mask(mask, size: tuple[int, int]):
    """Resize a class-index mask with nearest-neighbor interpolation."""
    torch = _require_torch()
    if mask.ndim != 2:
        raise ValueError(f"class-index mask must be shaped [H,W], got {tuple(mask.shape)}")
    resized = torch.nn.functional.interpolate(
        mask.float().unsqueeze(0).unsqueeze(0),
        size=(size[1], size[0]),
        mode="nearest",
    )
    return resized.squeeze(0).squeeze(0).to(dtype=torch.uint8)


def save_downsampled_sample(
    sample: SourceOreSegmentationSample,
    *,
    output_root: str | Path,
    factor: int = DEFAULT_DOWNSAMPLE_FACTOR,
    size_divisor: int = DEFAULT_SIZE_DIVISOR,
) -> PreparedDownsampledSample:
    """Downsample one image/mask pair and save it under ``output_root``."""
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
        image = opened.convert("RGB")
        image = image.resize(target_size, Image.Resampling.BILINEAR)
        image.save(image_target)

    one_hot = load_one_hot_mask(sample.mask_path)
    if sample.width != int(one_hot.shape[2]) or sample.height != int(one_hot.shape[1]):
        raise ValueError(f"image/mask size mismatch for {sample.image_path} and {sample.mask_path}")
    class_index = resize_class_index_mask(one_hot_to_class_index(one_hot), target_size)
    torch.save(class_index, mask_target)

    return PreparedDownsampledSample(
        dataset=sample.dataset,
        split=sample.split,
        stem=sample.stem,
        image_path=image_target,
        mask_path=mask_target,
        width=target_size[0],
        height=target_size[1],
    )


def calculate_rgb_mean_std_from_images(image_paths: Iterable[str | Path]) -> dict[str, object]:
    """Calculate RGB channel mean/std from image files in the 0..1 scale."""
    channel_sum = [0.0, 0.0, 0.0]
    channel_sum_sq = [0.0, 0.0, 0.0]
    pixel_count = 0
    image_count = 0

    for image_path in image_paths:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            histogram = image.histogram()
            pixels = image.width * image.height
        values = range(256)
        for channel_index in range(3):
            counts = histogram[channel_index * 256 : (channel_index + 1) * 256]
            channel_sum[channel_index] += sum((value / 255.0) * count for value, count in zip(values, counts))
            channel_sum_sq[channel_index] += sum(
                ((value / 255.0) ** 2) * count for value, count in zip(values, counts)
            )
        pixel_count += pixels
        image_count += 1

    if pixel_count == 0:
        raise ValueError("cannot calculate RGB stats from an empty image set")

    mean = tuple(value / pixel_count for value in channel_sum)
    std = []
    for channel_index in range(3):
        variance = (channel_sum_sq[channel_index] / pixel_count) - (mean[channel_index] * mean[channel_index])
        std.append(max(variance, 0.0) ** 0.5)
    return {
        "mean": mean,
        "std": tuple(value if value > 0 else 1.0 for value in std),
        "pixel_count": pixel_count,
        "image_count": image_count,
    }


def format_train_stats_const(
    stats: dict[str, object],
    *,
    factor: int = DEFAULT_DOWNSAMPLE_FACTOR,
    size_divisor: int = DEFAULT_SIZE_DIVISOR,
) -> str:
    """Return Python source for the generated source-ore training constants."""
    mean = tuple(float(value) for value in stats["mean"])  # type: ignore[index]
    std = tuple(float(value) for value in stats["std"])  # type: ignore[index]
    image_count = int(stats["image_count"])  # type: ignore[arg-type]
    pixel_count = int(stats["pixel_count"])  # type: ignore[arg-type]
    return (
        '"""Generated source ore segmentation training constants.\n\n'
        "This file is updated by scripts/prepare_source_ore_downsampled_dataset.py.\n"
        'Values are calculated from downsampled train images in the 0..1 tensor scale.\n'
        '"""\n\n'
        f"SOURCE_ORE_TRAIN_RGB_MEAN = {mean!r}\n"
        f"SOURCE_ORE_TRAIN_RGB_STD = {std!r}\n"
        f"SOURCE_ORE_TRAIN_STATS_IMAGE_COUNT = {image_count}\n"
        f"SOURCE_ORE_TRAIN_STATS_PIXEL_COUNT = {pixel_count}\n"
        f"SOURCE_ORE_DOWNSAMPLE_FACTOR = {factor}\n"
        f"SOURCE_ORE_DOWNSAMPLE_SIZE_DIVISOR = {size_divisor}\n"
    )


def write_train_stats_const(
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
        format_train_stats_const(stats, factor=factor, size_divisor=size_divisor),
        encoding="utf-8",
    )


def prepare_downsampled_source_ore_dataset(
    *,
    datasets_root: str | Path = "datasets",
    masks_root: str | Path = "data_work/source_ore_type_masks",
    output_root: str | Path = "data_work/source_ore_downsampled",
    datasets: Iterable[str] = DEFAULT_DATASETS,
    factor: int = DEFAULT_DOWNSAMPLE_FACTOR,
    size_divisor: int = DEFAULT_SIZE_DIVISOR,
    const_path: str | Path | None = "src/ore_detection/training/const.py",
) -> dict[str, object]:
    """Prepare downsampled image/mask files and optional train statistics constants."""
    source_samples = list_source_ore_samples(
        datasets_root=datasets_root,
        masks_root=masks_root,
        datasets=datasets,
    )
    prepared = [
        save_downsampled_sample(
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
        write_train_stats_const(stats, target_path=const_path, factor=factor, size_divisor=size_divisor)
    return {
        "sample_count": len(prepared),
        "train_count": sum(1 for sample in prepared if sample.split == "train"),
        "test_count": sum(1 for sample in prepared if sample.split == "test"),
        "stats": stats,
        "output_root": str(output_root),
    }
