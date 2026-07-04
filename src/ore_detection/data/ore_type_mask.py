"""Conversion utilities for multiclass source ore segmentation masks."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from ore_detection.data.color_mask import unique_rgb_colors
from ore_detection.data.ore_type_legend import OreTypeLegend, RgbColor, format_rgb


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required to save source ore-type tensors. Install the `ml` optional dependencies.") from exc
    return torch


def _iter_rgba_pixels(image: Image.Image):
    rgba = image.convert("RGBA")
    pixels = rgba.get_flattened_data() if hasattr(rgba, "get_flattened_data") else rgba.getdata()
    return rgba, pixels


def unknown_colors(image: Image.Image, *, dataset: str, legend: OreTypeLegend) -> dict[RgbColor, int]:
    """Return RGB colors in a mask that are absent from the audited legend."""
    known = set(legend.color_to_entry(dataset))
    unknown: dict[RgbColor, int] = {}
    for rgb, count in unique_rgb_colors(image).items():
        if rgb not in known:
            unknown[rgb] = count
    return unknown


def color_mask_to_class_image(image: Image.Image, *, dataset: str, legend: OreTypeLegend) -> Image.Image:
    """Convert an RGB color-coded mask into a single-channel class-index mask."""
    fast_image = _color_mask_to_class_image_numpy(image, dataset=dataset, legend=legend)
    if fast_image is not None:
        return fast_image

    color_to_index = legend.color_to_index(dataset)
    rgba, pixels = _iter_rgba_pixels(image)
    background_index = legend.background_index
    values: list[int] = []
    missing: Counter[RgbColor] = Counter()

    for r, g, b, alpha in pixels:
        if alpha == 0:
            values.append(background_index)
            continue
        rgb = (int(r), int(g), int(b))
        class_index = color_to_index.get(rgb)
        if class_index is None:
            missing[rgb] += 1
            values.append(background_index)
            continue
        values.append(class_index)

    if missing:
        details = ", ".join(f"{format_rgb(rgb)}={count}" for rgb, count in missing.most_common())
        raise ValueError(f"{dataset} mask has colors missing from source ore-type legend: {details}")

    class_image = Image.new("L", rgba.size)
    class_image.putdata(values)
    return class_image


def _color_mask_to_class_image_numpy(
    image: Image.Image,
    *,
    dataset: str,
    legend: OreTypeLegend,
) -> Image.Image | None:
    try:
        import numpy as np
    except ModuleNotFoundError:
        return None

    rgba = np.asarray(image.convert("RGBA"))
    class_indices = np.full(rgba.shape[:2], legend.background_index, dtype=np.uint8)
    known = rgba[:, :, 3] == 0

    for rgb, class_index in legend.color_to_index(dataset).items():
        match = (
            (rgba[:, :, 0] == rgb[0])
            & (rgba[:, :, 1] == rgb[1])
            & (rgba[:, :, 2] == rgb[2])
            & (rgba[:, :, 3] != 0)
        )
        class_indices[match] = class_index
        known |= match

    missing_mask = ~known
    if bool(missing_mask.any()):
        missing_pixels = rgba[:, :, :3][missing_mask]
        colors, counts = np.unique(missing_pixels, axis=0, return_counts=True)
        details = ", ".join(
            f"{format_rgb((int(color[0]), int(color[1]), int(color[2])))}={int(count)}"
            for color, count in zip(colors, counts)
        )
        raise ValueError(f"{dataset} mask has colors missing from source ore-type legend: {details}")

    return Image.fromarray(class_indices, mode="L")


def class_image_to_one_hot_tensor(class_image: Image.Image, *, class_count: int):
    """Convert a class-index PIL image into a uint8 torch tensor shaped [C,H,W]."""
    torch = _require_torch()
    image = class_image.convert("L")
    data = list(image.tobytes())
    class_indices = torch.tensor(data, dtype=torch.long).view(image.height, image.width)
    if class_indices.numel() and int(class_indices.max()) >= class_count:
        raise ValueError(f"class image contains label >= class_count ({class_count})")
    one_hot = torch.nn.functional.one_hot(class_indices, num_classes=class_count)
    return one_hot.permute(2, 0, 1).contiguous().to(torch.uint8)


def color_mask_to_one_hot_tensor(image: Image.Image, *, dataset: str, legend: OreTypeLegend):
    """Convert a color-coded mask into a uint8 one-hot tensor shaped [C,H,W]."""
    class_image = color_mask_to_class_image(image, dataset=dataset, legend=legend)
    return class_image_to_one_hot_tensor(class_image, class_count=legend.class_count)


def one_hot_to_binary_ore(one_hot: Any, *, legend: OreTypeLegend):
    """Derive a binary ore mask by summing all non-background channels.

    Accepts either [C,H,W] or [B,C,H,W] tensors. The output keeps a singleton
    mask channel: [1,H,W] or [B,1,H,W].
    """
    if one_hot.ndim == 3:
        binary = one_hot[list(legend.non_background_indices)].sum(dim=0, keepdim=True)
    elif one_hot.ndim == 4:
        binary = one_hot[:, list(legend.non_background_indices)].sum(dim=1, keepdim=True)
    else:
        raise ValueError(f"expected [C,H,W] or [B,C,H,W] one-hot tensor, got ndim={one_hot.ndim}")
    return binary.clamp(max=1)


def validate_one_hot_tensor(one_hot: Any, *, legend: OreTypeLegend) -> None:
    """Validate one-hot tensor shape and per-pixel exclusivity."""
    if one_hot.ndim != 3:
        raise ValueError(f"saved mask tensor must be shaped [C,H,W], got ndim={one_hot.ndim}")
    if int(one_hot.shape[0]) != legend.class_count:
        raise ValueError(f"saved mask has {one_hot.shape[0]} channels, expected {legend.class_count}")
    per_pixel_sum = one_hot.sum(dim=0)
    if bool((per_pixel_sum != 1).any()):
        raise ValueError("saved mask is not one-hot: each pixel must sum to exactly 1")


def convert_color_mask_file(
    source_path: str | Path,
    target_path: str | Path,
    *,
    dataset: str,
    legend: OreTypeLegend,
) -> dict[str, Any]:
    """Convert one color mask file to a saved one-hot `.pt` tensor."""
    torch = _require_torch()
    source_path = Path(source_path)
    target_path = Path(target_path)
    with Image.open(source_path) as image:
        one_hot = color_mask_to_one_hot_tensor(image, dataset=dataset, legend=legend)

    validate_one_hot_tensor(one_hot, legend=legend)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(one_hot, target_path)
    class_pixels = one_hot.view(legend.class_count, -1).sum(dim=1).tolist()
    return {
        "source_path": str(source_path),
        "target_path": str(target_path),
        "width": int(one_hot.shape[2]),
        "height": int(one_hot.shape[1]),
        "class_pixels": {name: int(class_pixels[index]) for index, name in enumerate(legend.class_names)},
    }


def audit_color_mask_file(source_path: str | Path, *, dataset: str, legend: OreTypeLegend) -> dict[str, Any]:
    """Return a small conversion audit without writing any artifacts."""
    source_path = Path(source_path)
    with Image.open(source_path) as image:
        colors = unique_rgb_colors(image)
        missing = unknown_colors(image, dataset=dataset, legend=legend)
    return {
        "source_path": str(source_path),
        "known_color_count": len(colors) - len(missing),
        "unknown_color_count": len(missing),
        "unknown_colors": {format_rgb(rgb): count for rgb, count in sorted(missing.items())},
    }
