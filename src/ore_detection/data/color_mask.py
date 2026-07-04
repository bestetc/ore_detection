"""Utilities for converting color-coded ore masks to binary masks."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image

RgbColor = Tuple[int, int, int]
DEFAULT_BACKGROUND_COLORS: frozenset[RgbColor] = frozenset({(0, 0, 0)})


def _as_rgb(color: Iterable[int]) -> RgbColor:
    values = tuple(int(v) for v in color)
    if len(values) != 3:
        raise ValueError(f"RGB color must have exactly 3 channels, got {values}")
    return values  # type: ignore[return-value]


def unique_rgb_colors(image: Image.Image, *, max_colors: int = 1_000_000) -> dict[RgbColor, int]:
    """Return RGB color counts for a color-coded mask."""
    rgb = image.convert("RGB")
    colors = rgb.getcolors(maxcolors=max_colors)
    if colors is None:
        pixels = rgb.get_flattened_data() if hasattr(rgb, "get_flattened_data") else rgb.getdata()
        return dict(Counter(_as_rgb(pixel) for pixel in pixels))
    return {_as_rgb(color): count for count, color in colors}


def color_mask_to_binary(
    image: Image.Image,
    *,
    background_colors: Iterable[RgbColor] = DEFAULT_BACKGROUND_COLORS,
) -> Image.Image:
    """Convert a color-coded mask to a binary ore/background mask.

    Background colors become 0. Every other visible color becomes 1. For RGBA
    masks, fully transparent pixels are background even if their RGB channels are
    non-black.
    """
    background = {tuple(color) for color in background_colors}
    rgba = image.convert("RGBA")
    binary_values = []
    pixels = rgba.get_flattened_data() if hasattr(rgba, "get_flattened_data") else rgba.getdata()
    for r, g, b, a in pixels:
        if a == 0 or (r, g, b) in background:
            binary_values.append(0)
        else:
            binary_values.append(1)
    binary = Image.new("L", rgba.size)
    binary.putdata(binary_values)
    return binary


def convert_color_mask_file(
    source_path: str | Path,
    target_path: str | Path,
    *,
    background_colors: Iterable[RgbColor] = DEFAULT_BACKGROUND_COLORS,
) -> dict[str, int]:
    """Convert one color-coded mask file and write a 0/1 PNG binary mask."""
    source_path = Path(source_path)
    target_path = Path(target_path)
    with Image.open(source_path) as image:
        binary = color_mask_to_binary(image, background_colors=background_colors)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    binary.save(target_path)
    hist = binary.histogram()
    return {
        "background_pixels": hist[0],
        "ore_pixels": hist[1],
        "total_pixels": binary.width * binary.height,
    }
