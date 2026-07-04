"""Mask overlay utilities for visual QA."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

DEFAULT_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (0, 80, 255),
    255: (255, 255, 255),
}


def colorize_mask(
    mask: Image.Image,
    *,
    palette: dict[int, tuple[int, int, int]] | None = None,
    alpha: int = 96,
) -> Image.Image:
    """Convert a label mask to an RGBA overlay.

    Class 0 is transparent by default because it is background in the coarse
    taxonomy.
    """
    palette = palette or DEFAULT_PALETTE
    mask_l = mask.convert("L")
    raw = mask_l.tobytes()
    pixels: list[tuple[int, int, int, int]] = []
    for value in raw:
        if value == 0:
            pixels.append((0, 0, 0, 0))
            continue
        rgb = palette.get(value, (255, 255, 255))
        pixels.append((*rgb, alpha))
    overlay = Image.new("RGBA", mask_l.size)
    overlay.putdata(pixels)
    return overlay


def overlay_mask_on_image(
    image: Image.Image,
    mask: Image.Image,
    *,
    palette: dict[int, tuple[int, int, int]] | None = None,
    alpha: int = 96,
) -> Image.Image:
    """Alpha-composite a colorized mask over an image."""
    base = image.convert("RGBA")
    overlay = colorize_mask(mask, palette=palette, alpha=alpha)
    if base.size != overlay.size:
        raise ValueError("image and mask sizes must match")
    return Image.alpha_composite(base, overlay)


def save_overlay(image: Image.Image, mask: Image.Image, output_path: str | Path, *, alpha: int = 96) -> None:
    """Save an image/mask QA overlay."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_mask_on_image(image, mask, alpha=alpha).save(output_path)
