#!/usr/bin/env python
"""Generate a draft talc candidate mask for human correction.

This script is an active-learning helper, not a trained model. It detects dark
regions in non-ore matrix and writes a binary mask plus a blue overlay preview.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from ore_detection.talc.candidates import detect_dark_matrix_candidates


def _load_grayscale(path: Path) -> list[list[int]]:
    image = Image.open(path).convert("L")
    width, height = image.size
    pixels = list(image.tobytes())
    return [pixels[row * width : (row + 1) * width] for row in range(height)]


def _load_binary_mask(path: Path) -> list[list[int]]:
    image = Image.open(path).convert("L")
    width, height = image.size
    pixels = [1 if value > 0 else 0 for value in image.tobytes()]
    return [pixels[row * width : (row + 1) * width] for row in range(height)]


def _save_binary_mask(mask: list[list[int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height = len(mask)
    width = len(mask[0]) if height else 0
    image = Image.new("L", (width, height))
    image.putdata([255 if value else 0 for row in mask for value in row])
    image.save(path)


def _save_overlay(image_path: Path, mask: list[list[int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    overlay.putdata([(0, 80, 255, 110) if value else (0, 0, 0, 0) for row in mask for value in row])
    Image.alpha_composite(image, overlay).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Input microscopy image")
    parser.add_argument("--ore-mask", help="Optional binary ore mask; non-zero pixels are excluded")
    parser.add_argument("--output-mask", required=True, help="Output binary talc candidate PNG")
    parser.add_argument("--output-overlay", help="Optional blue overlay preview PNG")
    parser.add_argument("--dark-offset", type=float, default=25.0, help="Candidate threshold = mean matrix intensity - offset")
    parser.add_argument("--min-component-area", type=int, default=12, help="Remove smaller dark components")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    ore_mask = _load_binary_mask(Path(args.ore_mask)) if args.ore_mask else None
    candidates = detect_dark_matrix_candidates(
        _load_grayscale(image_path),
        ore_mask=ore_mask,
        dark_offset=args.dark_offset,
        min_component_area=args.min_component_area,
    )
    _save_binary_mask(candidates, Path(args.output_mask))
    if args.output_overlay:
        _save_overlay(image_path, candidates, Path(args.output_overlay))
    candidate_pixels = sum(sum(row) for row in candidates)
    total_pixels = sum(len(row) for row in candidates)
    fraction = candidate_pixels / total_pixels if total_pixels else 0.0
    print(f"candidate_pixels={candidate_pixels} total_pixels={total_pixels} fraction={fraction:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
