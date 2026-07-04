"""Mask IO and conversion utilities."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ore_detection.data.label_mapping import TargetTaxonomy, map_value


def convert_source_mask_file(
    source_path: str | Path,
    target_path: str | Path,
    *,
    source_dataset: str,
    target_taxonomy: TargetTaxonomy,
) -> None:
    """Convert an RGB/grayscale source mask to a single-channel mapped PNG."""
    source_path = Path(source_path)
    target_path = Path(target_path)
    image = Image.open(source_path).convert("L")
    translation = bytes(
        map_value(value, source_dataset=source_dataset, target_taxonomy=target_taxonomy)
        for value in range(256)
    )
    target = Image.frombytes("L", image.size, image.tobytes().translate(translation))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target.save(target_path)
