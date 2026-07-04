#!/usr/bin/env python
"""Convert set_N color-coded ore masks to binary ore/background masks."""

from __future__ import annotations

import argparse
from pathlib import Path

from ore_detection.data.color_mask import convert_color_mask_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["set_1", "set_2", "set_3"])
    parser.add_argument("--source-root", default="datasets")
    parser.add_argument("--output-root", default="data_work/binary_masks")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.source_root) / args.dataset / "masks_colored"
    output_root = Path(args.output_root) / args.dataset
    count = 0
    ore_pixels = 0
    total_pixels = 0
    for source_path in sorted(dataset_root.rglob("*.png")):
        relative = source_path.relative_to(dataset_root)
        target_path = output_root / relative
        stats = convert_color_mask_file(source_path, target_path)
        count += 1
        ore_pixels += stats["ore_pixels"]
        total_pixels += stats["total_pixels"]
    fraction = ore_pixels / total_pixels if total_pixels else 0.0
    print(
        f"converted={count} dataset={args.dataset} source=masks_colored "
        f"output={output_root} ore_fraction={fraction:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
