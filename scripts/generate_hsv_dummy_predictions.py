#!/usr/bin/env python
"""Generate HSV dummy prediction artifacts for a batch of images."""

from __future__ import annotations

import argparse
from pathlib import Path

from ore_detection.backend.service import BackendConfig, list_ui_images
from ore_detection.inference.prediction_store import save_hsv_dummy_prediction
from ore_detection.segmentation.hsv_dummy import HsvDummyConfig
from ore_detection.talc.hsv_candidates import calculate_rgb_mean_std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--output-root", default="data_work/predictions/hsv_dummy")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--value-threshold", type=int, default=90)
    parser.add_argument("--foreground", choices=["bright", "dark"], default="bright")
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--standardize-max-image-size", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = BackendConfig(
        project_root=Path.cwd(),
        datasets_root=Path(args.datasets_root),
        predictions_root=Path(args.output_root),
    ).resolve()
    images = list_ui_images(config.datasets_root, limit=args.limit)
    stats = None
    if args.standardize and images:
        stats = calculate_rgb_mean_std(images, max_image_size=args.standardize_max_image_size)
    hsv_config = HsvDummyConfig(
        value_threshold=args.value_threshold,
        foreground=args.foreground,
        standardize=args.standardize,
        standardize_stats=stats,
    )
    count = 0
    for image_path in images:
        save_hsv_dummy_prediction(image_path, output_root=config.predictions_root, config=hsv_config)
        count += 1
    print(
        f"predictions_written={count} output_root={config.predictions_root} "
        f"foreground={args.foreground} value_threshold={args.value_threshold} standardize={args.standardize}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
