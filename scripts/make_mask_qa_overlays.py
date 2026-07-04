#!/usr/bin/env python
"""Create source image + binary-mask overlay previews for visual QA."""

from __future__ import annotations

import argparse
from pathlib import Path

from ore_detection.training.source_dataset import list_source_samples, load_sample_images
from ore_detection.visualization.overlay import save_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--binary-masks-root", default="data_work/binary_masks")
    parser.add_argument("--output-dir", default="data_work/qa_overlays")
    parser.add_argument("--split", choices=["train", "test"])
    parser.add_argument("--limit-per-dataset", type=int, default=10)
    parser.add_argument("--alpha", type=int, default=96)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = list_source_samples(
        datasets_root=args.datasets_root,
        binary_masks_root=args.binary_masks_root,
        split=args.split,
    )
    output_dir = Path(args.output_dir) / "binary"
    written = 0
    per_dataset: dict[str, int] = {}
    for sample in samples:
        count = per_dataset.get(sample.dataset, 0)
        if count >= args.limit_per_dataset:
            continue
        image, mask = load_sample_images(sample)
        output_path = output_dir / sample.dataset / sample.split / f"{sample.stem}_overlay.png"
        save_overlay(image, mask, output_path, alpha=args.alpha)
        per_dataset[sample.dataset] = count + 1
        written += 1
    print(f"qa_overlays_written={written} output_dir={output_dir}")
    for dataset, count in sorted(per_dataset.items()):
        print(f"{dataset}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
