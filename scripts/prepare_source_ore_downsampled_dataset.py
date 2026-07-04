#!/usr/bin/env python
"""Prepare 4x-downsampled source ore images and masks for training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    project_root = _project_root()
    src = project_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from ore_detection.training.source_ore_downsample import prepare_downsampled_source_ore_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", default=str(project_root / "datasets"))
    parser.add_argument("--masks-root", default=str(project_root / "data_work" / "source_ore_type_masks"))
    parser.add_argument("--output-root", default=str(project_root / "data_work" / "source_ore_downsampled"))
    parser.add_argument("--const-path", default=str(project_root / "src" / "ore_detection" / "training" / "const.py"))
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--size-divisor", type=int, default=4)
    parser.add_argument("--datasets", nargs="+", default=["set_1", "set_2", "set_3"])
    args = parser.parse_args()

    summary = prepare_downsampled_source_ore_dataset(
        datasets_root=args.datasets_root,
        masks_root=args.masks_root,
        output_root=args.output_root,
        datasets=args.datasets,
        factor=args.factor,
        size_divisor=args.size_divisor,
        const_path=args.const_path,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
