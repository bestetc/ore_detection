#!/usr/bin/env python
"""Audit project datasets and write CSV inventories."""

from __future__ import annotations

import argparse
from pathlib import Path

from ore_detection.data.inventory import (
    inventory_baseline_images,
    inventory_source_dataset,
    write_inventory_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", default="datasets", help="Root folder containing baseline/set_1/set_2/set_3")
    parser.add_argument("--output-dir", default="data_work/inventories", help="Where CSV inventories are written")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets_root = Path(args.datasets_root)
    output_dir = Path(args.output_dir)

    baseline_records = inventory_baseline_images(datasets_root / "baseline")
    write_inventory_csv(baseline_records, output_dir / "baseline_inventory.csv")
    print(f"baseline: {len(baseline_records)} images")

    for dataset_name in ("set_1", "set_2", "set_3"):
        records = inventory_source_dataset(datasets_root / dataset_name, dataset_name=dataset_name)
        write_inventory_csv(records, output_dir / f"{dataset_name}_inventory.csv")
        mismatches = sum(1 for record in records if not record.get("shape_match", False))
        print(f"{dataset_name}: {len(records)} pairs, {mismatches} shape mismatches")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
