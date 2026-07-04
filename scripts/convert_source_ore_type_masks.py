#!/usr/bin/env python
"""Convert source color masks to multiclass one-hot ore-type tensors."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ore_detection.data.ore_type_legend import load_legend_config, write_legend_metadata
from ore_detection.data.ore_type_mask import audit_color_mask_file, convert_color_mask_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/source_ore_type_legend.json")
    parser.add_argument("--source-root", default="datasets")
    parser.add_argument("--output-root", default="data_work/source_ore_type_masks")
    parser.add_argument("--dataset", action="append", choices=["set_1", "set_2", "set_3"])
    parser.add_argument("--dry-run", action="store_true", help="Audit color coverage without writing tensors.")
    parser.add_argument("--skip-existing", action="store_true", help="Do not rewrite existing .pt tensors.")
    parser.add_argument("--limit", type=int, default=None, help="Convert at most N masks per dataset.")
    return parser.parse_args()


def _iter_masks(source_root: Path, dataset: str) -> list[Path]:
    return sorted((source_root / dataset / "masks_colored").rglob("*.png"))


def main() -> int:
    args = parse_args()
    legend = load_legend_config(args.config)
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    datasets = args.dataset or ["set_1", "set_2", "set_3"]

    if not args.dry_run:
        write_legend_metadata(legend, output_root / "class_map.json")

    total_converted = 0
    total_unknown = 0
    class_pixels: Counter[str] = Counter()

    for dataset in datasets:
        mask_paths = _iter_masks(source_root, dataset)
        if args.limit is not None:
            mask_paths = mask_paths[: args.limit]
        dataset_converted = 0
        dataset_unknown = 0
        for source_path in mask_paths:
            relative = source_path.relative_to(source_root / dataset / "masks_colored")
            target_path = (output_root / dataset / relative).with_suffix(".pt")
            if args.dry_run:
                audit = audit_color_mask_file(source_path, dataset=dataset, legend=legend)
                dataset_unknown += int(audit["unknown_color_count"])
                if audit["unknown_color_count"]:
                    print(f"UNKNOWN dataset={dataset} path={source_path} colors={audit['unknown_colors']}")
                continue
            if args.skip_existing and target_path.exists():
                dataset_converted += 1
                continue
            stats = convert_color_mask_file(source_path, target_path, dataset=dataset, legend=legend)
            class_pixels.update(stats["class_pixels"])
            dataset_converted += 1

        total_converted += dataset_converted
        total_unknown += dataset_unknown
        if args.dry_run:
            print(f"audited dataset={dataset} masks={len(mask_paths)} unknown_masks={dataset_unknown}")
        else:
            print(f"converted dataset={dataset} masks={dataset_converted} output={output_root / dataset}")

    if not args.dry_run:
        print(f"converted_total={total_converted} output_root={output_root}")
        for class_name in legend.class_names:
            print(f"class_pixels {class_name}={class_pixels[class_name]}")

    return 1 if total_unknown else 0


if __name__ == "__main__":
    raise SystemExit(main())
