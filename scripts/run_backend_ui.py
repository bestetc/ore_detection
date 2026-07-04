#!/usr/bin/env python
"""Run the local backend UI for HSV dummy segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ore_detection.backend.app import run_server
from ore_detection.backend.service import BackendConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--predictions-root", default="data_work/predictions/ui")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = BackendConfig(
        project_root=Path.cwd(),
        datasets_root=Path(args.datasets_root),
        predictions_root=Path(args.predictions_root),
    )
    run_server(host=args.host, port=args.port, config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
