#!/usr/bin/env python
"""Temporary canonical project verification command.

This project does not depend on pytest yet. Use this script as the canonical
lightweight verification command until the ML stack/test runner is finalized.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    src_path = str(project_root / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]
    return subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=project_root,
        env=env,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
