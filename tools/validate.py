#!/usr/bin/env python3
"""Framework-agnostic validation entrypoint."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runtime import SUPPORTED_FRAMEWORKS, load_config, validate_script_for_framework


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run setup validation for a framework.")
    parser.add_argument(
        "--framework",
        choices=list(SUPPORTED_FRAMEWORKS),
        help="Framework to validate. Defaults to active configured framework.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    framework = args.framework or load_config()["framework"]
    script = validate_script_for_framework(framework)

    framework_dir = ROOT / "frameworks" / framework
    if str(framework_dir) not in sys.path:
        sys.path.insert(0, str(framework_dir))
    runpy.run_path(str((ROOT / script).resolve()), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
