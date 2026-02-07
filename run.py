#!/usr/bin/env python3
"""Universal runner that uses the framework selected by setup_framework.py."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CONFIG_FILE = Path(".t2c/config.json")
CHAPTER_FILES = [
    "0_computational_primitives.py",
    "1_automatic_differentiation.py",
    "2_optimization_theory.py",
    "3_neural_theory.py",
    "4_advanced_theory.py",
    "5_research_frontiers.py",
    "6_theoretical_limits.py",
]


def python_in_venv(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def load_config() -> dict[str, str]:
    if not CONFIG_FILE.exists():
        print(
            "No active framework config found. Run `python setup_framework.py <framework>` first.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))


def validate_script(framework: str) -> str:
    if framework == "mlx":
        return "scripts/mlx/test_mlx_setup.py"
    return f"scripts/{framework}/test_{framework}_setup.py"


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chapters/validation for active framework.")
    parser.add_argument(
        "target",
        help="One of: validate, 0,1,2,3,4,5,6, all",
    )
    parser.add_argument(
        "--framework",
        choices=["mlx", "jax", "pytorch", "numpy", "keras", "cupy"],
        help="Override framework instead of using saved active framework.",
    )
    parser.add_argument(
        "--venv",
        help="Override venv path instead of using saved config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()

    framework = args.framework or config["framework"]
    venv_dir = Path(args.venv or config["venv"])
    py = python_in_venv(venv_dir)
    if not py.exists():
        print(f"Python executable not found in venv: {py}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    if framework == "mlx":
        env["T2C_BACKEND"] = "mlx"

    scripts: list[str]
    if args.target == "validate":
        scripts = [validate_script(framework)]
    elif args.target == "all":
        scripts = [f"scripts/{framework}/{name}" for name in CHAPTER_FILES]
    elif args.target in {str(i) for i in range(7)}:
        idx = int(args.target)
        scripts = [f"scripts/{framework}/{CHAPTER_FILES[idx]}"]
    else:
        print("Invalid target. Use: validate, 0,1,2,3,4,5,6, all", file=sys.stderr)
        return 1

    for script in scripts:
        run_cmd([str(py), script], env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
