#!/usr/bin/env python3
"""Unified framework setup and validation runner.

Creates/uses a virtual environment via `uv`, installs framework dependencies,
then runs the corresponding validation script.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

FRAMEWORK_CONFIG = {
    "mlx": {
        "deps": ["mlx"],
        "validate": "scripts/mlx/test_mlx_setup.py",
    },
    "jax": {
        "deps": ["jax[cpu]"],
        "validate": "scripts/jax/test_jax_setup.py",
    },
    "pytorch": {
        "deps": ["torch"],
        "validate": "scripts/pytorch/test_pytorch_setup.py",
    },
    "numpy": {
        "deps": ["numpy"],
        "validate": "scripts/numpy/test_numpy_setup.py",
    },
    "keras": {
        "deps": ["keras", "tensorflow"],
        "validate": "scripts/keras/test_keras_setup.py",
    },
    "cupy": {
        "deps": ["cupy-cuda12x"],
        "validate": "scripts/cupy/test_cupy_setup.py",
    },
}
CONFIG_DIR = Path(".t2c")
CONFIG_FILE = CONFIG_DIR / "config.json"


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def write_active_config(framework: str, venv_dir: Path) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "framework": framework,
        "venv": str(venv_dir),
    }
    CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def python_in_venv(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def setup_one(framework: str, venv_dir: Path, skip_validate: bool) -> None:
    config = FRAMEWORK_CONFIG[framework]
    py = python_in_venv(venv_dir)

    run_cmd(["uv", "pip", "install", "--python", str(py), *config["deps"]])

    if skip_validate:
        return

    validate_script = Path(config["validate"])
    env = os.environ.copy()
    if framework == "mlx":
        env["T2C_BACKEND"] = "mlx"
    run_cmd([str(py), str(validate_script)], env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up a framework environment and run validation."
    )
    parser.add_argument(
        "framework",
        choices=[*FRAMEWORK_CONFIG.keys(), "all"],
        help="Framework to set up.",
    )
    parser.add_argument(
        "--venv",
        default=".venv",
        help="Virtual environment directory (default: .venv).",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Install dependencies but do not run validation scripts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if shutil.which("uv") is None:
        print("Error: `uv` is required but was not found in PATH.", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    venv_dir = Path(args.venv)
    run_cmd(["uv", "venv", str(venv_dir)])

    frameworks = list(FRAMEWORK_CONFIG.keys()) if args.framework == "all" else [args.framework]
    for fw in frameworks:
        print(f"\n=== Setting up {fw} ===")
        setup_one(fw, venv_dir, args.skip_validate)

    if args.framework != "all":
        write_active_config(args.framework, venv_dir)
        print(f"\nActive framework set to: {args.framework}")
        print("Use universal commands:")
        print("  python run.py validate")
        print("  python run.py 0")
        print("  python run.py all")

    print("\nSetup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
