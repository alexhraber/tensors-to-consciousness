#!/usr/bin/env python3
"""Unified framework setup and validation runner.

Creates/uses a virtual environment via `uv`, installs framework dependencies,
then runs the corresponding validation script.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .runtime import SUPPORTED_FRAMEWORKS, python_in_venv, save_config

FRAMEWORK_CONFIG = {
    "mlx": {
        "deps": ["mlx"],
    },
    "jax": {
        "deps": ["jax[cpu]"],
    },
    "pytorch": {
        "deps": ["torch"],
    },
    "numpy": {
        "deps": ["numpy"],
    },
    "keras": {
        "deps": ["keras", "tensorflow"],
    },
    "cupy": {
        "deps": ["cupy-cuda12x"],
    },
}
COMMON_DEPS = ["matplotlib"]


def resolve_framework_deps(framework: str) -> list[str]:
    deps = list(FRAMEWORK_CONFIG[framework]["deps"])
    # In Linux containers (including Apple Docker Desktop VMs), install MLX CPU backend.
    if framework == "mlx" and sys.platform != "darwin":
        return ["mlx[cpu]"]
    return deps


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(cmd)}")
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)

    # `uv` defaults to $XDG_CACHE_HOME (~/.cache), which can be unwritable in
    # sandboxed/CI environments. Keep it repo-local by default.
    if "UV_CACHE_DIR" not in merged_env:
        repo_root = Path(__file__).resolve().parents[1]
        uv_cache_dir = repo_root / ".uv-cache"
        uv_cache_dir.mkdir(parents=True, exist_ok=True)
        merged_env["UV_CACHE_DIR"] = str(uv_cache_dir)

    subprocess.run(cmd, check=True, env=merged_env)


def write_active_config(framework: str, venv_dir: Path) -> None:
    save_config(
        {
        "framework": framework,
        "venv": str(venv_dir),
        }
    )


def setup_one(framework: str, venv_dir: Path, skip_validate: bool) -> None:
    py = python_in_venv(venv_dir)
    deps = resolve_framework_deps(framework)

    # Latest-first dependency policy: refresh to newest available versions on setup.
    run_cmd(["uv", "pip", "install", "--python", str(py), "--upgrade", *COMMON_DEPS, *deps])

    if skip_validate:
        return

    run_cmd([str(py), "-m", "tools.validate", "--framework", framework])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up a framework environment and run validation."
    )
    parser.add_argument(
        "framework",
        choices=[*SUPPORTED_FRAMEWORKS, "all"],
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
        help="Install dependencies but do not run validation checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if shutil.which("uv") is None:
        print("Error: `uv` is required but was not found in PATH.", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[1]
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
        print("Use these commands:")
        print("  explorer --help")
        print("  explorer list-transforms")
        print("  explorer run --framework jax --transforms default")

    print("\nSetup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
