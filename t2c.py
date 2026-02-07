#!/usr/bin/env python3
"""Primary execution entrypoint for configured framework research modules."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from tools.runtime import SUPPORTED_FRAMEWORKS, load_config, python_in_venv

CHAPTER_FILES = [
    "0_computational_primitives.py",
    "1_automatic_differentiation.py",
    "2_optimization_theory.py",
    "3_neural_theory.py",
    "4_advanced_theory.py",
    "5_research_frontiers.py",
    "6_theoretical_limits.py",
]


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run research-module/validation targets for a selected framework."
    )
    parser.add_argument(
        "target",
        help="One of: validate, viz, 0,1,2,3,4,5,6, all",
    )
    parser.add_argument(
        "--framework",
        choices=list(SUPPORTED_FRAMEWORKS),
        help="Framework override. If missing, use saved active framework.",
    )
    parser.add_argument(
        "--venv",
        help="Venv override. If missing, use saved config venv.",
    )
    parser.add_argument(
        "--no-setup",
        action="store_true",
        help="Do not auto-run setup when config/venv is missing.",
    )
    return parser.parse_args()


def ensure_setup_if_needed(
    framework: str | None,
    venv: Path | None,
    framework_overridden: bool,
    allow_setup: bool,
    env: dict[str, str],
) -> dict[str, str]:
    config: dict[str, str] = {}
    config_exists = True
    try:
        config = load_config()
    except RuntimeError:
        config = {}
        config_exists = False

    framework = framework or config.get("framework")
    venv_dir = Path(venv or config.get("venv", ".venv"))
    py = python_in_venv(venv_dir)

    if framework is None:
        raise RuntimeError(
            "No active framework configured. Run `python -m tools.setup <framework>` once "
            "or pass `--framework`."
        )

    requested_differs = framework_overridden and framework != config.get("framework")
    needs_setup = (not config_exists) or (not py.exists()) or requested_differs

    if not needs_setup:
        return {"framework": framework, "venv": str(venv_dir)}

    if not allow_setup:
        raise RuntimeError(
            f"Setup needed for framework '{framework}'. Run `python -m tools.setup {framework}` first."
        )

    run_cmd(
        [sys.executable, "-m", "tools.setup", framework, "--venv", str(venv_dir), "--skip-validate"],
        env=env,
    )
    return {"framework": framework, "venv": str(venv_dir)}


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)
    env = os.environ.copy()

    try:
        config = ensure_setup_if_needed(
            framework=args.framework,
            venv=Path(args.venv) if args.venv else None,
            framework_overridden=args.framework is not None,
            allow_setup=not args.no_setup,
            env=env,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    framework = config["framework"]
    venv_dir = Path(config["venv"])
    py = python_in_venv(venv_dir)

    if args.target == "validate":
        run_cmd([str(py), "-m", "tools.validate", "--framework", framework], env=env)
        return 0
    if args.target == "viz":
        run_cmd([str(py), "-m", "tools.viz_terminal", "--framework", framework], env=env)
        return 0
    if args.target == "all":
        scripts = [f"scripts/{framework}/{name}" for name in CHAPTER_FILES]
    elif args.target in {str(i) for i in range(7)}:
        idx = int(args.target)
        scripts = [f"scripts/{framework}/{CHAPTER_FILES[idx]}"]
    else:
        print("Invalid target. Use: validate, viz, 0,1,2,3,4,5,6, all", file=sys.stderr)
        return 1

    for script in scripts:
        run_cmd([str(py), script], env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
