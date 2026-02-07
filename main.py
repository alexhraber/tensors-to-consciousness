#!/usr/bin/env python3
"""Primary execution entrypoint for configured framework transform playground."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from transforms.registry import list_transform_keys
from transforms.registry import resolve_transform_keys
from tools.runtime import SUPPORTED_FRAMEWORKS, load_config, python_in_venv

DEFAULT_FRAMEWORK = "numpy"


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run T2C interactive explorer or sandbox transform targets for a selected framework."
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="One of: validate, viz, run (default: interactive explorer).",
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
    parser.add_argument(
        "--inputs",
        help="Path to input-override JSON (or raw JSON string) used by playground runs.",
    )
    parser.add_argument(
        "--transforms",
        help="Comma-separated transform keys, or 'default'/'all' (default: default).",
    )
    parser.add_argument(
        "--transform",
        help="Initial focused transform selector for TUI (key or title fragment).",
    )
    parser.add_argument(
        "--list-transforms",
        action="store_true",
        help="Print available transform keys and exit.",
    )
    parser.add_argument(
        "-c",
        "--cli",
        action="store_true",
        help="Force CLI execution flow (skip interactive explorer on default run).",
    )
    return parser.parse_args()


def ensure_setup_if_needed(
    framework: str | None,
    venv: Path | None,
    framework_overridden: bool,
    allow_setup: bool,
    env: dict[str, str],
) -> tuple[dict[str, str], bool]:
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
            "No active framework configured. Run a t2c command with "
            "`--framework <framework>` once (example: `python main.py validate --framework jax`)."
        )

    requested_differs = framework_overridden and framework != config.get("framework")
    needs_setup = (not config_exists) or (not py.exists()) or requested_differs

    if not needs_setup:
        return {"framework": framework, "venv": str(venv_dir)}, False

    if not allow_setup:
        raise RuntimeError(
            f"Setup needed for framework '{framework}'. Re-run with setup enabled or run "
            f"`python main.py validate --framework {framework}`."
        )

    run_cmd(
        [sys.executable, "-m", "tools.setup", framework, "--venv", str(venv_dir), "--skip-validate"],
        env=env,
    )
    return {"framework": framework, "venv": str(venv_dir)}, True


def main() -> int:
    args = parse_args()
    if getattr(args, "list_transforms", False):
        print("Available transforms:")
        for key in list_transform_keys():
            print(f"- {key}")
        return 0

    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)
    env = os.environ.copy()
    inputs = getattr(args, "inputs", None)
    transform_focus = getattr(args, "transform", None)
    transform_selector = getattr(args, "transforms", None)

    existing_framework = None
    try:
        existing_framework = load_config().get("framework")
    except RuntimeError:
        existing_framework = None

    onboarding_mode = args.target is None
    framework = args.framework
    if onboarding_mode and framework is None and existing_framework is None:
        framework = DEFAULT_FRAMEWORK

    if inputs:
        env["T2C_INPUTS"] = inputs

    try:
        config, setup_ran = ensure_setup_if_needed(
            framework=framework,
            venv=Path(args.venv) if args.venv else None,
            framework_overridden=framework is not None,
            allow_setup=not args.no_setup,
            env=env,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    framework = config["framework"]
    venv_dir = Path(config["venv"])
    py = python_in_venv(venv_dir)

    force_cli = bool(getattr(args, "cli", False))
    if args.target is None:
        if setup_ran:
            run_cmd([str(py), "-m", "tools.validate", "--framework", framework], env=env)
        if force_cli:
            cmd = [
                str(py),
                "-m",
                "tools.playground",
                "--framework",
                framework,
                "--transforms",
                transform_selector or "default",
                "--viz",
            ]
            run_cmd(cmd, env=env)
            return 0
        cmd = [str(py), "-m", "tools.tui", "--framework", framework]
        if transform_selector:
            cmd.extend(["--transforms", transform_selector])
        if transform_focus:
            cmd.extend(["--transform", transform_focus])
        run_cmd(cmd, env=env)
        return 0

    if args.target == "validate":
        run_cmd([str(py), "-m", "tools.validate", "--framework", framework], env=env)
        return 0

    if args.target == "viz":
        cmd = [str(py), "-m", "tools.tui", "--framework", framework]
        if transform_selector:
            cmd.extend(["--transforms", transform_selector])
        if transform_focus:
            cmd.extend(["--transform", transform_focus])
        run_cmd(cmd, env=env)
        return 0

    if args.target == "run":
        try:
            _ = resolve_transform_keys(transform_selector)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        cmd = [
            str(py),
            "-m",
            "tools.playground",
            "--framework",
            framework,
            "--transforms",
            transform_selector or "default",
            "--viz",
        ]
        run_cmd(cmd, env=env)
        return 0

    print("Invalid target. Use: validate, viz, run", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
