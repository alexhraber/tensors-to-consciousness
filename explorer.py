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
from tools import diagnostics
from tools.runtime import (
    SUPPORTED_FRAMEWORKS,
    default_framework_for_platform,
    load_config,
    python_in_venv,
)

DEFAULT_FRAMEWORK = default_framework_for_platform()


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    print(f"+ {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt as exc:
        raise SystemExit(130) from exc


def _default_venv_for_framework(framework: str) -> Path:
    return Path(f".venv-{framework}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run T2C interactive explorer or sandbox transform targets for a selected framework."
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="One of: validate, render, run (default: interactive explorer).",
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
        "--start-explorer",
        action="store_true",
        help="Skip TUI landing and open directly into explorer view.",
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
    if framework is None:
        raise RuntimeError(
            "No active framework configured. Run a command with "
            "`--framework <framework>` once (example: `python explorer.py validate --framework jax`)."
        )

    configured_framework = config.get("framework")
    configured_venv = config.get("venv")
    if venv is not None:
        venv_dir = Path(venv)
    elif config_exists and configured_venv and framework == configured_framework:
        venv_dir = Path(configured_venv)
    else:
        # Keep framework environments isolated by default.
        venv_dir = _default_venv_for_framework(framework)

    py = python_in_venv(venv_dir)

    requested_differs = framework_overridden and framework != config.get("framework")
    needs_setup = (not config_exists) or (not py.exists()) or requested_differs

    if not needs_setup:
        return {"framework": framework, "venv": str(venv_dir)}, False

    if not allow_setup:
        raise RuntimeError(
            f"Setup needed for framework '{framework}'. Re-run with setup enabled or run "
            f"`python explorer.py validate --framework {framework}`."
        )

    run_cmd(
        [sys.executable, "-m", "tools.setup", framework, "--venv", str(venv_dir), "--skip-validate"],
        env=env,
    )
    return {"framework": framework, "venv": str(venv_dir)}, True


def main() -> int:
    diagnostics.configure_logging()
    logger = diagnostics.get_logger("main")
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
    start_explorer = bool(getattr(args, "start_explorer", False))

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
        env["INPUTS"] = inputs

    logger.info("launch target=%s framework=%s", args.target or "interactive", framework or "auto")

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
                "--render",
            ]
            run_cmd(cmd, env=env)
            return 0
        cmd = [str(py), "-m", "tools.tui", "--framework", framework]
        if transform_selector:
            cmd.extend(["--transforms", transform_selector])
        if transform_focus:
            cmd.extend(["--transform", transform_focus])
        if start_explorer:
            cmd.append("--start-explorer")
        run_cmd(cmd, env=env)
        return 0

    if args.target == "validate":
        run_cmd([str(py), "-m", "tools.validate", "--framework", framework], env=env)
        return 0

    if args.target == "render":
        cmd = [str(py), "-m", "tools.tui", "--framework", framework]
        if transform_selector:
            cmd.extend(["--transforms", transform_selector])
        if transform_focus:
            cmd.extend(["--transform", transform_focus])
        if start_explorer:
            cmd.append("--start-explorer")
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
            "--render",
        ]
        run_cmd(cmd, env=env)
        return 0

    print("Invalid target. Use: validate, render, run", file=sys.stderr)
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
