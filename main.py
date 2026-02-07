#!/usr/bin/env python3
"""Primary execution entrypoint for configured framework algorithm playground."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from algos.registry import list_algorithm_keys
from algos.registry import resolve_algorithm_keys
from tools.runtime import SUPPORTED_FRAMEWORKS, load_config, python_in_venv

DEFAULT_FRAMEWORK = "numpy"


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run T2C interactive studio or sandbox algorithm targets for a selected framework."
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="One of: validate, viz, run (default: interactive studio).",
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
        "--algos",
        help="Comma-separated algorithm keys, or 'default'/'all' (default: default).",
    )
    parser.add_argument(
        "--algorithm",
        help="Initial algorithm selector for TUI (key or title fragment).",
    )
    parser.add_argument(
        "--list-algos",
        action="store_true",
        help="Print available algorithm keys and exit.",
    )
    parser.add_argument(
        "-c",
        "--cli",
        action="store_true",
        help="Force CLI execution flow (skip interactive studio on default run).",
    )
    return parser.parse_args()


def prompt_framework_choice() -> str:
    print("Select framework:")
    for idx, framework in enumerate(SUPPORTED_FRAMEWORKS, start=1):
        print(f"  {idx}. {framework}")
    while True:
        raw = input("Framework number or name: ").strip().lower()
        if raw in SUPPORTED_FRAMEWORKS:
            return raw
        if raw.isdigit():
            pos = int(raw)
            if 1 <= pos <= len(SUPPORTED_FRAMEWORKS):
                return SUPPORTED_FRAMEWORKS[pos - 1]
        print(f"Invalid selection: {raw}")


def prompt_inputs_override() -> str | None:
    print("Input overrides (optional): provide JSON file path or inline JSON.")
    raw = input("Inputs (Enter for random defaults): ").strip()
    return raw or None


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
    if getattr(args, "list_algos", False):
        print("Available algos:")
        for key in list_algorithm_keys():
            print(f"- {key}")
        return 0

    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)
    env = os.environ.copy()
    inputs = getattr(args, "inputs", None)
    algorithm_selector = getattr(args, "algorithm", None)
    algo_selector = getattr(args, "algos", None)

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
            cmd = [str(py), "-m", "tools.playground", "--framework", framework, "--algos", algo_selector or "default", "--viz"]
            run_cmd(cmd, env=env)
            return 0
        else:
            cmd = [str(py), "-m", "tools.tui", "--framework", framework]
            if algo_selector:
                cmd.extend(["--algos", algo_selector])
            if algorithm_selector:
                cmd.extend(["--algorithm", algorithm_selector])
            run_cmd(cmd, env=env)
            return 0
    elif args.target == "validate":
        run_cmd([str(py), "-m", "tools.validate", "--framework", framework], env=env)
        return 0
    elif args.target == "viz":
        cmd = [str(py), "-m", "tools.tui", "--framework", framework]
        if algo_selector:
            cmd.extend(["--algos", algo_selector])
        if algorithm_selector:
            cmd.extend(["--algorithm", algorithm_selector])
        run_cmd(cmd, env=env)
        return 0
    elif args.target == "run":
        try:
            _ = resolve_algorithm_keys(algo_selector)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        cmd = [str(py), "-m", "tools.playground", "--framework", framework, "--algos", algo_selector or "default", "--viz"]
        run_cmd(cmd, env=env)
        return 0
    else:
        print("Invalid target. Use: validate, viz, run", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
