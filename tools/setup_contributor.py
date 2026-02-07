#!/usr/bin/env python3
"""Contributor bootstrap utilities for local hooks/tooling."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".githooks"


def _run(cmd: list[str], quiet: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    if not quiet:
        print(f"+ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True, capture_output=quiet)


def _has_pre_commit_command() -> bool:
    return shutil.which("pre-commit") is not None


def _has_pre_commit_module() -> bool:
    return importlib.util.find_spec("pre_commit") is not None


def _pre_commit_runner() -> list[str] | None:
    if _has_pre_commit_command():
        return ["pre-commit"]
    if _has_pre_commit_module():
        return [sys.executable, "-m", "pre_commit"]
    return None


def _ensure_githooks(quiet: bool) -> None:
    _run([sys.executable, "tools/install_githooks.py"], quiet=quiet)


def _hooks_path_configured() -> bool:
    try:
        configured = subprocess.check_output(
            ["git", "config", "--get", "core.hooksPath"],
            cwd=ROOT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return False
    if not configured:
        return False
    configured_path = Path(configured)
    if not configured_path.is_absolute():
        configured_path = (ROOT / configured_path).resolve()
    return configured_path == HOOKS.resolve()


def _install_pre_commit(quiet: bool) -> bool:
    if shutil.which("uv") is not None:
        result = _run(["uv", "pip", "install", "--python", sys.executable, "pre-commit"], quiet=quiet, check=False)
        if result.returncode == 0:
            return True
    result = _run([sys.executable, "-m", "pip", "install", "pre-commit"], quiet=quiet, check=False)
    return result.returncode == 0


def _ensure_pre_commit(quiet: bool) -> list[str] | None:
    runner = _pre_commit_runner()
    if runner is not None:
        return runner
    if not _install_pre_commit(quiet=quiet):
        return None
    return _pre_commit_runner()


def bootstrap(quiet: bool = False, strict: bool = False) -> int:
    runner = _pre_commit_runner()
    if runner is not None and _hooks_path_configured():
        return 0

    _ensure_githooks(quiet=quiet)
    if runner is None:
        runner = _ensure_pre_commit(quiet=quiet)
    if runner is None:
        message = (
            "warning: pre-commit could not be installed automatically. "
            "Install it manually with `python -m pip install pre-commit`."
        )
        print(message, file=sys.stderr)
        return 1 if strict else 0
    _run([*runner, "install", "--install-hooks"], quiet=quiet, check=False)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap contributor hooks/tooling.")
    parser.add_argument("--quiet", action="store_true", help="Suppress command echo output.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if bootstrap cannot install required tooling.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return bootstrap(quiet=args.quiet, strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
