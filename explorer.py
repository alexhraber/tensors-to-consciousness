#!/usr/bin/env python3
"""Compatibility shim for the legacy Python entrypoint.

The product surface is the Rust `explorer` binary. This module remains for
backward compatibility with existing scripts/tests.

Important: integration tests patch symbols on the `explorer` module. We
therefore provide wrappers that can temporarily "inject" those patched symbols
into the real implementation module (`app.explorer`) before dispatching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import app.explorer as _impl

# Expose commonly patched/used symbols (defaults to the implementation).
DEFAULT_FRAMEWORK = _impl.DEFAULT_FRAMEWORK
SUPPORTED_FRAMEWORKS = _impl.SUPPORTED_FRAMEWORKS

run_cmd = _impl.run_cmd
parse_args = _impl.parse_args
load_config = _impl.load_config
python_in_venv = _impl.python_in_venv


def ensure_setup_if_needed(
    framework: str | None,
    venv: Path | None,
    framework_overridden: bool,
    allow_setup: bool,
    env: dict[str, str],
) -> tuple[dict[str, str], bool]:
    """Wrapper around `_impl.ensure_setup_if_needed` that honors test patches."""

    saved = {
        "run_cmd": _impl.run_cmd,
        "load_config": _impl.load_config,
        "python_in_venv": _impl.python_in_venv,
    }
    _impl.run_cmd = run_cmd
    _impl.load_config = load_config
    _impl.python_in_venv = python_in_venv
    try:
        return _impl.ensure_setup_if_needed(
            framework=framework,
            venv=venv,
            framework_overridden=framework_overridden,
            allow_setup=allow_setup,
            env=env,
        )
    finally:
        _impl.run_cmd = saved["run_cmd"]
        _impl.load_config = saved["load_config"]
        _impl.python_in_venv = saved["python_in_venv"]


def _inject_and_call_main() -> int:
    """Run `_impl.main()` while honoring any monkeypatches on this module."""

    # These are the symbols integration tests patch on the `explorer` module.
    inject: dict[str, Any] = {
        "run_cmd": run_cmd,
        "parse_args": parse_args,
        "ensure_setup_if_needed": ensure_setup_if_needed,
        "load_config": load_config,
        "python_in_venv": python_in_venv,
    }

    saved: dict[str, Any] = {}
    for name, val in inject.items():
        saved[name] = getattr(_impl, name)
        setattr(_impl, name, val)

    try:
        return cast(int, _impl.main())
    finally:
        for name, val in saved.items():
            setattr(_impl, name, val)


def main() -> int:
    return _inject_and_call_main()


if __name__ == "__main__":
    raise SystemExit(main())
