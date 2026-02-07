from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from tools import rust_core

CONFIG_FILE = Path(".config/config.json")
SUPPORTED_FRAMEWORKS = ("mlx", "jax", "pytorch", "numpy", "keras", "cupy")
DEFAULT_PLATFORM = "gpu"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_DEBUG = False


def default_framework_for_platform(platform: str | None = None) -> str:
    return "mlx" if (platform or sys.platform) == "darwin" else "numpy"


DEFAULT_FRAMEWORK = default_framework_for_platform()


def python_in_venv(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def with_config_defaults(config: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = config if isinstance(config, dict) else {}
    framework = str(raw.get("framework") or DEFAULT_FRAMEWORK)
    venv = str(raw.get("venv") or f".venv-{framework}")
    rust_venv = rust_core.default_venv(framework)
    if "venv" not in raw and rust_venv:
        venv = rust_venv

    platform = str(raw.get("platform") or DEFAULT_PLATFORM).lower()
    rust_platform = rust_core.normalize_platform(platform, default=DEFAULT_PLATFORM)
    if rust_platform:
        platform = rust_platform

    out: dict[str, Any] = {
        "framework": framework,
        "venv": venv,
        "platform": platform,
        "diagnostics": {
            "log_level": DEFAULT_LOG_LEVEL,
            "debug": DEFAULT_DEBUG,
        },
    }

    if out["platform"] not in {"cpu", "gpu"}:
        out["platform"] = DEFAULT_PLATFORM

    diag = raw.get("diagnostics")
    if isinstance(diag, dict):
        if "log_level" in diag:
            out["diagnostics"]["log_level"] = str(diag["log_level"])
        if "debug" in diag:
            out["diagnostics"]["debug"] = bool(diag["debug"])

    if "log_level" in raw and "log_level" not in (diag or {}):
        out["diagnostics"]["log_level"] = str(raw["log_level"])
    if "debug" in raw and "debug" not in (diag or {}):
        out["diagnostics"]["debug"] = bool(raw["debug"])

    return out


def save_config(config: dict[str, Any]) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(with_config_defaults(config), indent=2) + "\n", encoding="utf-8")


def load_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        raise RuntimeError(
            "No active framework config found. Run `python -m tools.setup <framework>` first."
        )
    return with_config_defaults(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))


def load_config_optional() -> dict[str, object]:
    if not CONFIG_FILE.exists():
        return with_config_defaults({})
    try:
        raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return with_config_defaults({})
    return with_config_defaults(raw if isinstance(raw, dict) else {})


def validate_script_for_framework(framework: str) -> str:
    if framework not in SUPPORTED_FRAMEWORKS:
        raise RuntimeError(
            f"Unsupported framework '{framework}'. Run `python -m tools.setup <framework>` first."
        )
    return f"frameworks/{framework}/test_setup.py"
