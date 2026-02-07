from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from tools import core

DEFAULT_CONFIG_FILE = Path(".explorer/config.json")
DEFAULT_LEGACY_CONFIG_FILES = (
    Path(".config/config.json"),
    Path(".t2c/config.json"),
)

# Patchable in tests.
CONFIG_FILE = DEFAULT_CONFIG_FILE
SUPPORTED_FRAMEWORKS = ("mlx", "jax", "pytorch", "numpy", "keras", "cupy")
DEFAULT_PLATFORM = "gpu"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_DEBUG = False


def default_framework_for_platform(platform: str | None = None) -> str:
    return "mlx" if (platform or sys.platform) == "darwin" else "jax"


DEFAULT_FRAMEWORK = default_framework_for_platform()


def python_in_venv(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def with_config_defaults(config: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = config if isinstance(config, dict) else {}
    framework = str(raw.get("framework") or DEFAULT_FRAMEWORK)
    venv = str(raw.get("venv") or f".venv-{framework}")
    core_venv = core.default_venv(framework)
    if "venv" not in raw and core_venv:
        venv = core_venv

    platform = str(raw.get("platform") or DEFAULT_PLATFORM).lower()
    core_platform = core.normalize_platform(platform, default=DEFAULT_PLATFORM)
    if core_platform:
        platform = core_platform

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


def _load_raw_config_from_disk() -> dict[str, Any]:
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    # Only consult legacy locations when using the default config path. This keeps
    # unit tests (which patch CONFIG_FILE) isolated and deterministic.
    if CONFIG_FILE == DEFAULT_CONFIG_FILE:
        for p in DEFAULT_LEGACY_CONFIG_FILES:
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError


def load_config() -> dict[str, Any]:
    try:
        raw = _load_raw_config_from_disk()
    except FileNotFoundError:
        raise RuntimeError(
            "No active framework config found. Run `python -m tools.setup <framework>` first."
        )
    cfg = with_config_defaults(raw if isinstance(raw, dict) else {})
    # Migration: persist to the new location once we successfully load.
    if not CONFIG_FILE.exists():
        try:
            save_config(cfg)
        except Exception:
            pass
    return cfg


def load_config_optional() -> dict[str, object]:
    try:
        raw = _load_raw_config_from_disk()
    except Exception:
        return with_config_defaults({})
    return with_config_defaults(raw if isinstance(raw, dict) else {})


def validate_script_for_framework(framework: str) -> str:
    if framework not in SUPPORTED_FRAMEWORKS:
        raise RuntimeError(
            f"Unsupported framework '{framework}'. Run `python -m tools.setup <framework>` first."
        )
    return f"frameworks/{framework}/test_setup.py"
