from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_FILE = Path(".ttc/config.json")
LEGACY_CONFIG_FILE = Path(".t2c/config.json")
SUPPORTED_FRAMEWORKS = ("mlx", "jax", "pytorch", "numpy", "keras", "cupy")


def python_in_venv(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def env_get(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is not None:
        return value
    if name.startswith("TTC_"):
        legacy = "T2C_" + name[4:]
        return os.environ.get(legacy, default)
    return default


def resolve_config_file() -> Path:
    if CONFIG_FILE.exists():
        return CONFIG_FILE
    if LEGACY_CONFIG_FILE.exists():
        return LEGACY_CONFIG_FILE
    return CONFIG_FILE


def load_config() -> dict[str, str]:
    config_path = resolve_config_file()
    if not config_path.exists():
        raise RuntimeError(
            "No active framework config found. Run `python -m tools.setup <framework>` first."
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def validate_script_for_framework(framework: str) -> str:
    if framework not in SUPPORTED_FRAMEWORKS:
        raise RuntimeError(
            f"Unsupported framework '{framework}'. Run `python -m tools.setup <framework>` first."
        )
    return f"frameworks/{framework}/test_setup.py"
