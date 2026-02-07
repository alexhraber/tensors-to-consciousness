from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_FILE = Path(".t2c/config.json")
SUPPORTED_FRAMEWORKS = ("mlx", "jax", "pytorch", "numpy", "keras", "cupy")


def python_in_venv(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def load_config() -> dict[str, str]:
    if not CONFIG_FILE.exists():
        raise RuntimeError(
            "No active framework config found. Run `python -m tools.setup <framework>` first."
        )
    return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))


def validate_script_for_framework(framework: str) -> str:
    if framework not in SUPPORTED_FRAMEWORKS:
        raise RuntimeError(
            f"Unsupported framework '{framework}'. Run `python -m tools.setup <framework>` first."
        )
    return f"frameworks/{framework}/test_setup.py"
