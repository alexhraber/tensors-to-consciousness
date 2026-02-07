from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / ".t2c" / "config.json"
CHAPTER_FILES = [
    "0_computational_primitives.py",
    "1_automatic_differentiation.py",
    "2_optimization_theory.py",
    "3_neural_theory.py",
    "4_advanced_theory.py",
    "5_research_frontiers.py",
    "6_theoretical_limits.py",
]


def _load_framework() -> str:
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            fw = data.get("framework")
            if isinstance(fw, str) and fw:
                return fw
        except Exception:
            pass
    env_fw = os.environ.get("T2C_FRAMEWORK")
    if env_fw:
        return env_fw
    raise RuntimeError(
        "No active framework configured. Run `python setup.py <framework>` first."
    )


def run_target(target: str) -> None:
    framework = _load_framework()
    if framework not in {"mlx", "jax", "pytorch", "numpy", "keras", "cupy"}:
        raise RuntimeError(
            f"Unsupported framework '{framework}'. Run `python setup.py <framework>` first."
        )

    if target == "validate":
        script = "scripts/mlx/test_mlx_setup.py" if framework == "mlx" else f"scripts/{framework}/test_{framework}_setup.py"
    elif target in {str(i) for i in range(7)}:
        script = f"scripts/{framework}/{CHAPTER_FILES[int(target)]}"
    else:
        raise RuntimeError("Invalid target. Use validate or 0..6")

    sys.path.insert(0, str((ROOT / "scripts" / framework).resolve()))
    runpy.run_path(str((ROOT / script).resolve()), run_name="__main__")
