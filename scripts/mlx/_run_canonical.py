from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(script_name: str) -> None:
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    os.environ["T2C_BACKEND"] = "mlx"
    runpy.run_path(str(ROOT / script_name), run_name="__main__")
