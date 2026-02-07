import numpy as np
import mlx.core as mx
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.common_viz import viz_stage as _common_viz_stage



def _to_numpy(value):
    if hasattr(value, "shape") and hasattr(value, "tolist"):
        try:
            return np.asarray(value.tolist(), dtype=np.float32)
        except Exception:
            return None
    try:
        arr = np.asarray(value)
        if arr.dtype == np.dtype("O"):
            return None
        return arr
    except Exception:
        return None


def viz_stage(stage, scope):
    _common_viz_stage(stage, scope, _to_numpy, framework="mlx")
