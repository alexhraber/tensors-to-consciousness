import numpy as np
import mlx.core as mx
from scripts.common_viz import viz_stage as _common_viz_stage


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
