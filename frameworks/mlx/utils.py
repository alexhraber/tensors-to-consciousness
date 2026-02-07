import numpy as np
import mlx.core as mx
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.input_controls import annotate, metadata_for_scope, resolve_seed, tune_normal, tune_uniform


DTYPE = mx.float32
try:
    mx.random.seed(resolve_seed("mlx", 0))
except Exception:
    pass


def normal(shape, dtype=DTYPE):
    cfg = tune_normal("mlx", shape)
    sample = mx.random.normal(cfg["shape"], dtype=dtype)
    return sample * cfg["std"] + cfg["mean"]


def uniform(low, high, shape, dtype=DTYPE):
    cfg = tune_uniform("mlx", low, high, shape)
    return mx.random.uniform(cfg["low"], cfg["high"], cfg["shape"], dtype=dtype)


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
