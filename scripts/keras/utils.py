import numpy as np
import tensorflow as tf
from keras import ops
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.common_viz import viz_stage as _common_viz_stage
from tools.input_controls import annotate, metadata_for_scope, resolve_seed, tune_normal, tune_uniform


DTYPE = "float32"
RNG = tf.random.Generator.from_seed(resolve_seed("keras", 0))


def normal(shape, dtype=DTYPE):
    cfg = tune_normal("keras", shape)
    sample = RNG.normal(shape=cfg["shape"], dtype=getattr(tf, dtype))
    return sample * cfg["std"] + cfg["mean"]


def uniform(low, high, shape, dtype=DTYPE):
    cfg = tune_uniform("keras", low, high, shape)
    return RNG.uniform(
        shape=cfg["shape"],
        minval=cfg["low"],
        maxval=cfg["high"],
        dtype=getattr(tf, dtype),
    )


def init_linear(in_dim, out_dim, dtype=DTYPE):
    scale = tf.sqrt(tf.cast(2.0 / in_dim, getattr(tf, dtype)))
    weight = normal((in_dim, out_dim), dtype=dtype) * scale
    bias = tf.zeros((out_dim,), dtype=getattr(tf, dtype))
    return {"weight": weight, "bias": bias}


def linear(params, x):
    return ops.matmul(x, params["weight"]) + params["bias"]


def relu(x):
    return ops.relu(x)


def sigmoid(x):
    return ops.sigmoid(x)


def gelu(x):
    return ops.gelu(x)


def softmax(x, axis=-1):
    return ops.softmax(x, axis=axis)


def tree_l2_norm(tree):
    if isinstance(tree, dict):
        leaves = tree.values()
    else:
        leaves = tree
    total = tf.constant(0.0, dtype=tf.float32)
    for leaf in leaves:
        total = total + ops.sum(leaf * leaf)
    return ops.sqrt(total)


def scalar(x):
    return float(np.array(ops.convert_to_numpy(x)).item())


def finite_diff_grad_scalar(f, x, eps=1e-4):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def finite_diff_grad_vector(f, x, eps=1e-4):
    grad = np.zeros_like(x, dtype=np.float32)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (f(xp) - f(xm)) / (2 * eps)
    return grad


def _to_numpy(value):
    try:
        arr = ops.convert_to_numpy(value)
        arr = np.asarray(arr)
        if arr.dtype == np.dtype("O"):
            return None
        return arr
    except Exception:
        return None


def viz_stage(stage, scope):
    _common_viz_stage(
        stage,
        scope,
        _to_numpy,
        framework="keras",
        metadata=metadata_for_scope("keras", scope),
    )
