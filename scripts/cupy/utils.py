import numpy as np
import cupy as cp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.common_viz import viz_stage as _common_viz_stage


DTYPE = cp.float32
RNG = cp.random.RandomState(0)


def normal(shape, dtype=DTYPE):
    return RNG.standard_normal(size=shape).astype(dtype)


def uniform(low, high, shape, dtype=DTYPE):
    return RNG.uniform(low, high, size=shape).astype(dtype)


def init_linear(in_dim, out_dim, dtype=DTYPE):
    weight = normal((in_dim, out_dim), dtype=dtype) * cp.sqrt(cp.array(2.0 / in_dim, dtype=dtype))
    bias = cp.zeros((out_dim,), dtype=dtype)
    return {"weight": weight, "bias": bias}


def linear(params, x):
    return x @ params["weight"] + params["bias"]


def relu(x):
    return cp.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + cp.exp(-x))


def gelu(x):
    return 0.5 * x * (1 + cp.tanh(cp.sqrt(2 / cp.pi) * (x + 0.044715 * (x**3))))


def softmax(x, axis=-1):
    shifted = x - cp.max(x, axis=axis, keepdims=True)
    exps = cp.exp(shifted)
    return exps / cp.sum(exps, axis=axis, keepdims=True)


def tree_l2_norm(tree):
    if isinstance(tree, dict):
        leaves = tree.values()
    else:
        leaves = tree
    total = cp.array(0.0, dtype=DTYPE)
    for leaf in leaves:
        total = total + cp.sum(leaf * leaf)
    return cp.sqrt(total)


def scalar(x):
    if isinstance(x, cp.ndarray):
        return float(cp.asnumpy(x).item())
    return float(x)


def finite_diff_grad_scalar(f, x, eps=1e-4):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def finite_diff_grad_vector(f, x, eps=1e-4):
    grad = cp.zeros_like(x, dtype=DTYPE)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (f(xp) - f(xm)) / (2 * eps)
    return grad


def finite_diff_grad_dict(loss_fn, params, eps=1e-4):
    grads = {}
    for name, value in params.items():
        grad = cp.zeros_like(value, dtype=DTYPE)
        it = np.ndindex(value.shape)
        for idx in it:
            orig = value[idx]
            value[idx] = orig + eps
            lp = loss_fn(params)
            value[idx] = orig - eps
            lm = loss_fn(params)
            value[idx] = orig
            grad[idx] = (lp - lm) / (2 * eps)
        grads[name] = grad
    return grads


def _to_numpy(value):
    if isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    if isinstance(value, np.ndarray):
        return value
    try:
        arr = np.asarray(value)
        if arr.dtype == np.dtype("O"):
            return None
        return arr
    except Exception:
        return None


def viz_stage(stage, scope):
    _common_viz_stage(stage, scope, _to_numpy, framework="cupy")
