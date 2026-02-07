import math

import numpy as np
from tools.common_viz import viz_stage as _common_viz_stage

DTYPE = np.float32
RNG = np.random.default_rng(0)


def normal(shape, dtype=DTYPE):
    return RNG.normal(size=shape).astype(dtype)


def uniform(low, high, shape, dtype=DTYPE):
    return RNG.uniform(low, high, size=shape).astype(dtype)


def init_linear(in_dim, out_dim, dtype=DTYPE):
    weight = normal((in_dim, out_dim), dtype=dtype) * np.sqrt(np.array(2.0 / in_dim, dtype=dtype))
    bias = np.zeros((out_dim,), dtype=dtype)
    return {"weight": weight, "bias": bias}


def linear(params, x):
    return x @ params["weight"] + params["bias"]


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * (x**3))))


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def tree_l2_norm(tree):
    if isinstance(tree, dict):
        leaves = tree.values()
    else:
        leaves = tree
    total = np.array(0.0, dtype=DTYPE)
    for leaf in leaves:
        total = total + np.sum(leaf * leaf)
    return np.sqrt(total)


def scalar(x):
    if isinstance(x, np.ndarray):
        return float(x.item())
    return float(x)


def finite_diff_grad_scalar(f, x, eps=1e-4):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def finite_diff_grad_vector(f, x, eps=1e-4):
    grad = np.zeros_like(x, dtype=DTYPE)
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
        grad = np.zeros_like(value, dtype=DTYPE)
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
    if isinstance(value, np.ndarray):
        return value
    if np.isscalar(value):
        return None
    try:
        arr = np.asarray(value)
        if arr.dtype == np.dtype("O"):
            return None
        return arr
    except Exception:
        return None


def viz_stage(stage, scope):
    _common_viz_stage(stage, scope, _to_numpy, framework="numpy")
