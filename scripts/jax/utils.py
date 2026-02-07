import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from tools.common_viz import viz_stage as _common_viz_stage

DTYPE = jnp.float32
_key = jax.random.PRNGKey(0)


def next_key():
    global _key
    _key, subkey = jax.random.split(_key)
    return subkey


def normal(shape, dtype=DTYPE):
    return jax.random.normal(next_key(), shape, dtype=dtype)


def uniform(low, high, shape, dtype=DTYPE):
    return jax.random.uniform(next_key(), shape, minval=low, maxval=high, dtype=dtype)


def init_linear(in_dim, out_dim, dtype=DTYPE):
    w = normal((in_dim, out_dim), dtype=dtype) * jnp.sqrt(jnp.array(2.0 / in_dim, dtype=dtype))
    b = jnp.zeros((out_dim,), dtype=dtype)
    return {"weight": w, "bias": b}


def linear(params, x):
    return x @ params["weight"] + params["bias"]


def relu(x):
    return jnn.relu(x)


def sigmoid(x):
    return jnn.sigmoid(x)


def gelu(x):
    return jnn.gelu(x)


def softmax(x, axis=-1):
    return jnn.softmax(x, axis=axis)


def tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(leaf * leaf) for leaf in leaves))


def _to_numpy(value):
    try:
        arr = np.asarray(value)
        if arr.dtype == np.dtype("O"):
            return None
        return arr
    except Exception:
        return None


def viz_stage(stage, scope):
    _common_viz_stage(stage, scope, _to_numpy, framework="jax")
