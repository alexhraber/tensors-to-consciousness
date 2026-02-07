import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.shinkei import viz_stage as _shinkei_viz_stage
from tools.input_controls import annotate, metadata_for_scope, resolve_seed, tune_normal, tune_uniform


DTYPE = torch.float32
GENERATOR = torch.Generator().manual_seed(resolve_seed("pytorch", 0))


def normal(shape, dtype=DTYPE):
    cfg = tune_normal("pytorch", shape)
    sample = torch.randn(cfg["shape"], generator=GENERATOR, dtype=dtype)
    return sample * cfg["std"] + cfg["mean"]


def uniform(low, high, shape, dtype=DTYPE):
    cfg = tune_uniform("pytorch", low, high, shape)
    return cfg["low"] + (cfg["high"] - cfg["low"]) * torch.rand(
        cfg["shape"],
        generator=GENERATOR,
        dtype=dtype,
    )


def init_linear(in_dim, out_dim, dtype=DTYPE):
    weight = normal((in_dim, out_dim), dtype=dtype) * torch.sqrt(torch.tensor(2.0 / in_dim, dtype=dtype))
    bias = torch.zeros((out_dim,), dtype=dtype)
    return {"weight": weight, "bias": bias}


def linear(params, x):
    return x @ params["weight"] + params["bias"]


def relu(x):
    return F.relu(x)


def sigmoid(x):
    return torch.sigmoid(x)


def gelu(x):
    return F.gelu(x)


def softmax(x, dim=-1):
    return F.softmax(x, dim=dim)


def tree_l2_norm(tree):
    if isinstance(tree, dict):
        leaves = tree.values()
    else:
        leaves = tree
    total = torch.tensor(0.0, dtype=DTYPE)
    for leaf in leaves:
        total = total + torch.sum(leaf * leaf)
    return torch.sqrt(total)


def scalar(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu())
    return float(x)


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
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
    _shinkei_viz_stage(
        stage,
        scope,
        _to_numpy,
        framework="pytorch",
        metadata=metadata_for_scope("pytorch", scope),
    )
