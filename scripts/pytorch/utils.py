import torch
import torch.nn.functional as F

DTYPE = torch.float32
GENERATOR = torch.Generator().manual_seed(0)


def normal(shape, dtype=DTYPE):
    return torch.randn(shape, generator=GENERATOR, dtype=dtype)


def uniform(low, high, shape, dtype=DTYPE):
    return low + (high - low) * torch.rand(shape, generator=GENERATOR, dtype=dtype)


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
