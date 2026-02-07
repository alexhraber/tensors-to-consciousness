"""MLX backend wiring."""

from t2c.frameworks.base import Backend


def load() -> Backend:
    import mlx.core as mx
    import mlx.nn as nn

    return Backend(name="mlx", core=mx, nn=nn)
