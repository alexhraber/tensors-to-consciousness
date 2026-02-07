"""Framework selection and compatibility exports.

Set T2C_BACKEND to choose a backend.
Currently supported: mlx.
"""

from __future__ import annotations

import importlib
import os
from functools import lru_cache

from t2c.frameworks.base import Backend, BackendConfigurationError

_AVAILABLE_BACKENDS = {
    "mlx": "t2c.frameworks.mlx_backend",
}


def _backend_from_env() -> str:
    return os.environ.get("T2C_BACKEND", "mlx").strip().lower()


@lru_cache(maxsize=1)
def get_backend() -> Backend:
    name = _backend_from_env()
    module_path = _AVAILABLE_BACKENDS.get(name)

    if module_path is None:
        supported = ", ".join(sorted(_AVAILABLE_BACKENDS))
        raise BackendConfigurationError(
            f"Unsupported backend '{name}'. Supported backends: {supported}."
        )

    try:
        module = importlib.import_module(module_path)
        backend = module.load()
    except ImportError as exc:
        raise BackendConfigurationError(
            f"Backend '{name}' could not be imported. Install its dependencies and try again."
        ) from exc

    return backend


backend = get_backend()
mx = backend.core
nn = backend.nn

__all__ = ["Backend", "BackendConfigurationError", "backend", "get_backend", "mx", "nn"]
