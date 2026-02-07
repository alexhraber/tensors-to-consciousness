from __future__ import annotations

import os
from functools import lru_cache
from types import ModuleType
from typing import Any


def _disabled() -> bool:
    return os.environ.get("TTC_DISABLE_RUST_CORE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@lru_cache(maxsize=1)
def load_rust_core() -> ModuleType | None:
    if _disabled():
        return None
    try:
        import ttc_rust_core as core
    except Exception:
        return None
    return core


def ascii_heatmap(arr: Any, *, width: int, height: int) -> str | None:
    core = load_rust_core()
    if core is None:
        return None
    try:
        return core.ascii_heatmap(arr, width, height)
    except Exception:
        return None


def pixel_heatmap(arr: Any, *, width: int, height: int) -> str | None:
    core = load_rust_core()
    if core is None:
        return None
    try:
        return core.pixel_heatmap(arr, width, height)
    except Exception:
        return None
