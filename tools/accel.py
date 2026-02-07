from __future__ import annotations

import os
from functools import lru_cache
from types import ModuleType
from typing import Any


def _disabled() -> bool:
    return os.environ.get("EXPLORER_DISABLE_ACCEL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@lru_cache(maxsize=1)
def load_accel() -> ModuleType | None:
    if _disabled():
        return None
    try:
        import explorer_accel as core
    except Exception:
        return None
    return core


def ascii_heatmap(arr: Any, *, width: int, height: int) -> str | None:
    core = load_accel()
    if core is None:
        return None
    try:
        return core.ascii_heatmap(arr, width, height)
    except Exception:
        return None


def pixel_heatmap(arr: Any, *, width: int, height: int) -> str | None:
    core = load_accel()
    if core is None:
        return None
    try:
        return core.pixel_heatmap(arr, width, height)
    except Exception:
        return None


def parse_assignment(expr: str) -> tuple[str, str] | None:
    core = load_accel()
    if core is None:
        return None
    try:
        value = core.parse_assignment(expr)
    except Exception:
        return None
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(x, str) for x in value):
        return value[0], value[1]
    return None


def normalize_platform(value: str | None, *, default: str) -> str | None:
    core = load_accel()
    if core is None:
        return None
    try:
        out = core.normalize_platform(value, default)
    except Exception:
        return None
    return out if isinstance(out, str) else None


def default_venv(framework: str) -> str | None:
    core = load_accel()
    if core is None:
        return None
    try:
        out = core.default_venv(framework)
    except Exception:
        return None
    return out if isinstance(out, str) else None


def frame_patch(prev: str, next_: str) -> str | None:
    core = load_accel()
    if core is None:
        return None
    try:
        out = core.frame_patch(prev, next_)
    except Exception:
        return None
    return out if isinstance(out, str) else None
