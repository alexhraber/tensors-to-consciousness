from __future__ import annotations

import os
from functools import lru_cache
from types import ModuleType
from typing import Any


def _disabled() -> bool:
    v = (os.environ.get("EXPLORER_DISABLE_CORE") or os.environ.get("EXPLORER_DISABLE_ACCEL") or "")
    return v.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@lru_cache(maxsize=1)
def load_core() -> ModuleType | None:
    if _disabled():
        return None
    try:
        import core as core_mod
    except Exception:
        return None
    return core_mod


def ascii_heatmap(arr: Any, *, width: int, height: int) -> str | None:
    core_mod = load_core()
    if core_mod is None:
        return None
    try:
        return core_mod.ascii_heatmap(arr, width, height)
    except Exception:
        return None


def pixel_heatmap(arr: Any, *, width: int, height: int) -> str | None:
    core_mod = load_core()
    if core_mod is None:
        return None
    try:
        return core_mod.pixel_heatmap(arr, width, height)
    except Exception:
        return None


def parse_assignment(expr: str) -> tuple[str, str] | None:
    core_mod = load_core()
    if core_mod is None:
        return None
    try:
        value = core_mod.parse_assignment(expr)
    except Exception:
        return None
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(x, str) for x in value):
        return value[0], value[1]
    return None


def normalize_platform(value: str | None, *, default: str) -> str | None:
    core_mod = load_core()
    if core_mod is None:
        return None
    try:
        out = core_mod.normalize_platform(value, default)
    except Exception:
        return None
    return out if isinstance(out, str) else None


def default_venv(framework: str) -> str | None:
    core_mod = load_core()
    if core_mod is None:
        return None
    try:
        out = core_mod.default_venv(framework)
    except Exception:
        return None
    return out if isinstance(out, str) else None


def frame_patch(prev: str, next_: str) -> str | None:
    core_mod = load_core()
    if core_mod is None:
        return None
    try:
        out = core_mod.frame_patch(prev, next_)
    except Exception:
        return None
    return out if isinstance(out, str) else None
