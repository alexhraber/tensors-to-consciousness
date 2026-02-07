from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np


def _ascii_heatmap(arr: np.ndarray, width: int = 36, height: int = 12) -> str:
    ramp = " .:-=+*#%@"
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return "(empty)"

    h, w = arr.shape
    ys = np.linspace(0, h - 1, num=min(height, h), dtype=int)
    xs = np.linspace(0, w - 1, num=min(width, w), dtype=int)
    sampled = arr[np.ix_(ys, xs)]
    mn = float(np.min(sampled))
    mx = float(np.max(sampled))
    if mx - mn < 1e-12:
        norm = np.zeros_like(sampled, dtype=np.float32)
    else:
        norm = (sampled - mn) / (mx - mn)
    idx = np.clip((norm * (len(ramp) - 1)).astype(int), 0, len(ramp) - 1)
    lines = ["".join(ramp[i] for i in row) for row in idx]
    return "\n".join(lines)


def viz_stage(
    stage: str,
    scope: dict[str, Any],
    to_numpy: Callable[[Any], np.ndarray | None],
    framework: str,
    limit: int = 3,
) -> None:
    if os.environ.get("T2C_VIZ", "1").strip().lower() in {"0", "false", "off", "no"}:
        return

    candidates: list[tuple[str, np.ndarray]] = []
    for name, value in scope.items():
        if name.startswith("_"):
            continue
        arr = to_numpy(value)
        if arr is None:
            continue
        if arr.ndim == 0:
            continue
        if arr.size < 4:
            continue
        candidates.append((name, arr))

    if not candidates:
        return

    # Prefer larger tensors to show the most informative surfaces.
    candidates.sort(key=lambda item: item[1].size, reverse=True)
    print(f"\n[VIS:{framework}] {stage}")
    for name, arr in candidates[:limit]:
        arr_f = np.asarray(arr, dtype=np.float32)
        print(
            f"- {name}: shape={arr_f.shape} mean={arr_f.mean():.4f} "
            f"std={arr_f.std():.4f} min={arr_f.min():.4f} max={arr_f.max():.4f}"
        )
        print(_ascii_heatmap(arr_f))
