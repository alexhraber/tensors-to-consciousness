from __future__ import annotations

import inspect
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


_LHS_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=")
_META_REGISTRY: dict[tuple[str, str, str], str] = {}


def _to_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    raw = os.environ.get("T2C_INPUTS", "").strip()
    if not raw:
        return {}
    path = Path(raw)
    try:
        if path.exists():
            return _to_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}
    try:
        return _to_dict(json.loads(raw))
    except Exception:
        return {}


def _merge_dicts(*parts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        if not isinstance(part, dict):
            continue
        merged.update(part)
    return merged


def _normalize_shape(shape: Any) -> Any:
    if isinstance(shape, list):
        return tuple(shape)
    return shape


def _call_context(depth: int = 1) -> dict[str, Any]:
    frame = inspect.currentframe()
    for _ in range(depth):
        if frame is None:
            break
        frame = frame.f_back
    while frame is not None:
        file_name = Path(frame.f_code.co_filename).name
        if file_name not in {"input_controls.py", "utils.py"}:
            break
        frame = frame.f_back
    if frame is None:
        return {"script": "", "line": 0, "label": None}

    file_path = frame.f_code.co_filename
    script = Path(file_path).name
    line = frame.f_lineno
    label = None
    try:
        line_text = inspect.getframeinfo(frame).code_context[0]
        match = _LHS_RE.match(line_text)
        if match:
            label = match.group(1)
    except Exception:
        pass
    return {"script": script, "line": line, "label": label}


def _distribution_overrides(framework: str, dist: str, ctx: dict[str, Any]) -> dict[str, Any]:
    cfg = _load_config()
    base = _to_dict(cfg)
    fw = _to_dict(_to_dict(cfg.get("frameworks")).get(framework))
    script_name = str(ctx.get("script", ""))
    base_script = _to_dict(_to_dict(base.get("scripts")).get(script_name))
    fw_script = _to_dict(_to_dict(fw.get("scripts")).get(script_name))

    merged = _merge_dicts(
        _to_dict(base.get(dist)),
        _to_dict(fw.get(dist)),
        _to_dict(base_script.get(dist)),
        _to_dict(fw_script.get(dist)),
    )

    label = ctx.get("label")
    line = str(ctx.get("line", 0))
    call_overrides = _merge_dicts(
        _to_dict(_to_dict(base_script.get("calls")).get(line)),
        _to_dict(_to_dict(fw_script.get("calls")).get(line)),
        _to_dict(_to_dict(base_script.get("calls")).get(str(label))),
        _to_dict(_to_dict(fw_script.get("calls")).get(str(label))),
    )
    merged.update(call_overrides)
    return merged


def _record_metadata(framework: str, ctx: dict[str, Any], text: str) -> None:
    label = ctx.get("label")
    script = str(ctx.get("script", ""))
    if not label or not script:
        return
    _META_REGISTRY[(framework, script, str(label))] = text


def resolve_seed(framework: str, default_seed: int) -> int:
    cfg = _load_config()
    base_seed = cfg.get("seed", default_seed)
    fw_seed = _to_dict(_to_dict(cfg.get("frameworks")).get(framework)).get("seed", base_seed)
    try:
        return int(fw_seed)
    except Exception:
        return int(default_seed)


def tune_normal(
    framework: str,
    shape: Any,
    *,
    mean: float = 0.0,
    std: float = 1.0,
) -> dict[str, Any]:
    ctx = _call_context()
    ovr = _distribution_overrides(framework, "normal", ctx)
    tuned = {
        "shape": _normalize_shape(ovr.get("shape", shape)),
        "mean": float(ovr.get("mean", mean)),
        "std": float(ovr.get("std", std)),
    }
    _record_metadata(
        framework,
        ctx,
        f"Generated via normal(mean={tuned['mean']:.3g}, std={tuned['std']:.3g}, shape={tuned['shape']}).",
    )
    return tuned


def tune_uniform(
    framework: str,
    low: float,
    high: float,
    shape: Any,
) -> dict[str, Any]:
    ctx = _call_context()
    ovr = _distribution_overrides(framework, "uniform", ctx)
    tuned = {
        "shape": _normalize_shape(ovr.get("shape", shape)),
        "low": float(ovr.get("low", low)),
        "high": float(ovr.get("high", high)),
    }
    _record_metadata(
        framework,
        ctx,
        f"Generated via uniform(low={tuned['low']:.3g}, high={tuned['high']:.3g}, shape={tuned['shape']}).",
    )
    return tuned


def annotate(framework: str, name: str, description: str) -> None:
    ctx = _call_context()
    script = str(ctx.get("script", ""))
    if not script or not name:
        return
    _META_REGISTRY[(framework, script, str(name))] = description.strip()


def metadata_for_scope(framework: str, scope: dict[str, Any]) -> dict[str, str]:
    ctx = _call_context()
    script = str(ctx.get("script", ""))
    if not script:
        return {}
    out: dict[str, str] = {}
    for name in scope:
        if (framework, script, name) in _META_REGISTRY:
            out[name] = _META_REGISTRY[(framework, script, name)]
    explicit = scope.get("VIZ_META")
    if not isinstance(explicit, dict):
        raise ValueError(
            "shinkei viz contract violation: module must define `VIZ_META = {}` "
            "before calling viz_stage()."
        )
    for k, v in explicit.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out
