from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from tools import runtime

_CONFIGURED = False
_DEBUG_ENABLED = False


@dataclass(frozen=True)
class DiagnosticsConfig:
    level_name: str
    debug_enabled: bool


def _parse_bool(raw: Any, *, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if not isinstance(raw, str):
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_level(raw: str | None) -> str:
    if not raw:
        return "INFO"
    value = raw.strip().upper()
    allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
    return value if value in allowed else "INFO"


def configure_logging() -> DiagnosticsConfig:
    global _CONFIGURED, _DEBUG_ENABLED
    cfg = runtime.load_config_optional()
    cfg_diag = cfg.get("diagnostics", {}) if isinstance(cfg, dict) else {}
    if not isinstance(cfg_diag, dict):
        cfg_diag = {}

    cfg_level = cfg_diag.get("log_level", cfg.get("log_level", "INFO") if isinstance(cfg, dict) else "INFO")
    cfg_debug = cfg_diag.get("debug", cfg.get("debug", False) if isinstance(cfg, dict) else False)

    level_name = _normalize_level(str(cfg_level) if cfg_level is not None else "INFO")
    _DEBUG_ENABLED = _parse_bool(cfg_debug, default=False)

    env_level = os.environ.get("LOG_LEVEL")
    if env_level is not None:
        level_name = _normalize_level(env_level)
    if "DEBUG" in os.environ:
        _DEBUG_ENABLED = _parse_bool(os.environ.get("DEBUG"), default=_DEBUG_ENABLED)

    if _DEBUG_ENABLED and level_name != "DEBUG":
        level_name = "DEBUG"

    if not _CONFIGURED:
        logging.basicConfig(
            level=getattr(logging, level_name, logging.INFO),
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
        _CONFIGURED = True
    else:
        logging.getLogger().setLevel(getattr(logging, level_name, logging.INFO))

    return DiagnosticsConfig(level_name=level_name, debug_enabled=_DEBUG_ENABLED)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def debug_enabled() -> bool:
    return _DEBUG_ENABLED


def kernel_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    if not _DEBUG_ENABLED:
        return
    payload = {"event": event, **fields}
    logger.debug("kernel=%s", json.dumps(payload, sort_keys=True, default=str))


def timed_event(logger: logging.Logger, event: str, **fields: Any) -> tuple[float, dict[str, Any]]:
    started = perf_counter()
    payload = dict(fields)
    kernel_event(logger, f"{event}.start", **payload)
    return started, payload


def finish_timed_event(logger: logging.Logger, event: str, started: float, **fields: Any) -> None:
    elapsed_ms = (perf_counter() - started) * 1000.0
    kernel_event(logger, f"{event}.end", elapsed_ms=round(elapsed_ms, 3), **fields)
