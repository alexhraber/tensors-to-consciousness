from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


CATALOG_FILE = Path(__file__).resolve().parent / "transforms.json"


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    data = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Transform catalog must be a JSON object.")
    # Backward-compatible key support: prefer `transforms`, accept `algorithms`.
    if "transforms" in data:
        items = data["transforms"]
    else:
        items = data.get("algorithms")
    if not isinstance(items, list):
        raise ValueError("Transform catalog must define a 'transforms' array.")
    data["transforms"] = items
    default = data.get("default", [])
    if not isinstance(default, list):
        raise ValueError("Transform catalog 'default' must be a list.")
    return data


def catalog_transforms() -> tuple[dict[str, Any], ...]:
    return tuple(load_catalog()["transforms"])


def catalog_default_keys() -> tuple[str, ...]:
    return tuple(str(k) for k in load_catalog().get("default", ()))


def catalog_framework_interface() -> dict[str, tuple[str, ...]]:
    raw = load_catalog().get("framework_interface", {})
    if not isinstance(raw, dict):
        raise ValueError("Transform catalog 'framework_interface' must be an object.")
    utils = raw.get("utils_entrypoints", ())
    ops = raw.get("ops_adapter", ())
    if not isinstance(utils, list) or not isinstance(ops, list):
        raise ValueError("Transform catalog framework interface lists are invalid.")
    return {
        "utils_entrypoints": tuple(str(v) for v in utils),
        "ops_adapter": tuple(str(v) for v in ops),
    }


# Backward compatibility alias.
def catalog_algorithms() -> tuple[dict[str, Any], ...]:
    return catalog_transforms()
