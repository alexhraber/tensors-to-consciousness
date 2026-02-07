from __future__ import annotations

import base64
import json
import sys
from dataclasses import dataclass
from typing import Any

from transforms.registry import list_transform_keys, resolve_transform_keys


@dataclass
class Request:
    id: int
    method: str
    params: dict[str, Any] | None


def _write(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _ok(req_id: int, result: Any) -> None:
    _write({"id": req_id, "ok": True, "result": result})


def _err(req_id: int, msg: str) -> None:
    _write({"id": req_id, "ok": False, "error": msg})


def _as_numpy_2d(engine: Any, tensor: Any) -> Any:
    arr = engine.to_numpy(tensor)
    if arr is None:
        raise RuntimeError("engine returned non-numpy tensor")
    import numpy as np

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def handle_list_transforms(req: Request) -> None:
    _ok(req.id, list(list_transform_keys()))


def handle_run_pipeline(req: Request) -> None:
    from frameworks.engine import FrameworkEngine

    p = req.params or {}
    framework = str(p.get("framework") or "numpy")
    transforms = str(p.get("transforms") or "default")
    size = int(p.get("size") or 96)
    steps = int(p.get("steps") or 1)
    inputs = p.get("inputs")
    if inputs:
        # Not yet threaded into FrameworkEngine.run_pipeline; reserved for future parity with python surface.
        pass

    keys = resolve_transform_keys(transforms)
    engine = FrameworkEngine(framework)
    result = engine.run_pipeline(tuple(keys), size=size, steps=steps)
    arr = _as_numpy_2d(engine, result.final_tensor)
    raw = arr.tobytes(order="C")
    payload = {
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "dtype": "f32",
        "data_b64": base64.b64encode(raw).decode("ascii"),
    }
    _ok(req.id, payload)


def main() -> int:
    handlers = {
        "list_transforms": handle_list_transforms,
        "run_pipeline": handle_run_pipeline,
    }
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            req = Request(id=int(obj["id"]), method=str(obj["method"]), params=obj.get("params"))
            if req.method == "shutdown":
                _ok(req.id, True)
                return 0
            fn = handlers.get(req.method)
            if fn is None:
                _err(req.id, f"unknown method: {req.method}")
                continue
            fn(req)
        except Exception as exc:
            req_id = 0
            try:
                req_id = int(json.loads(line).get("id", 0))
            except Exception:
                req_id = 0
            _err(req_id, f"{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

