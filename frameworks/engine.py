from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from algos.catalog import catalog_framework_interface
from algos.contracts import TensorField
from algos.definitions import get_transform_definition
from algos.registry import TRANSFORM_MAP


@dataclass
class PipelineResult:
    framework: str
    transform_keys: tuple[str, ...]
    final_tensor: Any
    trace: list[dict[str, object]]

    @property
    def algo_keys(self) -> tuple[str, ...]:
        # Compatibility alias for older call sites.
        return self.transform_keys


def _to_numpy_fallback(value: Any) -> Any | None:
    try:
        import numpy as np
    except ModuleNotFoundError:
        return None

    try:
        arr = np.asarray(value)
        if arr.dtype == np.dtype("O"):
            return None
        return arr
    except Exception:
        return None


class FrameworkEngine:
    def __init__(self, framework: str) -> None:
        self.framework = framework
        self.utils: ModuleType = importlib.import_module(f"frameworks.{framework}.utils")
        self._interface = catalog_framework_interface()
        self._validate_framework_interface()

    def _validate_framework_interface(self) -> None:
        for name in self._interface["utils_entrypoints"]:
            fn = getattr(self.utils, name, None)
            if not callable(fn):
                raise RuntimeError(
                    f"Framework '{self.framework}' missing required utils entrypoint '{name}'."
                )

    def to_numpy(self, value: Any) -> Any | None:
        fn = getattr(self.utils, "_to_numpy", None)
        if callable(fn):
            arr = fn(value)
            if arr is not None:
                try:
                    import numpy as np

                    return np.asarray(arr)
                except ModuleNotFoundError:
                    return arr
        return _to_numpy_fallback(value)

    def _normal(self, shape: tuple[int, int]) -> Any:
        return self.utils.normal(shape)

    class _Ops:
        def __init__(self, engine: "FrameworkEngine") -> None:
            self.engine = engine

        @staticmethod
        def add(a: Any, b: Any) -> Any:
            return a + b

        @staticmethod
        def sub(a: Any, b: Any) -> Any:
            return a - b

        @staticmethod
        def mul(a: Any, b: Any) -> Any:
            return a * b

        @staticmethod
        def matmul(a: Any, b: Any) -> Any:
            return a @ b

        @staticmethod
        def transpose(a: Any) -> Any:
            return a.T

        @staticmethod
        def zeros_like(a: Any) -> Any:
            return a * 0.0

        def normal_like(self, a: Any) -> Any:
            return self.engine._normal(tuple(a.shape))

    def _validate_ops_adapter(self, ops: "FrameworkEngine._Ops") -> None:
        for name in self._interface["ops_adapter"]:
            fn = getattr(ops, name, None)
            if not callable(fn):
                raise RuntimeError(f"Ops adapter missing required op '{name}'.")

    def _params_for(self, key: str) -> dict[str, float]:
        spec = TRANSFORM_MAP[key]
        p = spec.preset
        return {
            "alpha": 0.0015 * float(p.freq),
            "beta": 1.0 - min(0.95, float(p.damping) * 0.12),
            "gamma": float(p.noise) * 0.08,
        }

    def run_pipeline(self, algo_keys: tuple[str, ...], *, size: int = 96, steps: int = 1) -> PipelineResult:
        n = max(24, min(320, int(size)))
        field = TensorField(tensor=self._normal((n, n)))
        ops = self._Ops(self)
        self._validate_ops_adapter(ops)
        trace: list[dict[str, object]] = []

        for _ in range(max(1, int(steps))):
            for key in algo_keys:
                definition = get_transform_definition(key)
                params = {**definition.defaults, **self._params_for(key)}
                field = definition.transform(field, ops, params)
                arr = self.to_numpy(field.tensor)
                if arr is not None:
                    trace.append(
                        {
                            "transform": key,
                            "algo": key,
                            "shape": tuple(arr.shape),
                            "mean": float(arr.mean()),
                            "std": float(arr.std()),
                        }
                    )
                else:
                    trace.append(
                        {
                            "transform": key,
                            "algo": key,
                            "shape": tuple(getattr(field.tensor, "shape", ())),
                            "mean": 0.0,
                            "std": 0.0,
                        }
                    )

        return PipelineResult(
            framework=self.framework,
            transform_keys=algo_keys,
            final_tensor=field.tensor,
            trace=trace,
        )
