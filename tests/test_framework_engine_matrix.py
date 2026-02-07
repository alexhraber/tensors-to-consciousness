from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from tools.runtime import SUPPORTED_FRAMEWORKS


class FrameworkEngineMatrixTests(unittest.TestCase):
    def test_engine_pipeline_executes_for_all_framework_labels(self) -> None:
        from frameworks import engine as eng

        class _FakeTensor:
            def __init__(self, value: float = 1.0, shape: tuple[int, int] = (8, 8)) -> None:
                self.value = value
                self.shape = shape

            @property
            def T(self) -> "_FakeTensor":
                return _FakeTensor(self.value, (self.shape[1], self.shape[0]))

            def __add__(self, other):  # type: ignore[no-untyped-def]
                ov = other.value if isinstance(other, _FakeTensor) else float(other)
                return _FakeTensor(self.value + ov, self.shape)

            def __sub__(self, other):  # type: ignore[no-untyped-def]
                ov = other.value if isinstance(other, _FakeTensor) else float(other)
                return _FakeTensor(self.value - ov, self.shape)

            def __mul__(self, other):  # type: ignore[no-untyped-def]
                ov = other.value if isinstance(other, _FakeTensor) else float(other)
                return _FakeTensor(self.value * ov, self.shape)

            __rmul__ = __mul__

            def __matmul__(self, other):  # type: ignore[no-untyped-def]
                ov = other.value if isinstance(other, _FakeTensor) else float(other)
                return _FakeTensor(self.value * ov, self.shape)

        fake_utils = types.SimpleNamespace(
            normal=lambda shape: _FakeTensor(1.0, shape),
            _to_numpy=lambda value: None,
        )

        def _import_module(name: str):  # type: ignore[no-untyped-def]
            self.assertTrue(name.startswith("frameworks."))
            self.assertTrue(name.endswith(".utils"))
            return fake_utils

        with patch.object(eng.importlib, "import_module", side_effect=_import_module):
            for framework in SUPPORTED_FRAMEWORKS:
                with self.subTest(framework=framework):
                    runtime = eng.FrameworkEngine(framework)
                    result = runtime.run_pipeline(
                        ("laplacian_diffusion", "momentum", "constraint_projection"),
                        size=12,
                        steps=2,
                    )
                    self.assertEqual(result.framework, framework)
                    self.assertEqual(result.transform_keys, ("laplacian_diffusion", "momentum", "constraint_projection"))
                    self.assertEqual(len(result.trace), 6)
                    self.assertTrue(hasattr(result.final_tensor, "shape"))


if __name__ == "__main__":
    unittest.main()
