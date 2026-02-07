from __future__ import annotations

import types
import unittest
from unittest.mock import patch


class EngineContractTests(unittest.TestCase):
    def test_engine_executes_ordered_discovered_algorithms(self) -> None:
        from frameworks import engine as eng

        class _FakeTensor:
            def __init__(self, value: float = 1.0, shape: tuple[int, int] = (4, 4)) -> None:
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

        with patch.object(eng.importlib, "import_module", return_value=fake_utils):
            runtime = eng.FrameworkEngine("numpy")
            result = runtime.run_pipeline(("momentum", "adam", "chain_rule"), size=4, steps=1)

        self.assertEqual(result.framework, "numpy")
        self.assertEqual(result.algo_keys, ("momentum", "adam", "chain_rule"))
        self.assertEqual([step["algo"] for step in result.trace], ["momentum", "adam", "chain_rule"])
        self.assertTrue(hasattr(result.final_tensor, "shape"))


if __name__ == "__main__":
    unittest.main()
