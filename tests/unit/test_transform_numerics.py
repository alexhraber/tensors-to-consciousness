from __future__ import annotations

import unittest

from frameworks.engine import FrameworkEngine
from transforms.registry import TRANSFORM_MAP


class TransformNumericsTests(unittest.TestCase):
    def test_numpy_backend_produces_finite_output_for_all_transforms(self) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("numpy not installed in test interpreter")

        try:
            engine = FrameworkEngine("numpy")
        except ModuleNotFoundError:
            self.skipTest("numpy framework dependencies not installed in test interpreter")

        for key in sorted(TRANSFORM_MAP.keys()):
            with self.subTest(transform=key):
                result = engine.run_pipeline((key,), size=24, steps=1)
                arr = engine.to_numpy(result.final_tensor)
                self.assertIsNotNone(arr)
                arr_np = np.asarray(arr)
                self.assertEqual(arr_np.shape, (24, 24))
                self.assertTrue(np.isfinite(arr_np).all())
                self.assertEqual(len(result.trace), 1)
                self.assertEqual(result.trace[0]["transform"], key)


if __name__ == "__main__":
    unittest.main()
