from __future__ import annotations

import unittest

from frameworks.engine import FrameworkEngine
from tools.runtime import SUPPORTED_FRAMEWORKS


class FrameworkBackendsRealTests(unittest.TestCase):
    def test_installed_framework_backends_execute_pipeline(self) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("numpy not installed in test interpreter")

        executed = 0
        for framework in SUPPORTED_FRAMEWORKS:
            with self.subTest(framework=framework):
                try:
                    engine = FrameworkEngine(framework)
                except ModuleNotFoundError:
                    continue

                result = engine.run_pipeline(("tensor_ops", "chain_rule", "gradient_descent"), size=24, steps=1)
                arr = engine.to_numpy(result.final_tensor)
                self.assertIsNotNone(arr)
                arr_np = np.asarray(arr)
                self.assertEqual(arr_np.shape, (24, 24))
                self.assertTrue(np.isfinite(arr_np).all())
                self.assertEqual(result.framework, framework)
                self.assertEqual(len(result.trace), 3)
                executed += 1
        self.assertGreaterEqual(executed, 1)


if __name__ == "__main__":
    unittest.main()
