from __future__ import annotations

import unittest


class NumpyFrameworkBackendTests(unittest.TestCase):
    def test_numpy_utils_normal_shape(self) -> None:
        try:
            from frameworks.numpy import utils as np_utils
        except ModuleNotFoundError:
            self.skipTest("numpy framework dependencies not installed in test interpreter")

        arr = np_utils.normal((6, 7))
        self.assertEqual(getattr(arr, "shape", None), (6, 7))

    def test_numpy_engine_pipeline_generates_tensor(self) -> None:
        try:
            from frameworks.engine import FrameworkEngine
        except ModuleNotFoundError:
            self.skipTest("framework dependencies not installed in test interpreter")

        try:
            runtime = FrameworkEngine("numpy")
        except ModuleNotFoundError:
            self.skipTest("numpy framework dependencies not installed in test interpreter")
        result = runtime.run_pipeline(("reaction_diffusion", "entropy_flow"), size=24, steps=1)
        arr = runtime.to_numpy(result.final_tensor)
        if arr is None:
            self.skipTest("numpy adapter conversion unavailable in this interpreter")
        self.assertEqual(tuple(arr.shape), (24, 24))
        self.assertEqual(len(result.trace), 2)


if __name__ == "__main__":
    unittest.main()
