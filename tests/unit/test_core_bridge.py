from __future__ import annotations

import importlib
import unittest
from unittest.mock import patch


class CoreBridgeTests(unittest.TestCase):
    def _np(self):
        shinkei = importlib.import_module("tools.shinkei")
        try:
            np = shinkei._np_module()
        except ModuleNotFoundError as exc:
            if exc.name == "numpy":
                self.skipTest("numpy not installed in test interpreter")
            raise
        return shinkei, np

    def test_ascii_heatmap_prefers_accel_when_available(self) -> None:
        shinkei, np = self._np()
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)

        with patch.object(shinkei.core, "ascii_heatmap", return_value="RUST_ASCII") as rust_mock:
            out = shinkei._ascii_heatmap(arr, width=8, height=4)

        self.assertEqual(out, "RUST_ASCII")
        rust_mock.assert_called_once()

    def test_pixel_heatmap_prefers_accel_when_available(self) -> None:
        shinkei, np = self._np()
        arr = np.ones((8, 8), dtype=np.float32)

        with patch.object(shinkei.core, "pixel_heatmap", return_value="RUST_PIXEL") as rust_mock:
            out = shinkei._pixel_heatmap(arr, width=8, height=4)

        self.assertEqual(out, "RUST_PIXEL")
        rust_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
