from __future__ import annotations

import io
import importlib
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


def _load_common_viz():
    try:
        return importlib.import_module("tools.common_viz")
    except ModuleNotFoundError as exc:
        if exc.name == "numpy":
            raise unittest.SkipTest("numpy not installed in test interpreter")
        raise


class CommonVizTests(unittest.TestCase):
    def test_ascii_heatmap_shape(self) -> None:
        common_viz = _load_common_viz()
        np = common_viz.np
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        out = common_viz._ascii_heatmap(arr, width=8, height=4)
        lines = out.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(all(len(line) == 8 for line in lines))

    def test_ascii_heatmap_empty(self) -> None:
        common_viz = _load_common_viz()
        np = common_viz.np
        arr = np.array([], dtype=np.float32)
        self.assertEqual(common_viz._ascii_heatmap(arr), "(empty)")

    def test_viz_stage_filters_and_limit(self) -> None:
        common_viz = _load_common_viz()
        np = common_viz.np

        scope = {
            "_private": np.ones((20, 20), dtype=np.float32),
            "scalar": np.array(1.0, dtype=np.float32),
            "small": np.ones((2, 1), dtype=np.float32),  # size < 4 should be ignored
            "big": np.ones((20, 20), dtype=np.float32),
            "medium": np.ones((10, 10), dtype=np.float32),
            "other": np.ones((8, 8), dtype=np.float32),
        }

        def to_numpy(value):
            return value if hasattr(value, "shape") else None

        buf = io.StringIO()
        with redirect_stdout(buf):
            common_viz.viz_stage("stage_x", scope, to_numpy, framework="numpy", limit=2)
        out = buf.getvalue()
        self.assertIn("[VIS:numpy] stage_x", out)
        self.assertIn("- big:", out)
        self.assertIn("- medium:", out)
        self.assertNotIn("- other:", out)
        self.assertNotIn("- _private:", out)
        self.assertNotIn("- scalar:", out)
        self.assertNotIn("- small:", out)

    def test_viz_stage_respects_env_toggle(self) -> None:
        common_viz = _load_common_viz()
        np = common_viz.np
        scope = {"x": np.ones((4, 4), dtype=np.float32)}

        with patch.dict(os.environ, {"T2C_VIZ": "0"}):
            buf = io.StringIO()
            with redirect_stdout(buf):
                common_viz.viz_stage("stage_off", scope, lambda x: x, framework="numpy")
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
