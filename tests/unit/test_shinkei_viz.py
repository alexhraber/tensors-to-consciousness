from __future__ import annotations

import io
import importlib
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


def _load_shinkei_viz():
    try:
        module = importlib.import_module("tools.shinkei")
        module._np_module()
        return module
    except ModuleNotFoundError as exc:
        if exc.name == "numpy":
            raise unittest.SkipTest("numpy not installed in test interpreter")
        raise


class ShinkeiVizTests(unittest.TestCase):
    def test_ascii_heatmap_shape(self) -> None:
        shinkei_viz = _load_shinkei_viz()
        np = shinkei_viz._np_module()
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        out = shinkei_viz._ascii_heatmap(arr, width=8, height=4)
        lines = out.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(all(len(line) == 8 for line in lines))

    def test_ascii_heatmap_empty(self) -> None:
        shinkei_viz = _load_shinkei_viz()
        np = shinkei_viz._np_module()
        arr = np.array([], dtype=np.float32)
        self.assertEqual(shinkei_viz._ascii_heatmap(arr), "(empty)")

    def test_half_block_render_constant_tensor_still_visible(self) -> None:
        shinkei_viz = _load_shinkei_viz()
        np = shinkei_viz._np_module()
        arr = np.ones((8, 8), dtype=np.float32)
        out = shinkei_viz._pixel_heatmap(arr, width=8, height=4)
        self.assertTrue(any(ch.strip() for ch in out.splitlines()))

    def test_viz_stage_filters_and_limit(self) -> None:
        shinkei_viz = _load_shinkei_viz()
        np = shinkei_viz._np_module()

        scope = {
            "_private": np.ones((20, 20), dtype=np.float32),
            "scalar": np.array(1.0, dtype=np.float32),
            "small": np.ones((2, 1), dtype=np.float32),
            "big": np.ones((20, 20), dtype=np.float32),
            "medium": np.ones((10, 10), dtype=np.float32),
            "other": np.ones((8, 8), dtype=np.float32),
        }

        def to_numpy(value):
            return value if hasattr(value, "shape") else None

        buf = io.StringIO()
        with redirect_stdout(buf):
            shinkei_viz.viz_stage("stage_x", scope, to_numpy, framework="numpy", limit=2)
        out = buf.getvalue()
        self.assertIn("[VIS:numpy] stage_x", out)
        self.assertIn("- big:", out)
        self.assertIn("- medium:", out)
        self.assertNotIn("- other:", out)
        self.assertNotIn("- _private:", out)
        self.assertNotIn("- scalar:", out)
        self.assertNotIn("- small:", out)

    def test_viz_stage_respects_env_toggle(self) -> None:
        shinkei_viz = _load_shinkei_viz()
        np = shinkei_viz._np_module()
        scope = {"x": np.ones((4, 4), dtype=np.float32)}

        with patch.dict(os.environ, {"T2C_VIZ": "0"}):
            buf = io.StringIO()
            with redirect_stdout(buf):
                shinkei_viz.viz_stage("stage_off", scope, lambda x: x, framework="numpy")
        self.assertEqual(buf.getvalue(), "")

    def test_viz_stage_renders_metadata_line(self) -> None:
        shinkei_viz = _load_shinkei_viz()
        np = shinkei_viz._np_module()
        scope = {"x": np.arange(16, dtype=np.float32).reshape(4, 4)}
        buf = io.StringIO()
        with patch.object(shinkei_viz, "_supports_graphical_terminal", return_value=False):
            with redirect_stdout(buf):
                shinkei_viz.viz_stage(
                    "stage_meta",
                    scope,
                    lambda x: x,
                    framework="numpy",
                    metadata={"x": "Generated via test metadata."},
                )
        self.assertIn("Generated via test metadata.", buf.getvalue())

    def test_viz_stage_plots_falls_back_to_heatmap(self) -> None:
        shinkei_viz = _load_shinkei_viz()
        np = shinkei_viz._np_module()
        scope = {"x": np.arange(64, dtype=np.float32).reshape(8, 8)}

        with patch.dict(
            os.environ,
            {"T2C_VIZ_STYLE": "plots", "T2C_VIZ_TRACE": "1"},
            clear=False,
        ), patch.object(shinkei_viz, "_supports_graphical_terminal", return_value=True), patch.object(
            shinkei_viz, "_supports_kitty_graphics", return_value=False
        ):
            buf = io.StringIO()
            with redirect_stdout(buf):
                shinkei_viz.viz_stage("stage_fallback", scope, lambda x: x, framework="numpy")
        self.assertIn("[VIS renderer=heatmap]", buf.getvalue())

    def test_viz_stage_default_prefers_plots_when_inline_available(self) -> None:
        shinkei_viz = _load_shinkei_viz()
        np = shinkei_viz._np_module()
        scope = {"x": np.arange(64, dtype=np.float32).reshape(8, 8)}

        with patch.dict(os.environ, {"T2C_VIZ_TRACE": "1"}, clear=False), patch.object(
            shinkei_viz, "_supports_inline_image_graphics", return_value=True
        ), patch.object(shinkei_viz, "_supports_kitty_graphics", return_value=True), patch.object(
            shinkei_viz, "_supports_graphical_terminal", return_value=True
        ):
            buf = io.StringIO()
            with redirect_stdout(buf):
                shinkei_viz.viz_stage("stage_fluid", scope, lambda x: x, framework="numpy")
        self.assertIn("[VIS renderer=plots]", buf.getvalue())

    def test_inline_support_requires_capability_or_force(self) -> None:
        shinkei_viz = _load_shinkei_viz()
        with patch.dict(
            os.environ,
            {"COLORTERM": "truecolor", "TERM": "xterm-256color"},
            clear=False,
        ), patch.object(shinkei_viz.sys.stdout, "isatty", return_value=True):
            self.assertFalse(shinkei_viz._supports_inline_image_graphics())

        with patch.dict(
            os.environ,
            {"T2C_VIZ_FORCE_INLINE": "1"},
            clear=False,
        ), patch.object(shinkei_viz.sys.stdout, "isatty", return_value=True):
            self.assertTrue(shinkei_viz._supports_inline_image_graphics())


if __name__ == "__main__":
    unittest.main()
