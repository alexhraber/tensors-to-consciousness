from __future__ import annotations

import unittest
from unittest.mock import patch

from tools import tui


class TuiPipelineTests(unittest.TestCase):
    def test_handoff_framework_switch_starts_explorer_directly(self) -> None:
        with patch.object(tui.subprocess, "run") as run_mock:
            run_mock.return_value.returncode = 0
            rc = tui._handoff_framework_switch("jax")
        self.assertEqual(rc, 0)
        cmd = run_mock.call_args.args[0]
        self.assertEqual(
            cmd,
            [tui.sys.executable, "explorer.py", "render", "--framework", "jax", "--start-explorer"],
        )

    def test_toggle_pipeline_key_adds_and_removes(self) -> None:
        pipeline = ["tensor_ops", "chain_rule"]
        tui._toggle_pipeline_key(pipeline, "gradient_descent")
        self.assertEqual(pipeline, ["tensor_ops", "chain_rule", "gradient_descent"])

        tui._toggle_pipeline_key(pipeline, "chain_rule")
        self.assertEqual(pipeline, ["tensor_ops", "gradient_descent"])

    def test_move_pipeline_key_respects_precedence(self) -> None:
        pipeline = ["tensor_ops", "chain_rule", "gradient_descent"]

        tui._move_pipeline_key(pipeline, "gradient_descent", direction=-1)
        self.assertEqual(pipeline, ["tensor_ops", "gradient_descent", "chain_rule"])

        tui._move_pipeline_key(pipeline, "tensor_ops", direction=-1)
        self.assertEqual(pipeline, ["tensor_ops", "gradient_descent", "chain_rule"])

        tui._move_pipeline_key(pipeline, "tensor_ops", direction=1)
        self.assertEqual(pipeline, ["gradient_descent", "tensor_ops", "chain_rule"])

    def test_render_pipeline_selector_shows_checkbox_and_order(self) -> None:
        transforms = (
            {"key": "tensor_ops", "title": "Tensor Operations"},
            {"key": "chain_rule", "title": "Chain Rule Field"},
            {"key": "gradient_descent", "title": "Gradient Descent"},
        )
        pipeline = ["chain_rule", "tensor_ops"]

        lines = tui._render_pipeline_selector(transforms, pipeline, cursor_index=1, max_lines=8)

        self.assertTrue(any("[x] 02 tensor_ops" in line for line in lines))
        self.assertTrue(any("> [x] 01 chain_rule" in line for line in lines))
        self.assertTrue(any("[ ] -- gradient_descent" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
