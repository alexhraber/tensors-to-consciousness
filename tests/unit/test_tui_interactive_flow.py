from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from tools import shinkei
from tools import tui


class _FakeStdin:
    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return 0


class _FakeStdout:
    def __init__(self) -> None:
        self.buffer: list[str] = []

    def isatty(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self.buffer.append(text)
        return len(text)

    def flush(self) -> None:
        return None


def _profiles() -> tuple[dict[str, object], ...]:
    preset_a = {
        "samples": 640,
        "freq": 1.2,
        "amplitude": 1.0,
        "damping": 0.1,
        "noise": 0.05,
        "phase": 0.3,
        "grid": 48,
    }
    preset_b = {
        "samples": 700,
        "freq": 1.4,
        "amplitude": 1.1,
        "damping": 0.11,
        "noise": 0.06,
        "phase": 0.4,
        "grid": 48,
    }
    return (
        {
            "id": "tensor_ops",
            "title": "Tensor Operations",
            "complexity": 0,
            "transforms": [
                {
                    "key": "tensor_ops",
                    "title": "Tensor Operations",
                    "formula": "C=A⊙B",
                    "description": "base",
                    "preset": preset_a,
                }
            ],
        },
        {
            "id": "chain_rule",
            "title": "Chain Rule",
            "complexity": 1,
            "transforms": [
                {
                    "key": "chain_rule",
                    "title": "Chain Rule",
                    "formula": "d/dx",
                    "description": "grad",
                    "preset": preset_b,
                }
            ],
        },
    )


class TuiInteractiveFlowTests(unittest.TestCase):
    def test_ctrl_c_returns_to_landing_then_quit(self) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("numpy not installed in test interpreter")

        landing_calls: list[tuple[str, str]] = []

        class _FakeEngine:
            def __init__(self, framework: str) -> None:
                self.framework = framework

            def run_pipeline(self, transform_keys: tuple[str, ...], *, size: int = 96, steps: int = 1):
                arr = np.ones((size, size), dtype=np.float32)
                return types.SimpleNamespace(framework=self.framework, final_tensor=arr, trace=[])

            def to_numpy(self, value):  # type: ignore[no-untyped-def]
                return value

        keys = iter(["\r", "\x03", "q"])

        def _read_char(_fd: int, timeout_s: float = 5.0) -> str:
            return next(keys, "q")

        with patch.object(tui, "SCRIPT_PROFILES", _profiles()):
            with patch.object(tui, "FrameworkEngine", _FakeEngine):
                with patch.object(tui, "_read_char", side_effect=_read_char):
                    with patch.object(tui.termios, "tcgetattr", return_value=[0, 0, 0, 0, 0, 0, 0]):
                        with patch.object(tui.termios, "tcsetattr"):
                            with patch.object(tui.tty, "setcbreak"):
                                with patch.object(tui.shinkei, "_supports_kitty_graphics", return_value=False):
                                    with patch.object(tui.shinkei, "_supports_graphical_terminal", return_value=False):
                                        with patch.object(tui.shinkei, "_ascii_heatmap", return_value="plot"):
                                            with patch.object(tui.shinkei, "_format_caption", side_effect=lambda x: x):
                                                with patch.object(tui, "_layout_for_view", return_value={"cols": 120, "rows": 40, "header_w": 96, "plot_w": 80, "plot_h": 20, "ascii_w": 80, "ascii_h": 20}):
                                                    original_landing_frame_text = tui._landing_frame_text

                                                    def _record_landing_frame(
                                                        framework: str,
                                                        platform: str,
                                                        width: int,
                                                        term_cols: int,
                                                        term_rows: int,
                                                    ) -> str:
                                                        landing_calls.append((framework, platform))
                                                        return original_landing_frame_text(
                                                            framework,
                                                            platform,
                                                            width,
                                                            term_cols,
                                                            term_rows,
                                                        )

                                                    with patch.object(tui, "_landing_frame_text", side_effect=_record_landing_frame):
                                                        with patch.object(tui.sys, "stdin", _FakeStdin()):
                                                            with patch.object(tui.sys, "stdout", _FakeStdout()):
                                                                rc = tui._render_interactive(
                                                                    np=np,
                                                                    state=shinkei.RenderState(view=shinkei.EXPLORER_VIEW),
                                                                    framework="numpy",
                                                                    initial_pipeline=("tensor_ops",),
                                                                    transform_selector=None,
                                                                )
        self.assertEqual(rc, 0)
        self.assertGreaterEqual(len(landing_calls), 2)

    def test_framework_switch_on_landing_handoffs(self) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("numpy not installed in test interpreter")

        keys = iter(["f"])

        def _read_char(_fd: int, timeout_s: float = 5.0) -> str:
            return next(keys, "q")

        with patch.object(tui, "SCRIPT_PROFILES", _profiles()):
            with patch.object(tui, "_read_char", side_effect=_read_char):
                with patch.object(tui, "_framework_selector", return_value="jax"):
                    with patch.object(tui, "_persist_framework") as persist_mock:
                        with patch.object(tui, "_handoff_framework_switch", return_value=7) as handoff_mock:
                            with patch.object(tui.termios, "tcgetattr", return_value=[0, 0, 0, 0, 0, 0, 0]):
                                with patch.object(tui.termios, "tcsetattr"):
                                    with patch.object(tui.tty, "setcbreak"):
                                        with patch.object(tui.sys, "stdin", _FakeStdin()):
                                            with patch.object(tui.sys, "stdout", _FakeStdout()):
                                                rc = tui._render_interactive(
                                                    np=np,
                                                    state=shinkei.RenderState(view=shinkei.EXPLORER_VIEW),
                                                    framework="numpy",
                                                    initial_pipeline=("tensor_ops",),
                                                    transform_selector=None,
                                                )
        self.assertEqual(rc, 7)
        persist_mock.assert_called_once_with("jax")
        handoff_mock.assert_called_once_with("jax")

    def test_navigation_and_toggle_update_pipeline_used_by_engine(self) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("numpy not installed in test interpreter")

        pipelines_seen: list[tuple[str, ...]] = []

        class _FakeEngine:
            def __init__(self, framework: str) -> None:
                self.framework = framework

            def run_pipeline(self, transform_keys: tuple[str, ...], *, size: int = 96, steps: int = 1):
                pipelines_seen.append(transform_keys)
                arr = np.ones((size, size), dtype=np.float32)
                return types.SimpleNamespace(framework=self.framework, final_tensor=arr, trace=[])

            def to_numpy(self, value):  # type: ignore[no-untyped-def]
                return value

        keys = iter(["\r", "n", "x", "q"])

        def _read_char(_fd: int, timeout_s: float = 5.0) -> str:
            return next(keys, "q")

        with patch.object(tui, "SCRIPT_PROFILES", _profiles()):
            with patch.object(tui, "FrameworkEngine", _FakeEngine):
                with patch.object(tui, "_read_char", side_effect=_read_char):
                    with patch.object(tui.termios, "tcgetattr", return_value=[0, 0, 0, 0, 0, 0, 0]):
                        with patch.object(tui.termios, "tcsetattr"):
                            with patch.object(tui.tty, "setcbreak"):
                                with patch.object(tui.shinkei, "_supports_kitty_graphics", return_value=False):
                                    with patch.object(tui.shinkei, "_supports_graphical_terminal", return_value=False):
                                        with patch.object(tui.shinkei, "_ascii_heatmap", return_value="plot"):
                                            with patch.object(tui.shinkei, "_format_caption", side_effect=lambda x: x):
                                                with patch.object(tui, "_layout_for_view", return_value={"cols": 120, "rows": 40, "header_w": 96, "plot_w": 80, "plot_h": 20, "ascii_w": 80, "ascii_h": 20}):
                                                    with patch.object(tui.sys, "stdin", _FakeStdin()):
                                                        with patch.object(tui.sys, "stdout", _FakeStdout()):
                                                            rc = tui._render_interactive(
                                                                np=np,
                                                                state=shinkei.RenderState(view=shinkei.EXPLORER_VIEW),
                                                                framework="numpy",
                                                                initial_pipeline=("tensor_ops",),
                                                                transform_selector=None,
                                                            )
        self.assertEqual(rc, 0)
        self.assertTrue(any(p == ("tensor_ops", "chain_rule") for p in pipelines_seen))


if __name__ == "__main__":
    unittest.main()
