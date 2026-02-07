from __future__ import annotations

import unittest
from unittest.mock import patch

from tools import rust_core


class RustCoreWrapperTests(unittest.TestCase):
    def test_ascii_and_pixel_wrapper_calls_core(self) -> None:
        class _FakeCore:
            @staticmethod
            def ascii_heatmap(arr, width, height):  # type: ignore[no-untyped-def]
                return f"a:{width}x{height}"

            @staticmethod
            def pixel_heatmap(arr, width, height):  # type: ignore[no-untyped-def]
                return f"p:{width}x{height}"

        with patch.object(rust_core, "load_rust_core", return_value=_FakeCore()):
            self.assertEqual(rust_core.ascii_heatmap([[1]], width=8, height=4), "a:8x4")
            self.assertEqual(rust_core.pixel_heatmap([[1]], width=10, height=6), "p:10x6")

    def test_ascii_and_pixel_wrapper_fail_closed(self) -> None:
        class _FakeCore:
            @staticmethod
            def ascii_heatmap(arr, width, height):  # type: ignore[no-untyped-def]
                raise RuntimeError("boom")

            @staticmethod
            def pixel_heatmap(arr, width, height):  # type: ignore[no-untyped-def]
                raise RuntimeError("boom")

        with patch.object(rust_core, "load_rust_core", return_value=_FakeCore()):
            self.assertIsNone(rust_core.ascii_heatmap([[1]], width=8, height=4))
            self.assertIsNone(rust_core.pixel_heatmap([[1]], width=10, height=6))

    def test_default_venv_and_normalize_platform_wrappers(self) -> None:
        class _FakeCore:
            @staticmethod
            def default_venv(framework: str) -> str:
                return f".venv-{framework}"

            @staticmethod
            def normalize_platform(value: str | None, default: str) -> str:
                return default if value is None else value

        with patch.object(rust_core, "load_rust_core", return_value=_FakeCore()):
            self.assertEqual(rust_core.default_venv("jax"), ".venv-jax")
            self.assertEqual(rust_core.normalize_platform(None, default="gpu"), "gpu")
            self.assertEqual(rust_core.normalize_platform("cpu", default="gpu"), "cpu")

    def test_default_venv_and_normalize_platform_type_guard(self) -> None:
        class _FakeCore:
            @staticmethod
            def default_venv(framework: str):  # type: ignore[no-untyped-def]
                return 123

            @staticmethod
            def normalize_platform(value: str | None, default: str):  # type: ignore[no-untyped-def]
                return 123

        with patch.object(rust_core, "load_rust_core", return_value=_FakeCore()):
            self.assertIsNone(rust_core.default_venv("jax"))
            self.assertIsNone(rust_core.normalize_platform("cpu", default="gpu"))

    def test_frame_patch_wrapper_and_fail_closed(self) -> None:
        class _FakeCore:
            @staticmethod
            def frame_patch(prev: str, next_: str) -> str:
                return "PATCH"

        with patch.object(rust_core, "load_rust_core", return_value=_FakeCore()):
            self.assertEqual(rust_core.frame_patch("a", "b"), "PATCH")

        class _FailCore:
            @staticmethod
            def frame_patch(prev: str, next_: str) -> str:
                raise RuntimeError("boom")

        with patch.object(rust_core, "load_rust_core", return_value=_FailCore()):
            self.assertIsNone(rust_core.frame_patch("a", "b"))

    def test_parse_assignment_requires_str_tuple(self) -> None:
        class _FakeCore:
            @staticmethod
            def parse_assignment(expr: str):  # type: ignore[no-untyped-def]
                return ("seed", 1)

        with patch.object(rust_core, "load_rust_core", return_value=_FakeCore()):
            self.assertIsNone(rust_core.parse_assignment("seed=1"))


if __name__ == "__main__":
    unittest.main()
