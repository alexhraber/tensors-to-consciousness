from __future__ import annotations

import os
import unittest
import builtins
from unittest.mock import patch

from tools import accel


class AccelLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        accel.load_accel.cache_clear()

    def test_disabled_env_skips_loading(self) -> None:
        with patch.dict(os.environ, {"EXPLORER_DISABLE_ACCEL": "1"}, clear=False):
            accel.load_accel.cache_clear()
            self.assertIsNone(accel.load_accel())

    def test_missing_module_returns_none(self) -> None:
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
            if name == "explorer_accel":
                raise ModuleNotFoundError("no rust core")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            accel.load_accel.cache_clear()
            self.assertIsNone(accel.load_accel())

    def test_parse_assignment_wrapper(self) -> None:
        class _FakeCore:
            @staticmethod
            def parse_assignment(expr: str):
                return ("seed", "12") if "=" in expr else None

        with patch.object(accel, "load_accel", return_value=_FakeCore()):
            self.assertEqual(accel.parse_assignment("seed=12"), ("seed", "12"))
            self.assertIsNone(accel.parse_assignment("seed"))


if __name__ == "__main__":
    unittest.main()
