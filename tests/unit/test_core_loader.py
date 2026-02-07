from __future__ import annotations

import os
import unittest
import builtins
from unittest.mock import patch

from tools import core


class CoreLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        core.load_core.cache_clear()

    def test_disabled_env_skips_loading(self) -> None:
        with patch.dict(os.environ, {"EXPLORER_DISABLE_CORE": "1"}, clear=False):
            core.load_core.cache_clear()
            self.assertIsNone(core.load_core())

    def test_missing_module_returns_none(self) -> None:
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
            if name == "core":
                raise ModuleNotFoundError("no core module")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            core.load_core.cache_clear()
            self.assertIsNone(core.load_core())

    def test_parse_assignment_wrapper(self) -> None:
        class _FakeCore:
            @staticmethod
            def parse_assignment(expr: str):
                return ("seed", "12") if "=" in expr else None

        with patch.object(core, "load_core", return_value=_FakeCore()):
            self.assertEqual(core.parse_assignment("seed=12"), ("seed", "12"))
            self.assertIsNone(core.parse_assignment("seed"))


if __name__ == "__main__":
    unittest.main()
