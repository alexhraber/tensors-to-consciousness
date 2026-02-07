from __future__ import annotations

import os
import unittest
import builtins
from unittest.mock import patch

from tools import rust_core


class RustCoreLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        rust_core.load_rust_core.cache_clear()

    def test_disabled_env_skips_loading(self) -> None:
        with patch.dict(os.environ, {"TTC_DISABLE_RUST_CORE": "1"}, clear=False):
            rust_core.load_rust_core.cache_clear()
            self.assertIsNone(rust_core.load_rust_core())

    def test_missing_module_returns_none(self) -> None:
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
            if name == "ttc_rust_core":
                raise ModuleNotFoundError("no rust core")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            rust_core.load_rust_core.cache_clear()
            self.assertIsNone(rust_core.load_rust_core())


if __name__ == "__main__":
    unittest.main()
