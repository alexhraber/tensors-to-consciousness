from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

from tools import diagnostics


class DiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        diagnostics._CONFIGURED = False
        diagnostics._DEBUG_ENABLED = False

    def test_configure_logging_defaults_to_info_and_debug_off(self) -> None:
        with patch.object(diagnostics.runtime, "load_config_optional", return_value={}):
            with patch.dict("os.environ", {}, clear=True):
                cfg = diagnostics.configure_logging()
        self.assertEqual(cfg.level_name, "INFO")
        self.assertFalse(cfg.debug_enabled)
        self.assertFalse(diagnostics.debug_enabled())

    def test_configure_logging_debug_enables_debug_level(self) -> None:
        with patch.object(diagnostics.runtime, "load_config_optional", return_value={}):
            with patch.dict("os.environ", {"DEBUG": "1"}, clear=True):
                cfg = diagnostics.configure_logging()
        self.assertEqual(cfg.level_name, "DEBUG")
        self.assertTrue(cfg.debug_enabled)
        self.assertTrue(diagnostics.debug_enabled())

    def test_configure_logging_uses_config_json_defaults(self) -> None:
        config = {"diagnostics": {"log_level": "WARNING", "debug": True}}
        with patch.object(diagnostics.runtime, "load_config_optional", return_value=config):
            with patch.dict("os.environ", {}, clear=True):
                cfg = diagnostics.configure_logging()
        self.assertEqual(cfg.level_name, "DEBUG")
        self.assertTrue(cfg.debug_enabled)

    def test_env_overrides_config_json_diagnostics(self) -> None:
        config = {"diagnostics": {"log_level": "WARNING", "debug": False}}
        with patch.object(diagnostics.runtime, "load_config_optional", return_value=config):
            with patch.dict("os.environ", {"LOG_LEVEL": "ERROR", "DEBUG": "0"}, clear=True):
                cfg = diagnostics.configure_logging()
        self.assertEqual(cfg.level_name, "ERROR")
        self.assertFalse(cfg.debug_enabled)

    def test_kernel_event_noop_when_debug_off(self) -> None:
        logger = logging.getLogger("tests.diagnostics")
        with patch.object(diagnostics.runtime, "load_config_optional", return_value={}):
            with patch.dict("os.environ", {}, clear=True):
                diagnostics.configure_logging()
                diagnostics.kernel_event(logger, "event.test", key="value")


if __name__ == "__main__":
    unittest.main()
