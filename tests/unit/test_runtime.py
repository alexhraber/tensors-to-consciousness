from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import runtime


class RuntimeTests(unittest.TestCase):
    def test_default_framework_for_platform(self) -> None:
        self.assertEqual(runtime.default_framework_for_platform("darwin"), "mlx")
        self.assertEqual(runtime.default_framework_for_platform("linux"), "numpy")

    def test_validate_script_for_framework(self) -> None:
        self.assertEqual(runtime.validate_script_for_framework("jax"), "frameworks/jax/test_setup.py")
        with self.assertRaises(RuntimeError):
            runtime.validate_script_for_framework("invalid")

    def test_load_config_missing_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "config.json"
            with patch.object(runtime, "CONFIG_FILE", config_path):
                with self.assertRaises(RuntimeError):
                    runtime.load_config()

    def test_load_config_reads_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "config.json"
            config_path.write_text(json.dumps({"framework": "mlx", "venv": ".venv"}), encoding="utf-8")
            with patch.object(runtime, "CONFIG_FILE", config_path):
                self.assertEqual(runtime.load_config()["framework"], "mlx")

    def test_load_config_optional_includes_sensible_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "config.json"
            config_path.write_text(json.dumps({"framework": "jax", "venv": ".venv-jax"}), encoding="utf-8")
            with patch.object(runtime, "CONFIG_FILE", config_path):
                cfg = runtime.load_config_optional()
        self.assertEqual(cfg["framework"], "jax")
        self.assertEqual(cfg["venv"], ".venv-jax")
        self.assertEqual(cfg["platform"], "gpu")
        self.assertIsInstance(cfg["diagnostics"], dict)
        self.assertEqual(cfg["diagnostics"]["log_level"], "INFO")
        self.assertFalse(cfg["diagnostics"]["debug"])

if __name__ == "__main__":
    unittest.main()
