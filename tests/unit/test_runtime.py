from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import runtime


class RuntimeTests(unittest.TestCase):
    def test_validate_script_for_framework(self) -> None:
        self.assertEqual(runtime.validate_script_for_framework("jax"), "frameworks/jax/test_setup.py")
        with self.assertRaises(RuntimeError):
            runtime.validate_script_for_framework("invalid")

    def test_load_config_missing_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "config.json"
            legacy_path = Path(td) / "legacy-config.json"
            with patch.object(runtime, "CONFIG_FILE", config_path):
                with patch.object(runtime, "LEGACY_CONFIG_FILE", legacy_path):
                    with self.assertRaises(RuntimeError):
                        runtime.load_config()

    def test_load_config_reads_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "config.json"
            config_path.write_text(json.dumps({"framework": "mlx", "venv": ".venv"}), encoding="utf-8")
            with patch.object(runtime, "CONFIG_FILE", config_path):
                self.assertEqual(runtime.load_config()["framework"], "mlx")

    def test_load_config_falls_back_to_legacy_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "config.json"
            legacy_path = Path(td) / "legacy-config.json"
            legacy_path.write_text(json.dumps({"framework": "jax", "venv": ".venv-jax"}), encoding="utf-8")
            with patch.object(runtime, "CONFIG_FILE", config_path):
                with patch.object(runtime, "LEGACY_CONFIG_FILE", legacy_path):
                    self.assertEqual(runtime.load_config()["framework"], "jax")

    def test_env_get_falls_back_to_legacy_prefix(self) -> None:
        with patch.dict(runtime.os.environ, {"T2C_INPUTS": "legacy"}, clear=True):
            self.assertEqual(runtime.env_get("TTC_INPUTS"), "legacy")


if __name__ == "__main__":
    unittest.main()
