from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import subprocess

import explorer as app


class AppFlowIntegrationTests(unittest.TestCase):
    def test_list_transforms_forwards_to_rust(self) -> None:
        with patch.object(subprocess, "call", return_value=0) as call_mock:
            rc = app.main(["--list-transforms"])
        self.assertEqual(rc, 0)
        cmd = call_mock.call_args[0][0]
        self.assertGreaterEqual(len(cmd), 2)
        self.assertEqual(cmd[1], "list-transforms")

    def test_default_opens_tui(self) -> None:
        with patch.object(subprocess, "call", return_value=0) as call_mock:
            rc = app.main([])
        self.assertEqual(rc, 0)
        cmd = call_mock.call_args[0][0]
        self.assertEqual(cmd[1], "tui")

    def test_validate_target(self) -> None:
        with patch.object(subprocess, "call", return_value=0) as call_mock:
            rc = app.main(["validate", "--framework", "jax", "--venv", ".venv-jax"])
        self.assertEqual(rc, 0)
        cmd = call_mock.call_args[0][0]
        self.assertEqual(cmd[1], "validate")
        self.assertIn("--framework", cmd)
        self.assertIn("jax", cmd)
        self.assertIn("--venv", cmd)
        self.assertIn(".venv-jax", cmd)

    def test_render_target_maps_to_tui(self) -> None:
        with patch.object(subprocess, "call", return_value=0) as call_mock:
            rc = app.main(["render", "--framework", "jax", "--transforms", "default"])
        self.assertEqual(rc, 0)
        cmd = call_mock.call_args[0][0]
        self.assertEqual(cmd[1], "tui")
        self.assertIn("--framework", cmd)
        self.assertIn("jax", cmd)
        self.assertIn("--transforms", cmd)
        self.assertIn("default", cmd)

    def test_run_target_reads_inputs_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "inputs.json"
            p.write_text(json.dumps({"seed": 123}), encoding="utf-8")
            with patch.object(subprocess, "call", return_value=0) as call_mock:
                rc = app.main(["run", "--framework", "jax", "--inputs", str(p), "--transforms", "default"])
        self.assertEqual(rc, 0)
        cmd = call_mock.call_args[0][0]
        self.assertEqual(cmd[1], "run")
        # `--inputs` becomes an inline JSON string for the Rust CLI.
        self.assertIn("--inputs", cmd)
        idx = cmd.index("--inputs")
        self.assertEqual(json.loads(cmd[idx + 1]), {"seed": 123})

    def test_run_defaults_framework_from_config_or_platform(self) -> None:
        with patch.object(app, "_read_active_framework_from_config", return_value="jax"):
            with patch.object(subprocess, "call", return_value=0) as call_mock:
                rc = app.main(["run", "--transforms", "default"])
        self.assertEqual(rc, 0)
        cmd = call_mock.call_args[0][0]
        self.assertEqual(cmd[1], "run")
        self.assertIn("--framework", cmd)
        self.assertIn("jax", cmd)

    def test_invalid_target(self) -> None:
        with patch.object(subprocess, "call", return_value=0) as call_mock:
            rc = app.main(["bad"])
        self.assertEqual(rc, 1)
        call_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()

