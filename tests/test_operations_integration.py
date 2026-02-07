from __future__ import annotations

import argparse
import builtins
import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import t2c
from tools import setup
from tools import validate
from tools import viz_terminal


class ValidateEntrypointTests(unittest.TestCase):
    def test_validate_main_resolves_framework_script(self) -> None:
        args = argparse.Namespace(framework=None)
        with patch.object(validate, "parse_args", return_value=args):
            with patch.object(validate, "load_config", return_value={"framework": "jax"}):
                with patch.object(validate.runpy, "run_path") as run_path_mock:
                    rc = validate.main()
        self.assertEqual(rc, 0)
        run_path = run_path_mock.call_args[0][0]
        self.assertTrue(run_path.endswith("scripts/jax/test_setup.py"))


class SetupFlowIntegrationTests(unittest.TestCase):
    def test_setup_main_invokes_uv_and_installs_common_dep(self) -> None:
        args = argparse.Namespace(framework="mlx", venv=".venv-test", skip_validate=True)
        calls: list[list[str]] = []

        def fake_run_cmd(cmd: list[str], env=None) -> None:
            calls.append(cmd)

        with patch.object(setup, "parse_args", return_value=args):
            with patch.object(setup.shutil, "which", return_value="/usr/bin/uv"):
                with patch.object(setup, "run_cmd", side_effect=fake_run_cmd):
                    with patch.object(setup, "write_active_config") as write_config_mock:
                        rc = setup.main()
        self.assertEqual(rc, 0)
        self.assertEqual(calls[0][:2], ["uv", "venv"])
        self.assertIn("matplotlib", calls[1])
        self.assertIn("mlx", calls[1])
        write_config_mock.assert_called_once()


class VizTerminalTests(unittest.TestCase):
    def test_to_ascii_respects_dimensions(self) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("numpy not installed in test interpreter")

        rgba = np.zeros((20, 40, 4), dtype="uint8")
        out = viz_terminal._to_ascii(rgba, width=16, height=8)
        lines = out.splitlines()
        self.assertEqual(len(lines), 8)
        self.assertTrue(all(len(line) == 16 for line in lines))

    def test_main_without_matplotlib_returns_error(self) -> None:
        args = argparse.Namespace(framework="jax", width=80, height=24)
        stderr = io.StringIO()
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
            if name.startswith("matplotlib"):
                raise ModuleNotFoundError("mocked missing matplotlib")
            return real_import(name, globals, locals, fromlist, level)

        with patch.object(viz_terminal, "parse_args", return_value=args):
            with patch("builtins.__import__", side_effect=fake_import):
                with redirect_stderr(stderr):
                    rc = viz_terminal.main()
        self.assertEqual(rc, 1)
        self.assertIn("matplotlib is not installed", stderr.getvalue())


class RuntimeExecutionFlowTests(unittest.TestCase):
    def test_t2c_main_executes_module_zero(self) -> None:
        args = argparse.Namespace(
            target="0",
            framework=None,
            venv=None,
            no_setup=True,
        )
        with patch.object(t2c, "parse_args", return_value=args):
            with patch.object(
                t2c, "ensure_setup_if_needed", return_value=({"framework": "numpy", "venv": ".venv-np"}, False)
            ):
                with patch.object(t2c, "python_in_venv", return_value=Path(".venv-np/bin/python")):
                    with patch.object(t2c, "run_cmd") as run_cmd_mock:
                        rc = t2c.main()
        self.assertEqual(rc, 0)
        cmd = run_cmd_mock.call_args[0][0]
        self.assertEqual(cmd, [".venv-np/bin/python", "scripts/numpy/0_computational_primitives.py"])


if __name__ == "__main__":
    unittest.main()
