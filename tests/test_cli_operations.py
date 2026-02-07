from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main as t2c
from tools import runtime
from tools import setup


class RuntimeTests(unittest.TestCase):
    def test_validate_script_for_framework(self) -> None:
        self.assertEqual(runtime.validate_script_for_framework("jax"), "scripts/jax/test_setup.py")
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


class SetupScriptTests(unittest.TestCase):
    def test_setup_one_calls_install_and_validate(self) -> None:
        calls: list[list[str]] = []

        def fake_run_cmd(cmd: list[str], env=None) -> None:
            calls.append(cmd)

        with patch.object(setup, "run_cmd", side_effect=fake_run_cmd):
            setup.setup_one("jax", Path(".venv"), skip_validate=False)

        self.assertEqual(calls[0][:4], ["uv", "pip", "install", "--python"])
        self.assertIn("jax[cpu]", calls[0])
        self.assertIn("matplotlib", calls[0])
        self.assertEqual(calls[1][-2:], ["--framework", "jax"])

    def test_setup_one_skip_validate(self) -> None:
        calls: list[list[str]] = []

        def fake_run_cmd(cmd: list[str], env=None) -> None:
            calls.append(cmd)

        with patch.object(setup, "run_cmd", side_effect=fake_run_cmd):
            setup.setup_one("numpy", Path(".venv"), skip_validate=True)

        self.assertEqual(len(calls), 1)
        self.assertIn("numpy", calls[0])

    def test_setup_main_requires_uv(self) -> None:
        args = argparse.Namespace(framework="mlx", venv=".venv", skip_validate=False)
        with patch.object(setup, "parse_args", return_value=args):
            with patch.object(setup.shutil, "which", return_value=None):
                rc = setup.main()
        self.assertEqual(rc, 1)


class T2CTests(unittest.TestCase):
    def test_ensure_setup_not_needed(self) -> None:
        env: dict[str, str] = {}
        with tempfile.TemporaryDirectory() as td:
            venv_dir = Path(td) / ".venv"
            py = runtime.python_in_venv(venv_dir)
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("", encoding="utf-8")
            with patch.object(t2c, "load_config", return_value={"framework": "mlx", "venv": str(venv_dir)}):
                with patch.object(t2c, "run_cmd") as run_cmd_mock:
                    result = t2c.ensure_setup_if_needed(
                        framework=None,
                        venv=None,
                        framework_overridden=False,
                        allow_setup=True,
                        env=env,
                    )
        config, setup_ran = result
        self.assertEqual(config["framework"], "mlx")
        self.assertFalse(setup_ran)
        run_cmd_mock.assert_not_called()

    def test_ensure_setup_missing_config_runs_setup(self) -> None:
        env: dict[str, str] = {}
        with patch.object(t2c, "load_config", side_effect=RuntimeError("missing")):
            with patch.object(t2c, "run_cmd") as run_cmd_mock:
                result = t2c.ensure_setup_if_needed(
                    framework="jax",
                    venv=Path(".venv-jax"),
                    framework_overridden=True,
                    allow_setup=True,
                    env=env,
                )
        config, setup_ran = result
        self.assertEqual(config["framework"], "jax")
        self.assertTrue(setup_ran)
        setup_cmd = run_cmd_mock.call_args[0][0]
        self.assertEqual(setup_cmd[1], "-m")
        self.assertEqual(setup_cmd[2], "tools.setup")
        self.assertEqual(setup_cmd[3], "jax")

    def test_ensure_setup_missing_and_no_setup_fails(self) -> None:
        env: dict[str, str] = {}
        with patch.object(t2c, "load_config", side_effect=RuntimeError("missing")):
            with self.assertRaises(RuntimeError):
                t2c.ensure_setup_if_needed(
                    framework="mlx",
                    venv=Path(".venv"),
                    framework_overridden=True,
                    allow_setup=False,
                    env=env,
                )

    def test_t2c_main_invalid_target(self) -> None:
        args = argparse.Namespace(
            target="bad",
            framework="mlx",
            venv=".venv",
            no_setup=True,
        )
        with patch.object(t2c, "parse_args", return_value=args):
            with patch.object(
                t2c, "ensure_setup_if_needed", return_value=({"framework": "mlx", "venv": ".venv"}, False)
            ):
                with patch.object(t2c, "run_cmd") as run_cmd_mock:
                    rc = t2c.main()
        self.assertEqual(rc, 1)
        run_cmd_mock.assert_not_called()

    def test_t2c_main_validate_path(self) -> None:
        args = argparse.Namespace(
            target="validate",
            framework="mlx",
            venv=".venv",
            no_setup=True,
        )
        with patch.object(t2c, "parse_args", return_value=args):
            with patch.object(
                t2c, "ensure_setup_if_needed", return_value=({"framework": "mlx", "venv": ".venv"}, False)
            ):
                with patch.object(t2c, "run_cmd") as run_cmd_mock:
                    with patch.object(runtime, "python_in_venv", return_value=Path(".venv/bin/python")):
                        with patch.object(t2c, "python_in_venv", return_value=Path(".venv/bin/python")):
                            rc = t2c.main()
        self.assertEqual(rc, 0)
        cmd = run_cmd_mock.call_args[0][0]
        self.assertEqual(cmd, [".venv/bin/python", "-m", "tools.validate", "--framework", "mlx"])

    def test_t2c_main_viz_path(self) -> None:
        args = argparse.Namespace(
            target="viz",
            framework="jax",
            venv=".venv-jax",
            no_setup=True,
        )
        with patch.object(t2c, "parse_args", return_value=args):
            with patch.object(
                t2c, "ensure_setup_if_needed", return_value=({"framework": "jax", "venv": ".venv-jax"}, False)
            ):
                with patch.object(t2c, "run_cmd") as run_cmd_mock:
                    with patch.object(t2c, "python_in_venv", return_value=Path(".venv-jax/bin/python")):
                        rc = t2c.main()
        self.assertEqual(rc, 0)
        cmd = run_cmd_mock.call_args[0][0]
        self.assertEqual(cmd, [".venv-jax/bin/python", "-m", "tools.viz_terminal", "--framework", "jax"])

    def test_t2c_main_onboarding_prompts_then_runs_validate_and_all(self) -> None:
        args = argparse.Namespace(
            target=None,
            framework=None,
            venv=None,
            no_setup=False,
        )
        with patch.object(t2c, "parse_args", return_value=args):
            with patch.object(t2c, "load_config", side_effect=RuntimeError("missing")):
                with patch.object(t2c.sys.stdin, "isatty", return_value=True):
                    with patch.object(t2c, "prompt_framework_choice", return_value="jax"):
                        with patch.object(
                            t2c,
                            "ensure_setup_if_needed",
                            return_value=({"framework": "jax", "venv": ".venv-jax"}, True),
                        ):
                            with patch.object(t2c, "python_in_venv", return_value=Path(".venv-jax/bin/python")):
                                with patch.object(t2c, "run_cmd") as run_cmd_mock:
                                    rc = t2c.main()
        self.assertEqual(rc, 0)
        calls = [c[0][0] for c in run_cmd_mock.call_args_list]
        self.assertEqual(calls[0], [".venv-jax/bin/python", "-m", "tools.validate", "--framework", "jax"])
        self.assertEqual(len(calls), 8)


if __name__ == "__main__":
    unittest.main()
