from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import subprocess

import explorer as app
from tools import runtime


class AppFlowIntegrationTests(unittest.TestCase):
    def test_run_cmd_interrupt_exits_cleanly(self) -> None:
        with patch.object(subprocess, "run", side_effect=KeyboardInterrupt):
            with self.assertRaises(SystemExit) as exc:
                app.run_cmd(["python", "-V"], env={})
        self.assertEqual(exc.exception.code, 130)

    def test_ensure_setup_not_needed(self) -> None:
        env: dict[str, str] = {}
        with tempfile.TemporaryDirectory() as td:
            venv_dir = Path(td) / ".venv"
            py = runtime.python_in_venv(venv_dir)
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("", encoding="utf-8")
            with patch.object(app, "load_config", return_value={"framework": "mlx", "venv": str(venv_dir)}):
                with patch.object(app, "run_cmd") as run_cmd_mock:
                    result = app.ensure_setup_if_needed(
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
        with patch.object(app, "load_config", side_effect=RuntimeError("missing")):
            with patch.object(app, "run_cmd") as run_cmd_mock:
                result = app.ensure_setup_if_needed(
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
        with patch.object(app, "load_config", side_effect=RuntimeError("missing")):
            with self.assertRaises(RuntimeError):
                app.ensure_setup_if_needed(
                    framework="mlx",
                    venv=Path(".venv"),
                    framework_overridden=True,
                    allow_setup=False,
                    env=env,
                )

    def test_ensure_setup_switch_framework_uses_isolated_default_venv(self) -> None:
        env: dict[str, str] = {}
        with patch.object(app, "load_config", return_value={"framework": "mlx", "venv": ".venv-mlx"}):
            with patch.object(app, "run_cmd") as run_cmd_mock:
                config, setup_ran = app.ensure_setup_if_needed(
                    framework="jax",
                    venv=None,
                    framework_overridden=True,
                    allow_setup=True,
                    env=env,
                )
        self.assertTrue(setup_ran)
        self.assertEqual(config["framework"], "jax")
        self.assertEqual(config["venv"], ".venv-jax")
        cmd = run_cmd_mock.call_args[0][0]
        self.assertIn(".venv-jax", cmd)

    def test_app_main_invalid_target(self) -> None:
        args = argparse.Namespace(
            target="bad",
            framework="mlx",
            venv=".venv",
            no_setup=True,
            cli=False,
            inputs=None,
            transforms=None,
            transform=None,
            list_transforms=False,
        )
        with patch.object(app, "parse_args", return_value=args):
            with patch.object(
                app, "ensure_setup_if_needed", return_value=({"framework": "mlx", "venv": ".venv"}, False)
            ):
                with patch.object(app, "run_cmd") as run_cmd_mock:
                    rc = app.main()
        self.assertEqual(rc, 1)
        run_cmd_mock.assert_not_called()

    def test_app_main_validate_path(self) -> None:
        args = argparse.Namespace(
            target="validate",
            framework="mlx",
            venv=".venv",
            no_setup=True,
            cli=False,
            inputs=None,
            transforms=None,
            transform=None,
            list_transforms=False,
        )
        with patch.object(app, "parse_args", return_value=args):
            with patch.object(
                app, "ensure_setup_if_needed", return_value=({"framework": "mlx", "venv": ".venv"}, False)
            ):
                with patch.object(app, "run_cmd") as run_cmd_mock:
                    with patch.object(runtime, "python_in_venv", return_value=Path(".venv/bin/python")):
                        with patch.object(app, "python_in_venv", return_value=Path(".venv/bin/python")):
                            rc = app.main()
        self.assertEqual(rc, 0)
        cmd = run_cmd_mock.call_args[0][0]
        self.assertEqual(cmd, [".venv/bin/python", "-m", "tools.validate", "--framework", "mlx"])

    def test_app_main_render_path(self) -> None:
        args = argparse.Namespace(
            target="render",
            framework="jax",
            venv=".venv-jax",
            no_setup=True,
            cli=False,
            inputs=None,
            transforms=None,
            transform=None,
            list_transforms=False,
        )
        with patch.object(app, "parse_args", return_value=args):
            with patch.object(
                app, "ensure_setup_if_needed", return_value=({"framework": "jax", "venv": ".venv-jax"}, False)
            ):
                with patch.object(app, "run_cmd") as run_cmd_mock:
                    with patch.object(app, "python_in_venv", return_value=Path(".venv-jax/bin/python")):
                        rc = app.main()
        self.assertEqual(rc, 0)
        cmd = run_cmd_mock.call_args[0][0]
        self.assertEqual(cmd, [".venv-jax/bin/python", "-m", "tools.tui", "--framework", "jax"])

    def test_app_main_onboarding_uses_default_framework_then_runs_validate_and_render(self) -> None:
        args = argparse.Namespace(
            target=None,
            framework=None,
            venv=None,
            no_setup=False,
            cli=False,
            inputs=None,
            transforms=None,
            transform=None,
            list_transforms=False,
        )
        with patch.object(app, "parse_args", return_value=args):
            with patch.object(app, "load_config", side_effect=RuntimeError("missing")):
                with patch.object(
                    app,
                    "ensure_setup_if_needed",
                    return_value=({"framework": "numpy", "venv": ".venv-np"}, True),
                ) as ensure_setup_mock:
                    with patch.object(app, "python_in_venv", return_value=Path(".venv-np/bin/python")):
                        with patch.object(app, "run_cmd") as run_cmd_mock:
                            rc = app.main()
        self.assertEqual(rc, 0)
        calls = [c[0][0] for c in run_cmd_mock.call_args_list]
        self.assertEqual(calls[0], [".venv-np/bin/python", "-m", "tools.validate", "--framework", "numpy"])
        self.assertEqual(calls[1], [".venv-np/bin/python", "-m", "tools.tui", "--framework", "numpy"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(ensure_setup_mock.call_args.kwargs["framework"], app.DEFAULT_FRAMEWORK)

    def test_app_main_onboarding_cli_flag_runs_modules_without_prompting(self) -> None:
        args = argparse.Namespace(
            target=None,
            framework=None,
            venv=None,
            no_setup=False,
            cli=True,
            inputs=None,
            transforms=None,
            transform=None,
            list_transforms=False,
        )
        with patch.object(app, "parse_args", return_value=args):
            with patch.object(app, "load_config", side_effect=RuntimeError("missing")):
                with patch.object(
                    app,
                    "ensure_setup_if_needed",
                    return_value=({"framework": "numpy", "venv": ".venv-np"}, True),
                ) as ensure_setup_mock:
                    with patch.object(app, "python_in_venv", return_value=Path(".venv-np/bin/python")):
                        with patch.object(app, "run_cmd") as run_cmd_mock:
                            rc = app.main()
        self.assertEqual(rc, 0)
        calls = [c[0][0] for c in run_cmd_mock.call_args_list]
        self.assertEqual(calls[0], [".venv-np/bin/python", "-m", "tools.validate", "--framework", "numpy"])
        self.assertEqual(calls[1], [".venv-np/bin/python", "-m", "tools.playground", "--framework", "numpy", "--transforms", "default", "--render"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(ensure_setup_mock.call_args.kwargs["framework"], app.DEFAULT_FRAMEWORK)

    def test_app_main_passes_inputs_env(self) -> None:
        args = argparse.Namespace(
            target="run",
            framework="jax",
            venv=".venv-jax",
            no_setup=True,
            cli=False,
            inputs="examples/inputs.example.json",
            transforms="gradient_descent",
            transform=None,
            list_transforms=False,
        )
        with patch.object(app, "parse_args", return_value=args):
            with patch.object(
                app, "ensure_setup_if_needed", return_value=({"framework": "jax", "venv": ".venv-jax"}, False)
            ):
                with patch.object(app, "python_in_venv", return_value=Path(".venv-jax/bin/python")):
                    with patch.object(app, "run_cmd") as run_cmd_mock:
                        rc = app.main()
        self.assertEqual(rc, 0)
        env = run_cmd_mock.call_args.kwargs["env"]
        self.assertEqual(env.get("INPUTS"), "examples/inputs.example.json")

    def test_app_main_executes_selected_algo_module(self) -> None:
        args = argparse.Namespace(
            target="run",
            framework=None,
            venv=None,
            no_setup=True,
            cli=False,
            inputs=None,
            transforms="tensor_ops",
            transform=None,
            list_transforms=False,
        )
        with patch.object(app, "parse_args", return_value=args):
            with patch.object(
                app, "ensure_setup_if_needed", return_value=({"framework": "numpy", "venv": ".venv-np"}, False)
            ):
                with patch.object(app, "python_in_venv", return_value=Path(".venv-np/bin/python")):
                    with patch.object(app, "run_cmd") as run_cmd_mock:
                        rc = app.main()
        self.assertEqual(rc, 0)
        cmd = run_cmd_mock.call_args[0][0]
        self.assertEqual(
            cmd,
            [".venv-np/bin/python", "-m", "tools.playground", "--framework", "numpy", "--transforms", "tensor_ops", "--render"],
        )


if __name__ == "__main__":
    unittest.main()
