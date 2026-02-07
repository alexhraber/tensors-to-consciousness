from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import setup


class SetupScriptTests(unittest.TestCase):
    def test_framework_config_has_nonempty_dependencies(self) -> None:
        for framework, cfg in setup.FRAMEWORK_CONFIG.items():
            with self.subTest(framework=framework):
                deps = cfg.get("deps")
                self.assertIsInstance(deps, list)
                self.assertTrue(deps, "framework deps must be non-empty")
                self.assertTrue(all(isinstance(dep, str) and dep for dep in deps))

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

    def test_setup_one_dependency_chain_all_frameworks(self) -> None:
        for framework, cfg in setup.FRAMEWORK_CONFIG.items():
            with self.subTest(framework=framework):
                calls: list[list[str]] = []

                def fake_run_cmd(cmd: list[str], env=None) -> None:
                    calls.append(cmd)

                with patch.object(setup, "run_cmd", side_effect=fake_run_cmd):
                    setup.setup_one(framework, Path(".venv"), skip_validate=True)

                self.assertEqual(len(calls), 1)
                install_cmd = calls[0]
                self.assertEqual(install_cmd[:5], ["uv", "pip", "install", "--python", ".venv/bin/python"])
                self.assertIn("--upgrade", install_cmd)

                # Ensure common deps are always installed before framework deps.
                dep_chain = install_cmd[install_cmd.index("--upgrade") + 1 :]
                self.assertEqual(dep_chain[: len(setup.COMMON_DEPS)], setup.COMMON_DEPS)
                self.assertEqual(dep_chain[len(setup.COMMON_DEPS) :], cfg["deps"])


if __name__ == "__main__":
    unittest.main()
