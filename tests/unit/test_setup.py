from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import setup


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


if __name__ == "__main__":
    unittest.main()

