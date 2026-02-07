from __future__ import annotations

import ast
from pathlib import Path
import unittest

from tools.runtime import SUPPORTED_FRAMEWORKS


ROOT = Path(__file__).resolve().parents[2]


class FrameworkContractsTests(unittest.TestCase):
    def test_framework_layout_contract(self) -> None:
        for framework in SUPPORTED_FRAMEWORKS:
            with self.subTest(framework=framework):
                root = ROOT / "frameworks" / framework
                self.assertTrue(root.is_dir(), f"Missing framework directory: {root}")
                self.assertTrue((root / "utils.py").is_file(), f"Missing utils.py for {framework}")
                self.assertTrue((root / "test_setup.py").is_file(), f"Missing test_setup.py for {framework}")
                self.assertTrue(
                    (root / "algorithms" / "__init__.py").is_file(),
                    f"Missing algorithms package init for {framework}",
                )

    def test_utils_exports_required_surface(self) -> None:
        required_functions = {"normal", "viz_stage", "_to_numpy"}
        for framework in SUPPORTED_FRAMEWORKS:
            with self.subTest(framework=framework):
                path = ROOT / "frameworks" / framework / "utils.py"
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                found = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
                missing = required_functions - found
                self.assertFalse(missing, f"{framework} utils.py missing required functions: {sorted(missing)}")

    def test_setup_script_references_utils_contract(self) -> None:
        for framework in SUPPORTED_FRAMEWORKS:
            with self.subTest(framework=framework):
                path = ROOT / "frameworks" / framework / "test_setup.py"
                text = path.read_text(encoding="utf-8")
                self.assertIn("normal", text, f"{framework} test_setup.py should reference normal()")
                self.assertIn("Setup Test", text, f"{framework} test_setup.py should print setup banner")


if __name__ == "__main__":
    unittest.main()
