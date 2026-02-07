from __future__ import annotations

import unittest
from pathlib import Path


class CiContractTests(unittest.TestCase):
    def test_ci_workflow_includes_headless_capture_requirements(self) -> None:
        ci = Path('.github/workflows/ci.yml')
        self.assertTrue(ci.exists())
        text = ci.read_text(encoding='utf-8')
        for fragment in (
            'xvfb',
            'xterm',
            'xdotool',
            'Headless TUI capture smoke check',
            '--tui-capture headless',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)


if __name__ == '__main__':
    unittest.main()
