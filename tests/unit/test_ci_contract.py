from __future__ import annotations

import unittest
from pathlib import Path


class CiContractTests(unittest.TestCase):
    def test_ci_workflow_calls_mise_render_tasks(self) -> None:
        ci = Path('.github/workflows/ci.yml')
        self.assertTrue(ci.exists())
        text = ci.read_text(encoding='utf-8')
        for fragment in (
            'Headless TUI capture smoke check',
            'mise run install-render-system-deps',
            'mise run render-smoke-tui',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)

    def test_mise_render_tasks_include_headless_capture_requirements(self) -> None:
        mise = Path('mise.toml')
        self.assertTrue(mise.exists())
        text = mise.read_text(encoding='utf-8')
        for fragment in (
            'xvfb',
            'xterm',
            'xdotool',
            '--tui-capture headless',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)

    def test_assets_sync_workflow_only_triggers_on_asset_inputs(self) -> None:
        wf = Path('.github/workflows/assets-readme-sync.yml')
        self.assertTrue(wf.exists())
        text = wf.read_text(encoding='utf-8')
        # Should not run for README-only edits (README must not be in the trigger paths).
        self.assertNotIn('      - "README.md"', text)
        # Must include the sources that can change the generated GIFs.
        for fragment in (
            'transforms/**',
            'frameworks/**',
            'tools/shinkei.py',
            'tools/input_controls.py',
            'tools/generate_render_assets.py',
            'examples/inputs.framework_matrix.json',
            'examples/inputs.spectral_sweep.json',
            '.python-version',
            'mise.toml',
            '.github/ci/requirements-test.txt',
            '.github/workflows/assets-readme-sync.yml',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)

if __name__ == '__main__':
    unittest.main()
