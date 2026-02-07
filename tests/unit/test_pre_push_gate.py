from __future__ import annotations

import unittest

from tools.pre_push_gate import select_act_tasks


class PrePushGateTests(unittest.TestCase):
    def test_runtime_change_selects_core_jobs(self) -> None:
        tasks = select_act_tasks(["tools/tui.py"])
        self.assertIn("act-ci-test", tasks)
        self.assertIn("act-ci-transform-contract", tasks)
        self.assertIn("act-ci-framework-contract-numpy", tasks)
        self.assertIn("act-ci-docs-sync", tasks)
        self.assertNotIn("act-ci-assets-sync", tasks)

    def test_docs_only_change_selects_docs_job(self) -> None:
        tasks = select_act_tasks(["docs/usage/tui.md"])
        self.assertEqual(tasks, ["act-ci-docs-sync"])

    def test_render_change_selects_assets_sync_only(self) -> None:
        tasks = select_act_tasks(["assets/render/tui_explorer.gif"])
        self.assertEqual(tasks, [])

    def test_ci_change_selects_all_ci_jobs(self) -> None:
        tasks = select_act_tasks([".github/workflows/ci.yml"])
        self.assertIn("act-ci-test", tasks)
        self.assertIn("act-ci-transform-contract", tasks)
        self.assertIn("act-ci-framework-contract-numpy", tasks)
        self.assertIn("act-ci-docs-sync", tasks)
        self.assertNotIn("act-ci-assets-sync", tasks)

    def test_irrelevant_change_selects_nothing(self) -> None:
        self.assertEqual(select_act_tasks(["LICENSE"]), [])


if __name__ == "__main__":
    unittest.main()
