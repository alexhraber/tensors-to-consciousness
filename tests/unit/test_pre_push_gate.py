from __future__ import annotations

import os
import unittest

from tools.pre_push_gate import _parse_jobs_value, _resolve_jobs, select_act_tasks


class PrePushGateTests(unittest.TestCase):
    def test_parse_jobs_value_nproc(self) -> None:
        self.assertEqual(_parse_jobs_value("nproc"), os.cpu_count() or 2)

    def test_parse_jobs_value_int(self) -> None:
        self.assertEqual(_parse_jobs_value("4"), 4)

    def test_parse_jobs_value_invalid(self) -> None:
        self.assertIsNone(_parse_jobs_value("abc"))

    def test_resolve_jobs_defaults_to_cpu_for_multiple_tasks(self) -> None:
        self.assertEqual(
            _resolve_jobs(None, ["local-a", "local-b", "local-c", "local-d"]),
            min(os.cpu_count() or 2, 4),
        )

    def test_resolve_jobs_caps_by_task_count(self) -> None:
        self.assertEqual(_resolve_jobs("8", ["a", "b", "c"]), 3)

    def test_resolve_jobs_minimum_one(self) -> None:
        self.assertEqual(_resolve_jobs("0", ["a", "b", "c"]), 1)
        self.assertEqual(_resolve_jobs(None, ["a"]), 1)

    def test_resolve_jobs_parallelizes_act_tasks(self) -> None:
        self.assertEqual(_resolve_jobs("4", ["act-ci-test", "act-ci-docs-sync"]), 2)

    def test_runtime_change_selects_core_jobs(self) -> None:
        tasks = select_act_tasks(["tools/tui.py"])
        self.assertIn("act-ci-test", tasks)
        self.assertIn("act-ci-transform-contract", tasks)
        self.assertIn("act-ci-framework-contract-numpy", tasks)
        self.assertNotIn("act-ci-docs-sync", tasks)
        self.assertNotIn("act-ci-assets-sync", tasks)

    def test_tests_only_change_selects_test_job(self) -> None:
        tasks = select_act_tasks(["tests/unit/test_pre_push_gate.py"])
        self.assertEqual(tasks, ["act-ci-test"])

    def test_docs_only_change_selects_docs_job(self) -> None:
        tasks = select_act_tasks(["docs/usage/tui.md"])
        self.assertEqual(tasks, [])

    def test_catalog_source_change_selects_docs_job(self) -> None:
        tasks = select_act_tasks(["transforms/transforms.json"])
        self.assertIn("act-ci-docs-sync", tasks)

    def test_render_change_selects_assets_sync_only(self) -> None:
        tasks = select_act_tasks(["assets/render/tui_explorer.gif"])
        self.assertEqual(tasks, [])

    def test_ci_change_selects_all_ci_jobs(self) -> None:
        tasks = select_act_tasks([".github/workflows/ci.yml"])
        self.assertIn("act-ci-test", tasks)
        self.assertIn("act-ci-transform-contract", tasks)
        self.assertIn("act-ci-framework-contract-numpy", tasks)
        self.assertNotIn("act-ci-docs-sync", tasks)
        self.assertNotIn("act-ci-assets-sync", tasks)

    def test_irrelevant_change_selects_nothing(self) -> None:
        self.assertEqual(select_act_tasks(["LICENSE"]), [])


if __name__ == "__main__":
    unittest.main()
