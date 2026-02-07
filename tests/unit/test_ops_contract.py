import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class OpsContractTests(unittest.TestCase):
    def test_hooks_call_rust_ops(self) -> None:
        pre_commit = (ROOT / ".githooks" / "pre-commit").read_text(encoding="utf-8")
        pre_push = (ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8")

        self.assertIn("explorer ops git-policy --hook pre-commit", pre_commit)
        self.assertIn("explorer ops bootstrap", pre_commit)

        # pre-push policy is enforced in bash to avoid building Rust for docs-only diffs.
        self.assertIn("explorer ops pre-push-gate", pre_push)

    def test_mise_tasks_use_rust_ops(self) -> None:
        mise = (ROOT / "mise.toml").read_text(encoding="utf-8")
        self.assertIn("explorer ops pre-push-gate", mise)
        self.assertIn("explorer ops submit-pr", mise)
