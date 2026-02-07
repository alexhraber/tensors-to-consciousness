from __future__ import annotations

import os
import unittest

from tools.submit_pr import _api_resolve_ips, _parse_repo_from_remote_url


class SubmitPrTests(unittest.TestCase):
    def test_parse_repo_from_git_ssh_url(self) -> None:
        ref = _parse_repo_from_remote_url("git@github.com:owner/repo.git")
        self.assertEqual(ref.owner, "owner")
        self.assertEqual(ref.name, "repo")

    def test_parse_repo_from_https_url(self) -> None:
        ref = _parse_repo_from_remote_url("https://github.com/owner/repo.git")
        self.assertEqual(ref.owner, "owner")
        self.assertEqual(ref.name, "repo")

    def test_api_resolve_ips_uses_default(self) -> None:
        prev = os.environ.pop("GH_API_RESOLVE_IPS", None)
        try:
            self.assertEqual(_api_resolve_ips(), ("140.82.114.6", "140.82.113.6", "140.82.112.6"))
        finally:
            if prev is not None:
                os.environ["GH_API_RESOLVE_IPS"] = prev

    def test_api_resolve_ips_uses_env_override(self) -> None:
        prev = os.environ.get("GH_API_RESOLVE_IPS")
        os.environ["GH_API_RESOLVE_IPS"] = "1.1.1.1, 2.2.2.2"
        try:
            self.assertEqual(_api_resolve_ips(), ("1.1.1.1", "2.2.2.2"))
        finally:
            if prev is None:
                os.environ.pop("GH_API_RESOLVE_IPS", None)
            else:
                os.environ["GH_API_RESOLVE_IPS"] = prev


if __name__ == "__main__":
    unittest.main()
