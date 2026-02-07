from __future__ import annotations

import re
import unittest
from pathlib import Path


_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


class DocsLinksTests(unittest.TestCase):
    def test_markdown_relative_links_resolve(self) -> None:
        roots = [Path("README.md"), *sorted(Path("docs").rglob("*.md"))]
        for md in roots:
            text = md.read_text(encoding="utf-8")
            for raw_target in _LINK_RE.findall(text):
                target = raw_target.strip()
                if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                target = target.split("#", 1)[0]
                if not target:
                    continue
                resolved = (md.parent / target).resolve()
                with self.subTest(source=str(md), target=raw_target):
                    self.assertTrue(resolved.exists(), f"Broken link in {md}: {raw_target}")


if __name__ == "__main__":
    unittest.main()
