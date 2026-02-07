from __future__ import annotations

import unittest
from pathlib import Path


class ContainerContractTests(unittest.TestCase):
    def test_docker_compose_has_interactive_explorer_service(self) -> None:
        compose = Path("docker-compose.yml")
        self.assertTrue(compose.exists())
        text = compose.read_text(encoding="utf-8")

        required_fragments = [
            "explorer:",
            "explorer-nvidia:",
            "explorer-amd:",
            "explorer-intel:",
            "explorer-apple:",
            "stdin_open: true",
            "tty: true",
            'command: ["explorer"]',
            "explorer_config:",
        ]
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
