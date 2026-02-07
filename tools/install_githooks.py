#!/usr/bin/env python3
"""Install repository Git hooks by setting core.hooksPath to .githooks."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".githooks"


def main() -> int:
    if not HOOKS.exists():
        raise SystemExit(f"Missing hooks directory: {HOOKS}")
    subprocess.run(["git", "config", "core.hooksPath", str(HOOKS)], check=True, cwd=ROOT)
    print(f"Installed hooks path: {HOOKS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
