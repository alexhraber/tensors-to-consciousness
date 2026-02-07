#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys


ALLOWED_TYPES = ("build", "chore", "ci", "docs", "feat", "fix", "perf", "refactor", "revert", "style", "test")
BRANCH_PATTERN = re.compile(
    r"^(?:build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)/[a-z0-9]+(?:-[a-z0-9]+)*$"
)


def _current_branch() -> str:
    return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()


def _main_guard(branch: str, hook: str) -> tuple[bool, str]:
    if branch not in {"main", "master"}:
        return True, ""
    if hook == "pre-commit":
        if os.environ.get("ALLOW_MAIN_COMMIT") == "1":
            return True, ""
        return (
            False,
            "Direct commits to main/master are blocked. "
            "Create a feature branch or set ALLOW_MAIN_COMMIT=1 for exceptional maintenance.",
        )
    if hook == "pre-push":
        if os.environ.get("ALLOW_MAIN_PUSH") == "1":
            return True, ""
        return (
            False,
            "Direct pushes to main/master are blocked. "
            "Push a feature branch and open a PR, or set ALLOW_MAIN_PUSH=1 for exceptional maintenance.",
        )
    return True, ""


def _branch_name_guard(branch: str) -> tuple[bool, str]:
    if branch in {"HEAD", "main", "master"}:
        return True, ""
    if BRANCH_PATTERN.match(branch):
        return True, ""
    return (
        False,
        "Branch name must match "
        "'type/scope-short-topic' with lowercase kebab-case.\n"
        f"Allowed types: {', '.join(ALLOWED_TYPES)}\n"
        "Examples: fix/ci-act-container-collision, docs/readme-minimal-refresh",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enforce repository git workflow policy.")
    parser.add_argument("--hook", choices=["pre-commit", "pre-push"], required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    branch = _current_branch()

    ok, msg = _main_guard(branch, args.hook)
    if not ok:
        print(msg, file=sys.stderr)
        return 1

    ok, msg = _branch_name_guard(branch)
    if not ok:
        print(msg, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
