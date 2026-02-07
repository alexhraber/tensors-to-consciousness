from __future__ import annotations

import argparse
import unittest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Python tests (transform/ML surface only)."
    )
    parser.add_argument(
        "--suite",
        choices=["all", "unit"],
        default="unit",
        help="Select test suite scope (default: unit).",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=2,
        help="Unittest verbosity level (default: 2).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loader = unittest.defaultTestLoader
    # All Python tests live directly under `tests/` as `test_*.py`.
    suite = loader.discover("tests")
    result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

