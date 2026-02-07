# Test Layout

This repository separates test intent by scope:

- `tests/unit`: fast, isolated tests for single modules/functions.
- `tests/integration`: entrypoint and cross-module flow tests.

Run commands:

- `python -m tests` (all suites)
- `python -m tests --suite unit`
- `python -m tests --suite integration`

CI should execute `all`; local iteration should prefer `unit` first.

