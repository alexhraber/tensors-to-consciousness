# Tests

This repository is Rust-first.

- Rust tests: `cargo test -p explorer`
- Python (ML/transform) tests: `python -m tests.python --suite unit`

`mise` wraps the common paths:

- `mise run test` (Rust by default)
- `mise run test -- --scope python`
- `mise run test -- --scope all`
