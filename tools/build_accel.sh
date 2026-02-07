#!/usr/bin/env bash
set -euo pipefail

# Build/install the optional Rust acceleration module into the active Python env.
# Requires: cargo, rustc, uv (for uvx), and a Python environment.
uvx maturin develop --release -m crates/accel/Cargo.toml
