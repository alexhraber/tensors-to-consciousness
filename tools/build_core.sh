#!/usr/bin/env bash
set -euo pipefail

# Build/install the optional core module into the active Python env.
# Requires: cargo, rustc, uv (for uvx), and a Python environment.
uvx maturin develop --release
