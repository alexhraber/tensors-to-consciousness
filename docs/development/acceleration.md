# Core Module

Rust is the product surface (`explorer`).
Python executes ML math (transforms + framework backends). Rust provides optional accelerated kernels via `core`.

## Scope

Current Rust-accelerated paths:

- ASCII heatmap generation
- Pixel/ANSI heatmap generation
- Assignment parsing helpers for interactive edits (`key=value`)
- Runtime platform/venv normalization helpers
- Frame patch primitive for terminal diffing

Fallback behavior:

- If the extension is unavailable, Python implementations are used automatically.
- Set `EXPLORER_DISABLE_CORE=1` to force Python fallback.

## Build

```bash
./tools/build_core.sh
```

Equivalent manual command:

```bash
uvx maturin develop --release
```

## Verify

```bash
python - <<'PY'
from tools import core
print("loaded:", core.load_core() is not None)
PY
```

## Notes

- Rust crate location: `crates/core/`
- Python bridge: `tools/core.py`
- Runtime integration points: `crates/explorer/src/shinkei.rs`
