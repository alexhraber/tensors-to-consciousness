# Rust Core

Python remains the product surface (`explorer.py`, `tools/tui.py`, `tools/shinkei.py`).
Rust provides optional compute/render acceleration via `ttc_rust_core`.

## Scope

Current Rust-accelerated paths:

- ASCII heatmap generation
- Pixel/ANSI heatmap generation
- Assignment parsing helpers for interactive edits (`key=value`)
- Runtime platform/venv normalization helpers
- Frame patch primitive for terminal diffing

Fallback behavior:

- If the extension is unavailable, Python implementations are used automatically.
- Set `TTC_DISABLE_RUST_CORE=1` to force Python fallback.

## Build

```bash
./tools/build_rust_core.sh
```

Equivalent manual command:

```bash
uvx maturin develop --release -m rust_core/Cargo.toml
```

## Verify

```bash
python - <<'PY'
from tools import rust_core
print("loaded:", rust_core.load_rust_core() is not None)
PY
```

## Notes

- Rust crate location: `rust_core/`
- Python bridge: `tools/rust_core.py`
- Runtime integration points: `tools/shinkei.py`
