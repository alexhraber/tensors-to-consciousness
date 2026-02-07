# Acceleration

Rust is the product surface (`explorer`).
Python executes transform and framework math. Rust provides optional compute/render acceleration via `explorer_accel`.

## Scope

Current Rust-accelerated paths:

- ASCII heatmap generation
- Pixel/ANSI heatmap generation
- Assignment parsing helpers for interactive edits (`key=value`)
- Runtime platform/venv normalization helpers
- Frame patch primitive for terminal diffing

Fallback behavior:

- If the extension is unavailable, Python implementations are used automatically.
- Set `EXPLORER_DISABLE_ACCEL=1` to force Python fallback.

## Build

```bash
./tools/build_accel.sh
```

Equivalent manual command:

```bash
uvx maturin develop --release -m crates/accel/Cargo.toml
```

## Verify

```bash
python - <<'PY'
from tools import accel
print("loaded:", accel.load_accel() is not None)
PY
```

## Notes

- Rust crate location: `crates/accel/`
- Python bridge: `tools/accel.py`
- Runtime integration points: `tools/shinkei.py`
