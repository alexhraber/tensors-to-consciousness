# Architecture and Contracts

`tensors-to-consciousness` is organized around a strict separation of concerns:

1. Transform catalog (`transforms/`): mathematical definitions and metadata.
2. Framework engines (`frameworks/`): backend-specific execution surfaces.
3. Runtime and orchestration (`explorer` (Rust), `tools/runtime.py` (Python)): environment, setup, and dispatch.
4. Interaction and rendering (`explorer` (Rust) TUI host, `crates/explorer/src/shinkei.rs` (Rust)): operator control plane and terminal output.
5. Optional core module (`crates/core/`): accelerated compute/render kernels behind Python bridges.

## Execution Model

At runtime, the system evaluates a selected transform chain in order:

1. Resolve transform keys from registry.
2. Load framework engine and backend utility contract.
3. Materialize the tensor field.
4. Apply transforms sequentially as state mutations.
5. Emit render frames through Shinkei.

This model intentionally avoids hardcoding module permutations. Framework engines discover and execute transform sequences dynamically.

Rust is the public product surface (CLI + TUI). Python is the compute layer for transforms and framework execution.

## Transform Contract

Canonical transform metadata is defined in `transforms/transforms.json` and consumed through registry/catalog helpers.

Each transform entry includes:

- stable `key`
- operator-facing `title`
- `complexity`
- `formula`
- implementation binding and sensible defaults

## Framework Contract

Each framework backend must satisfy the runtime interface in `frameworks/<framework>/utils.py`.

Required utility entrypoints:

- `normal`
- `_to_numpy`

Required ops adapter behavior:

- `add`
- `sub`
- `mul`
- `matmul`
- `transpose`
- `zeros_like`
- `normal_like`

## Runtime Defaults

Default framework selection is platform-aware:

- macOS: `mlx`
- other platforms: `jax`

Diagnostics defaults:

- `log_level=INFO`
- `debug=false`

## Documentation Generation

The transform/framework references are generated artifacts.

```bash
python tools/generate_catalog_docs.py
```
