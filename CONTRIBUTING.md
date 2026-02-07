# Contributing

This project is maintained as an architecture-first exploration platform. Contributions are expected to preserve contract clarity, runtime reliability, and operator ergonomics.

## System Model

- `transforms/`: canonical transform catalog and metadata definitions.
- `frameworks/<framework>/`: backend execution adapters.
- `tools/`: runtime, diagnostics, rendering, and TUI surfaces.
- `explorer.py`: primary user-facing entrypoint.
- `rust_core/`: optional acceleration kernels consumed by Python bridges.

## Local Workflow

```bash
python tools/install_githooks.py
python -m pip install pre-commit
pre-commit install
python explorer.py
python explorer.py run --transforms default
python -m tests
```

## Commit Hygiene

- Keep changes scoped and reviewable.
- Regenerate generated references when catalog/framework contracts change.
- Do not commit runtime state (`.config/`, virtual environments, caches, output scratch files).

## Hooks and Automation

- Repo hook entry: `.githooks/pre-commit`
- Installer: `python tools/install_githooks.py`
- Full pre-commit run: `pre-commit run --all-files`
- Catalog docs generator: `python tools/generate_catalog_docs.py`
- Render asset generator: `python tools/generate_render_assets.py`
- Rust core build: `./tools/build_rust_core.sh`

## Adding or Updating Framework Backends

Each framework backend must include:

- `frameworks/<framework>/utils.py`
- `frameworks/<framework>/test_setup.py`

Optional backend-specialized helpers may live under `frameworks/<framework>/transforms/`.

Requirements:

- preserve conceptual behavior across frameworks
- prefer native autodiff where supported
- explicitly document numerical approximation tradeoffs

## Transform and Runtime Checklist

1. Add or modify transform definitions in `transforms/transforms.json`.
2. Verify registry/catalog resolution remains valid.
3. Validate setup and run paths through `explorer.py`.
4. Run both unit and integration test suites.

Validation commands:

```bash
python explorer.py --list-transforms
python explorer.py validate
python explorer.py run --transforms default
python -m tests --suite unit
python -m tests --suite integration
```

## Documentation Expectations

- Keep `README.md` concise and operator-oriented.
- Keep implementation detail in `docs/` reference and usage guides.
- Ensure every user-facing command in docs is executable as written.
- Keep `docs/reference/transforms.md` and `docs/reference/frameworks.md` synchronized with source metadata.
