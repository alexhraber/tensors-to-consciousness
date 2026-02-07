# Contributing

## Architecture

The repo is sandbox/playground based.

- Transform definitions/math contracts live in `transforms/`
- Framework backends live in `frameworks/<framework>/`
- Transform registry (ordering/complexity/defaults) lives in `transforms/registry.py`
- Operational commands are:
  - `main.py` (single public entrypoint; setup is auto-triggered)

## Local workflow

```bash
python tools/install_githooks.py
python main.py
python main.py run --transforms default
python -m tests
```

`python -m tests` runs top-level operational tests (setup/runner/config behavior) and engine contract checks.

Git hooks:
- Local `pre-commit` hook lives in `.githooks/pre-commit`.
- Install it once per clone with `python tools/install_githooks.py`.
- The hook regenerates `docs/transforms.md` and `docs/frameworks.md` before each commit.

## Add or update a framework track

A framework track should include:

- `frameworks/<framework>/utils.py`
- `frameworks/<framework>/test_setup.py`
- optional adapter helpers under `frameworks/<framework>/transforms/`

Requirements:

- Keep transform behavior aligned across frameworks.
- Keep conceptual behavior aligned across frameworks.
- Prefer framework-native autodiff where available.
- If numerical gradients are required, document it in `README.md`.

## Runner integration checklist

When adding/changing frameworks:

1. Update `tools/setup.py` framework map (`deps`, `validate` script).
2. Ensure `main.py` can resolve the framework name and transform selection.
3. Add/update transform metadata in `transforms/registry.py`.
4. Prefer scaffolding new abstract transforms via `python tools/scaffold_algo.py ...`.
5. Validate setup + run path:

```bash
python main.py
python main.py validate
python main.py run --transforms default
```

## Documentation expectations

- Keep `README.md` aligned with actual setup/run behavior.
- Keep examples aligned with the operational command (`main.py`).
- Regenerate transform/framework docs when catalog or framework contracts change:
  - `python tools/generate_catalog_docs.py`
- If visualization previews change, regenerate GIF assets with `python tools/generate_viz_assets.py`.
- Keep advanced usage discoverable through `python main.py --help`.

## Hygiene

- Do not commit generated runtime state (`.t2c/`, `.venv/`, `out/`, `__pycache__/`).
- Keep frameworks runnable with direct `python <script>.py`.
- Prefer small, focused commits with clear messages.
