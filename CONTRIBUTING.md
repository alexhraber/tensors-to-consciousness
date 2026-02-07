# Contributing

## Architecture

The repo is sandbox/playground based.

- Algorithm implementations live in `algos/<framework>/`
- Framework backends live in `frameworks/<framework>/`
- Algorithm registry (ordering/complexity/defaults) lives in `algos/registry.py`
- Operational commands are:
  - `main.py` (single public entrypoint; setup is auto-triggered)

## Local workflow

```bash
python main.py
python main.py run --algos default
python -m tests
```

`python -m tests` runs top-level operational tests (setup/runner/config behavior) and intentionally avoids framework-specific math validation.

## Add or update a framework track

A framework track should include:

- `frameworks/<framework>/utils.py`
- `frameworks/<framework>/test_setup.py`
- optional adapter modules under `frameworks/<framework>/algorithms/`

Requirements:

- Keep algorithm behavior aligned across frameworks.
- Keep conceptual behavior aligned across frameworks.
- Prefer framework-native autodiff where available.
- If numerical gradients are required, document it in `README.md`.

## Runner integration checklist

When adding/changing frameworks:

1. Update `tools/setup.py` framework map (`deps`, `validate` script).
2. Ensure `main.py` can resolve the framework name and algorithm selection.
3. Add/update algorithm metadata in `algos/registry.py`.
4. Prefer scaffolding new abstract algorithms via `python tools/scaffold_algo.py ...`.
5. Validate setup + run path:

```bash
python main.py
python main.py validate
python main.py run --algos default
```

## Documentation expectations

- Keep `README.md` aligned with actual setup/run behavior.
- Keep examples aligned with the operational command (`main.py`).
- If visualization previews change, regenerate GIF assets with `python tools/generate_viz_assets.py`.
- Keep advanced usage discoverable through `python main.py --help`.

## Hygiene

- Do not commit generated runtime state (`.t2c/`, `.venv/`, `out/`, `__pycache__/`).
- Keep frameworks runnable with direct `python <script>.py`.
- Prefer small, focused commits with clear messages.
