# Contributing

## Architecture

The repo is framework-track based.

- Source implementations live in `scripts/<framework>/`
- Operational commands are:
  - `t2c.py` (single public entrypoint; setup is auto-triggered)

## Local workflow

```bash
python t2c.py
python t2c.py all
python -m tests
```

`python -m tests` runs top-level operational tests (setup/runner/config behavior) and intentionally avoids framework-specific math validation.

## Add or update a framework track

A framework track should include:

- `scripts/<framework>/utils.py`
- `scripts/<framework>/test_setup.py`
- `scripts/<framework>/0_computational_primitives.py`
- `scripts/<framework>/1_automatic_differentiation.py`
- `scripts/<framework>/2_optimization_theory.py`
- `scripts/<framework>/3_neural_theory.py`
- `scripts/<framework>/4_advanced_theory.py`
- `scripts/<framework>/5_research_frontiers.py`
- `scripts/<framework>/6_theoretical_limits.py`

Requirements:

- Keep research-module names and sequence identical.
- Keep conceptual behavior aligned across frameworks.
- Prefer framework-native autodiff where available.
- If numerical gradients are required, document it in `README.md`.

## Runner integration checklist

When adding/changing frameworks:

1. Update `tools/setup.py` framework map (`deps`, `validate` script).
2. Ensure `t2c.py` can resolve the framework name and scripts.
3. Validate setup + run path:

```bash
python t2c.py
python t2c.py validate
python t2c.py 0
```

## Documentation expectations

- Keep `README.md` aligned with actual setup/run behavior.
- Keep examples aligned with the operational command (`t2c.py`).
- If visualization previews change, regenerate GIF assets with `python tools/generate_viz_assets.py`.
- Keep advanced usage discoverable through `python t2c.py --help`.

## Hygiene

- Do not commit generated runtime state (`.t2c/`, `.venv/`, `out/`, `__pycache__/`).
- Keep scripts runnable with direct `python <script>.py`.
- Prefer small, focused commits with clear messages.
