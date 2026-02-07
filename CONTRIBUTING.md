# Contributing

## Architecture

The repo is framework-track based.

- Source implementations live in `scripts/<framework>/`
- Operational commands are:
  - `setup.py` (choose/install/validate once)
  - `validate.py` (run validation for active framework)
  - `run.py` (run chapter targets)

## Local workflow

```bash
python setup.py <framework>
python validate.py
python run.py all
```

## Add or update a framework track

A framework track should include:

- `scripts/<framework>/utils.py`
- `scripts/<framework>/test_<framework>_setup.py`
- `scripts/<framework>/0_computational_primitives.py`
- `scripts/<framework>/1_automatic_differentiation.py`
- `scripts/<framework>/2_optimization_theory.py`
- `scripts/<framework>/3_neural_theory.py`
- `scripts/<framework>/4_advanced_theory.py`
- `scripts/<framework>/5_research_frontiers.py`
- `scripts/<framework>/6_theoretical_limits.py`

Requirements:

- Keep chapter names and sequence identical.
- Keep conceptual behavior aligned across frameworks.
- Prefer framework-native autodiff where available.
- If numerical gradients are required, document it in `README.md`.

## Runner integration checklist

When adding/changing frameworks:

1. Update `setup.py` framework map (`deps`, `validate` script).
2. Ensure `run.py` can resolve the framework name and scripts.
3. Validate setup + run path:

```bash
python setup.py <framework>
python validate.py
python run.py 0
```

## Documentation expectations

- Keep `README.md` aligned with actual setup/run behavior.
- Keep examples aligned with operational commands (`setup.py`, `validate.py`, `run.py`).

## Hygiene

- Do not commit generated runtime state (`.t2c/`, `.venv/`, `out/`, `__pycache__/`).
- Keep scripts runnable with direct `python <script>.py`.
- Prefer small, focused commits with clear messages.
