# Contributing

## Scope

This project has two extension surfaces:

1. Backend abstraction (`t2c/frameworks`) for canonical top-level scripts
2. Dedicated framework track (`scripts/<framework>/`) for per-framework chapter replicas

## Local Workflow

```bash
python setup_framework.py <framework>
python run.py validate
python run.py all
```

## Add a Backend (`t2c/frameworks`)

1. Add `t2c/frameworks/<name>_backend.py` with `load() -> Backend`.
2. Register `<name>` in `_AVAILABLE_BACKENDS` in `t2c/frameworks/__init__.py`.
3. Ensure backend exports meet contract:
   - `core` (`mx` alias)
   - optional `nn`
4. Validate:

```bash
T2C_BACKEND=<name> python 0_computational_primitives.py
T2C_BACKEND=<name> python 1_automatic_differentiation.py
python test_backend_setup.py
```

## Add a Framework Track (`scripts/<framework>`)

Create:

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

- Keep chapter naming and sequence identical.
- Keep conceptual output aligned across frameworks.
- Use framework-native autodiff where available.
- If numerical gradients are used, state it clearly in `README.md`.

## Documentation Requirements

When adding/changing frameworks:

- Update `README.md` setup and notes.
- Ensure `setup_framework.py` includes install + validation mapping.
- Ensure `run.py` can resolve validation/chapter execution for the framework.

## Style and Hygiene

- Do not commit generated runtime state (`.t2c/`, `out/`, `__pycache__/`).
- Keep scripts executable with direct `python <script>.py` usage.
- Prefer small, isolated commits with clear messages.
