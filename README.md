<div align="center">

# tensors-to-consciousness

<img src="assets/banner.svg" alt="tensors-to-consciousness banner" width="100%">

<p>
  <a href="https://pypi.org/project/mlx/"><img alt="MLX version" src="https://img.shields.io/pypi/v/mlx?label=MLX&logo=apple&logoColor=white&color=111111"></a>
  <a href="https://pypi.org/project/jax/"><img alt="JAX version" src="https://img.shields.io/pypi/v/jax?label=JAX&color=FE5F00&logo=https%3A%2F%2Fraw.githubusercontent.com%2Fjax-ml%2Fjax%2Fmain%2Fimages%2Fjax_logo_250px.png"></a>
  <a href="https://pypi.org/project/torch/"><img alt="PyTorch version" src="https://img.shields.io/pypi/v/torch?label=PyTorch&logo=pytorch&logoColor=white&color=EE4C2C"></a>
  <a href="https://pypi.org/project/numpy/"><img alt="NumPy version" src="https://img.shields.io/pypi/v/numpy?label=NumPy&logo=numpy&logoColor=white&color=013243"></a>
  <a href="https://pypi.org/project/keras/"><img alt="Keras version" src="https://img.shields.io/pypi/v/keras?label=Keras&logo=keras&logoColor=white&color=D00000"></a>
  <a href="https://pypi.org/project/cupy-cuda12x/"><img alt="CuPy version" src="https://img.shields.io/pypi/v/cupy-cuda12x?label=CuPy&logo=nvidia&logoColor=white&color=76B900"></a>
</p>

<p>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <a href="https://deepwiki.com/alexhraber/tensors-to-consciousness"><img alt="Ask DeepWiki" src="https://deepwiki.com/badge.svg"></a>
</p>

</div>

---

## CLI-First Research Workflow

This project is intentionally CLI-native.

- No notebook dependency for normal operation
- Single setup command per framework
- Single runtime command for validation, execution, and terminal visualization

Primary commands:

```bash
python -m tools.setup <framework>
python t2c.py <target>
```

## Overview

This repository contains 7 theory-first research modules implemented across peer framework tracks:

- `mlx`
- `jax`
- `pytorch`
- `numpy`
- `keras`
- `cupy`

Top-level operational commands:

- `python -m tools.setup <framework>`: install + validate selected framework and save active selection
- `python t2c.py <target>`: run targets (`validate`, `viz`, `0..6`, `all`) for active framework and auto-setup if needed

The standard workflow is:

1. `python -m tools.setup <framework>`
2. `python t2c.py validate`
3. `python t2c.py viz`
4. `python t2c.py 0` ... `python t2c.py 6` or `python t2c.py all`

## Research Module Sequence

| # | Research Module | Focus |
|---|---|---|
| 0 | Computational Primitives | Tensors, operations, reductions |
| 1 | Automatic Differentiation | Chain rule, gradients, backpropagation theory |
| 2 | Optimization Theory | Gradient descent, momentum, adaptive methods |
| 3 | Neural Network Theory | Universal approximation, information flow |
| 4 | Advanced Theory | Manifolds, attention, Riemannian optimization |
| 5 | Research Frontiers | Meta-learning, scaling laws, lottery tickets, grokking |
| 6 | Theoretical Limits | Information geometry, consciousness, quantum-inspired computation |

## Quickstart

### 1) Create environment

```bash
python -m venv env
source env/bin/activate
```

### 2) Setup framework (pick once)

```bash
python -m tools.setup mlx
# or: jax | pytorch | numpy | keras | cupy
```

### 3) Run commands

```bash
python t2c.py validate
python t2c.py viz
python t2c.py 0
python t2c.py all
```

### 4) Run top-level operational tests

```bash
python -m tests
```

## Primary Setup Script

`python -m tools.setup` is the only setup entrypoint.

```bash
python -m tools.setup <framework>
```

Supported values:

- `mlx`
- `jax`
- `pytorch`
- `numpy`
- `keras`
- `cupy`
- `all`

What setup does:

1. Creates/uses a virtual environment via `uv` (default: `.venv`)
2. Installs dependencies with `uv pip install`
3. Runs the corresponding validation script (`scripts/<framework>/test_setup.py`)
4. Saves active selection to `.t2c/config.json`

Useful options:

```bash
python -m tools.setup jax --venv .venv-jax
python -m tools.setup cupy --skip-validate
python -m tools.setup all
```

## Runtime Entrypoint

`t2c.py` is the single runtime entrypoint. It reads `.t2c/config.json`, and if the configured venv is missing, it runs `python -m tools.setup` automatically before executing your target.

```bash
python t2c.py 0
python t2c.py 1
python t2c.py 2
python t2c.py 3
python t2c.py 4
python t2c.py 5
python t2c.py 6
python t2c.py all
python t2c.py viz
```

Optional overrides:

```bash
python t2c.py 0 --framework jax
python t2c.py all --venv .venv-jax
python t2c.py 0 --no-setup
```

Framework scripts still exist and are used by `t2c.py` internally:

- `scripts/mlx/*`
- `scripts/jax/*`
- `scripts/pytorch/*`
- `scripts/numpy/*`
- `scripts/keras/*`
- `scripts/cupy/*`

## Framework Tracks

All framework scripts live under `scripts/<framework>/`.

| Framework | Setup | Validation | Research Modules |
|---|---|---|---|
| MLX | `python -m tools.setup mlx` | `python t2c.py validate` | `python t2c.py 0` ... `python t2c.py 6` |
| JAX | `python -m tools.setup jax` | `python t2c.py validate` | `python t2c.py 0` ... `python t2c.py 6` |
| PyTorch | `python -m tools.setup pytorch` | `python t2c.py validate` | `python t2c.py 0` ... `python t2c.py 6` |
| NumPy | `python -m tools.setup numpy` | `python t2c.py validate` | `python t2c.py 0` ... `python t2c.py 6` |
| Keras | `python -m tools.setup keras` | `python t2c.py validate` | `python t2c.py 0` ... `python t2c.py 6` |
| CuPy | `python -m tools.setup cupy` | `python t2c.py validate` | `python t2c.py 0` ... `python t2c.py 6` |

## Terminal Visualization

`python t2c.py viz` renders a Matplotlib chart directly in your terminal as ASCII.

- Works over SSH/headless sessions (no GUI required)
- Uses Matplotlib backend rendering, then converts the figure to terminal characters
- Great for quick signal checks while staying in CLI workflows

## Notes

- `numpy` and `cupy` use finite-difference gradients in autodiff-heavy sections.
- `keras` mixes gradient tape and numerical approximations in selected sections.
- CuPy install in setup defaults to `cupy-cuda12x`; use a wheel matching your CUDA runtime.
- `python -m tests` validates setup/runtime orchestration and CLI behavior (not framework theory correctness).

## Contributing

See `CONTRIBUTING.md` for backend and framework-track contribution workflow.
