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

## Overview

This repository contains 7 theory-first chapters implemented across peer framework tracks:

- `mlx`
- `jax`
- `pytorch`
- `numpy`
- `keras`
- `cupy`

The standard workflow is:

1. `python setup_framework.py <framework>`
2. `python run.py validate`
3. `python run.py 0` ... `python run.py 6` or `python run.py all`

## Chapter Sequence

| # | Chapter | Focus |
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
python setup_framework.py mlx
# or: jax | pytorch | numpy | keras | cupy
```

### 3) Run commands

```bash
python run.py validate
python run.py 0
python run.py all
```

## Primary Setup Script

`setup_framework.py` is the only setup entrypoint.

```bash
python setup_framework.py <framework>
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
3. Runs the corresponding validation script (`test_*_setup.py`)
4. Saves active selection to `.t2c/config.json`

Useful options:

```bash
python setup_framework.py jax --venv .venv-jax
python setup_framework.py cupy --skip-validate
python setup_framework.py all
```

## Run Script

`run.py` reads `.t2c/config.json` and runs commands for your selected framework.

```bash
python run.py validate
python run.py 0
python run.py 1
python run.py 2
python run.py 3
python run.py 4
python run.py 5
python run.py 6
python run.py all
```

Optional overrides:

```bash
python run.py 0 --framework jax
python run.py all --venv .venv-jax
```

Framework scripts still exist and are used by `run.py` internally:

- `scripts/mlx/*`
- `scripts/jax/*`
- `scripts/pytorch/*`
- `scripts/numpy/*`
- `scripts/keras/*`
- `scripts/cupy/*`

## Framework Tracks

All framework scripts live under `scripts/<framework>/`.

| Framework | Setup | Validation | Chapters |
|---|---|---|---|
| MLX | `python setup_framework.py mlx` | `python run.py validate` | `python run.py 0` ... `python run.py 6` |
| JAX | `python setup_framework.py jax` | `python run.py validate` | `python run.py 0` ... `python run.py 6` |
| PyTorch | `python setup_framework.py pytorch` | `python run.py validate` | `python run.py 0` ... `python run.py 6` |
| NumPy | `python setup_framework.py numpy` | `python run.py validate` | `python run.py 0` ... `python run.py 6` |
| Keras | `python setup_framework.py keras` | `python run.py validate` | `python run.py 0` ... `python run.py 6` |
| CuPy | `python setup_framework.py cupy` | `python run.py validate` | `python run.py 0` ... `python run.py 6` |

## Notes

- `numpy` and `cupy` use finite-difference gradients in autodiff-heavy sections.
- `keras` mixes gradient tape and numerical approximations in selected sections.
- CuPy install in setup defaults to `cupy-cuda12x`; use a wheel matching your CUDA runtime.
- Legacy scripts `test_backend_setup.py` / `test_mlx_setup.py` are validation aliases retained for compatibility.

## Contributing

See `CONTRIBUTING.md` for backend and framework-track contribution workflow.
