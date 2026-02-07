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
- Single public entrypoint (`t2c.py`)
- Setup is inferred and executed automatically from config + CLI inputs

Primary command:

```bash
python t2c.py <target>
```

## Visualization Preview

<p align="center">
  <img src="assets/viz/optimization_flow.gif" alt="Optimization flow visualization" width="32%">
  <img src="assets/viz/attention_dynamics.gif" alt="Attention dynamics visualization" width="32%">
  <img src="assets/viz/phase_portraits.gif" alt="Phase portrait visualization" width="32%">
</p>

## Overview

This repository contains 7 theory-first research modules implemented across peer framework tracks:

- `mlx`
- `jax`
- `pytorch`
- `numpy`
- `keras`
- `cupy`

Top-level operational command:

- `python t2c.py <target>`: run targets (`validate`, `viz`, `0..6`, `all`); setup auto-runs when required

The standard workflow is:

1. `python t2c.py validate --framework jax` (first run sets/bootstraps framework)
2. `python t2c.py viz`
3. `python t2c.py 0` ... `python t2c.py 6` or `python t2c.py all`

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

### 2) First run (framework selection + auto-setup)

```bash
python t2c.py validate --framework mlx
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

## Runtime Entrypoint

`t2c.py` is the single runtime entrypoint. It reads `.t2c/config.json`; if framework config or venv is missing, it bootstraps setup automatically based on your `--framework` input.

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

| Framework | First Run | Validation | Research Modules |
|---|---|---|---|
| MLX | `python t2c.py validate --framework mlx` | `python t2c.py validate` | `python t2c.py 0` ... `python t2c.py 6` |
| JAX | `python t2c.py validate --framework jax` | `python t2c.py validate` | `python t2c.py 0` ... `python t2c.py 6` |
| PyTorch | `python t2c.py validate --framework pytorch` | `python t2c.py validate` | `python t2c.py 0` ... `python t2c.py 6` |
| NumPy | `python t2c.py validate --framework numpy` | `python t2c.py validate` | `python t2c.py 0` ... `python t2c.py 6` |
| Keras | `python t2c.py validate --framework keras` | `python t2c.py validate` | `python t2c.py 0` ... `python t2c.py 6` |
| CuPy | `python t2c.py validate --framework cupy` | `python t2c.py validate` | `python t2c.py 0` ... `python t2c.py 6` |

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
