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

<p><strong>Mathematical Foundations of AI/ML Across Frameworks</strong></p>
<p>Canonical backend-selected chapter path + dedicated peer tracks for MLX, JAX, PyTorch, NumPy, Keras, and CuPy.</p>

</div>

---

## Contents

- [What This Repo Is](#what-this-repo-is)
- [Chapter Sequence](#chapter-sequence)
- [Live Previews](#live-previews)
- [Quickstart](#quickstart)
- [Primary Setup Script](#primary-setup-script)
- [Universal Runner](#universal-runner)
- [Canonical Path (Backend-Selected)](#canonical-path-backend-selected)
- [Dedicated Framework Tracks](#dedicated-framework-tracks)
- [Notes](#notes)
- [Contributing](#contributing)

## What This Repo Is

This repository walks through 7 chapters of AI/ML theory with:

- A backend-abstracted canonical path (`t2c.frameworks`)
- Dedicated framework replicas in `scripts/`:
  - `mlx`
  - `jax`
  - `pytorch`
  - `numpy`
  - `keras`
  - `cupy`

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

## Live Previews

<table>
  <tr>
    <td><strong>MLX</strong><br><img src="assets/previews/mlx.gif" alt="MLX preview" width="360"></td>
    <td><strong>JAX</strong><br><img src="assets/previews/jax.gif" alt="JAX preview" width="360"></td>
  </tr>
  <tr>
    <td><strong>PyTorch</strong><br><img src="assets/previews/pytorch.gif" alt="PyTorch preview" width="360"></td>
    <td><strong>NumPy</strong><br><img src="assets/previews/numpy.gif" alt="NumPy preview" width="360"></td>
  </tr>
  <tr>
    <td><strong>Keras</strong><br><img src="assets/previews/keras.gif" alt="Keras preview" width="360"></td>
    <td><strong>CuPy</strong><br><img src="assets/previews/cupy.gif" alt="CuPy preview" width="360"></td>
  </tr>
</table>

## Quickstart

### 1) Create environment

```bash
python -m venv env
source env/bin/activate
```

### 2) Pick a framework track

```bash
python setup_framework.py <framework>
```

Examples:

```bash
python setup_framework.py mlx
python setup_framework.py jax
python setup_framework.py pytorch
python setup_framework.py numpy
python setup_framework.py keras
python setup_framework.py cupy
```

### 3) Use universal commands (no framework re-entry)

```bash
python run.py validate
python run.py 0
python run.py all
```

## Primary Setup Script

Use `setup_framework.py` as the entrypoint for environment setup and dependency install:

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

What it does:

1. Creates/uses a virtual environment via `uv` (default: `.venv`).
2. Installs framework dependencies with `uv pip install`.
3. Runs the corresponding validation script (`test_*_setup.py`).
4. Saves the selected active framework to `.t2c/config.json`.

Useful options:

```bash
python setup_framework.py jax --venv .venv-jax
python setup_framework.py cupy --skip-validate
python setup_framework.py all
```

## Universal Runner

After selecting a framework once with `setup_framework.py`, use `run.py` for all routine commands:

```bash
python run.py validate   # run framework validation script
python run.py 0          # run chapter 0
python run.py 1          # run chapter 1
python run.py 2
python run.py 3
python run.py 4
python run.py 5
python run.py 6
python run.py all        # run chapters 0..6 in sequence
```

Optional overrides:

```bash
python run.py 0 --framework jax
python run.py all --venv .venv-jax
```

## Canonical Path (Backend-Selected)

Main chapter scripts (`0_...py` to `6_...py`) import from `t2c.frameworks`:

```python
import t2c.frameworks as fw

mx = fw.mx
nn = fw.nn
```

Select backend via env var (`mlx` is currently available in the backend abstraction):

```bash
export T2C_BACKEND=mlx
```

### Setup

```bash
python setup_framework.py mlx
python run.py validate
python run.py all
```

Validation scripts (legacy-compatible naming):

```bash
python test_mlx_setup.py
python test_backend_setup.py
```

### Run all main chapters

```bash
python 0_computational_primitives.py
python 1_automatic_differentiation.py
python 2_optimization_theory.py
python 3_neural_theory.py
python 4_advanced_theory.py
python 5_research_frontiers.py
python 6_theoretical_limits.py
```

## Dedicated Framework Tracks

All dedicated tracks live under `scripts/<framework>/`.

| Framework | Install | Setup Test | Chapters |
|---|---|---|---|
| MLX | `python setup_framework.py mlx` | `python run.py validate` | `python run.py 0` ... `python run.py 6` |
| JAX | `python setup_framework.py jax` | `python run.py validate` | `python run.py 0` ... `python run.py 6` |
| PyTorch | `python setup_framework.py pytorch` | `python run.py validate` | `python run.py 0` ... `python run.py 6` |
| NumPy | `python setup_framework.py numpy` | `python run.py validate` | `python run.py 0` ... `python run.py 6` |
| Keras | `python setup_framework.py keras` | `python run.py validate` | `python run.py 0` ... `python run.py 6` |
| CuPy | `python setup_framework.py cupy` | `python run.py validate` | `python run.py 0` ... `python run.py 6` |

> CuPy: choose the wheel that matches your CUDA version.
> The setup script uses `cupy-cuda12x` by default.

## Notes

- `numpy` and `cupy` tracks use finite-difference gradients in autodiff-heavy sections.
- `keras` track mixes gradient tape with numerical approximations in selected sections.
- Missing dependencies will surface as `ModuleNotFoundError` in setup tests.
- `test_*_setup.py` scripts are validation scripts and are run by `setup_framework.py`.
- `setup_framework.py` selects and stores active framework once; `run.py` is the universal runner afterward.

## Contributing

Contributions are welcome for both backend abstraction work and dedicated framework tracks.

### Add a Backend to `t2c.frameworks` (Main Path)

1. Add `t2c/frameworks/<name>_backend.py` implementing `load() -> Backend`.
2. Register `<name>` in `_AVAILABLE_BACKENDS` inside `t2c/frameworks/__init__.py`.
3. Keep exports aligned with the current contract (`mx`, optional `nn`).
4. Validate with:
   - `python test_backend_setup.py`
   - `T2C_BACKEND=<name> python 0_computational_primitives.py`
   - `T2C_BACKEND=<name> python 1_automatic_differentiation.py`

### Add a Dedicated Framework Track (Latest Pattern)

1. Create `scripts/<framework>/` with:
   - `utils.py`
   - `test_<framework>_setup.py`
   - `0_...py` through `6_...py`
2. Keep chapter naming and progression identical to main path.
3. Document install + setup test + run commands in `README.md`.
4. Prefer framework-native autodiff; if not available, use finite-difference and note it in `Notes`.
