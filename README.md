<div align="center">

# tensors-to-consciousness

<img src="assets/banner.svg" alt="tensors-to-consciousness banner" width="100%">

<p>
  <a href="https://pypi.org/project/mlx/"><img alt="MLX version" src="https://img.shields.io/pypi/v/mlx?label=MLX&logo=apple&logoColor=white&color=111111"></a>
  <a href="https://pypi.org/project/jax/"><img alt="JAX logo" src="https://pypi-camo.freetls.fastly.net/a1669ade48edd86167330c456c5da04cde93a850/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6a61782d6d6c2f6a61782f6d61696e2f696d616765732f6a61785f6c6f676f5f32353070782e706e67" height="20"></a>
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

#### MLX

```bash
pip install mlx
python scripts/mlx/test_mlx_setup.py
python scripts/mlx/0_computational_primitives.py
```

#### JAX

```bash
pip install "jax[cpu]"
python scripts/jax/test_jax_setup.py
python scripts/jax/0_computational_primitives.py
```

#### PyTorch

```bash
pip install torch
python scripts/pytorch/test_pytorch_setup.py
python scripts/pytorch/0_computational_primitives.py
```

#### NumPy

```bash
pip install numpy
python scripts/numpy/test_numpy_setup.py
python scripts/numpy/0_computational_primitives.py
```

#### Keras

```bash
pip install keras tensorflow
python scripts/keras/test_keras_setup.py
python scripts/keras/0_computational_primitives.py
```

#### CuPy

```bash
pip install cupy-cuda12x
python scripts/cupy/test_cupy_setup.py
python scripts/cupy/0_computational_primitives.py
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
pip install mlx
export T2C_BACKEND=mlx
python test_backend_setup.py
```

Compatibility entrypoint (legacy naming):

```bash
python test_mlx_setup.py
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
| MLX | `pip install mlx` | `python scripts/mlx/test_mlx_setup.py` | `python scripts/mlx/0_computational_primitives.py` ... `python scripts/mlx/6_theoretical_limits.py` |
| JAX | `pip install "jax[cpu]"` | `python scripts/jax/test_jax_setup.py` | `python scripts/jax/0_computational_primitives.py` ... `python scripts/jax/6_theoretical_limits.py` |
| PyTorch | `pip install torch` | `python scripts/pytorch/test_pytorch_setup.py` | `python scripts/pytorch/0_computational_primitives.py` ... `python scripts/pytorch/6_theoretical_limits.py` |
| NumPy | `pip install numpy` | `python scripts/numpy/test_numpy_setup.py` | `python scripts/numpy/0_computational_primitives.py` ... `python scripts/numpy/6_theoretical_limits.py` |
| Keras | `pip install keras tensorflow` | `python scripts/keras/test_keras_setup.py` | `python scripts/keras/0_computational_primitives.py` ... `python scripts/keras/6_theoretical_limits.py` |
| CuPy | `pip install cupy-cuda12x` | `python scripts/cupy/test_cupy_setup.py` | `python scripts/cupy/0_computational_primitives.py` ... `python scripts/cupy/6_theoretical_limits.py` |

> CuPy: choose the wheel that matches your CUDA version.

## Notes

- `numpy` and `cupy` tracks use finite-difference gradients in autodiff-heavy sections.
- `keras` track mixes gradient tape with numerical approximations in selected sections.
- Missing dependencies will surface as `ModuleNotFoundError` in setup tests.

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
