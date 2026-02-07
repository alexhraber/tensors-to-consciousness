# tensors-to-consciousness

[![Backend: MLX (default)](https://img.shields.io/badge/Backend-MLX-blue?logo=apple&logoColor=white)](https://github.com/ml-explore/mlx)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/alexhraber/tensors-to-consciousness)

## Mathematical Foundations of AI/ML

A comprehensive journey through the theoretical foundations of artificial intelligence and machine learning, now structured with a backend abstraction so frameworks beyond MLX can be added without rewriting chapter code.

### Overview

This repository contains a systematic exploration of computational theory, progressing through 7 levels:

0. **Computational Primitives** - Tensors, operations, reductions
1. **Automatic Differentiation** - Chain rule, gradients, backpropagation theory
2. **Optimization Theory** - Gradient descent, momentum, adaptive methods
3. **Neural Network Theory** - Universal approximation, information flow
4. **Advanced Theory** - Manifold learning, attention mechanisms, Riemannian optimization
5. **Research Frontiers** - Meta-learning, scaling laws, lottery tickets, grokking
6. **Theoretical Limits** - Information geometry, consciousness, quantum computation

## Backend Architecture

Chapter scripts import from `t2c.frameworks` instead of importing MLX directly:

```python
import t2c.frameworks as fw

mx = fw.mx
nn = fw.nn  # where needed
```

Backend selection is environment-driven:

```bash
export T2C_BACKEND=mlx
```

Currently supported backend:
- `mlx` (default)

If an unsupported backend is requested, startup fails with a clear configuration error.

## Requirements

- Python 3.11+
- Virtual environment recommended
- Backend-specific dependency (currently `mlx`)

## Setup

```bash
python -m venv env
source env/bin/activate
pip install mlx
```

Then validate backend wiring:

```bash
python test_backend_setup.py
```

`test_mlx_setup.py` is retained as a compatibility entrypoint and runs the same backend setup test.

## Usage

Run the scripts in sequence:

```bash
python 0_computational_primitives.py
python 1_automatic_differentiation.py
python 2_optimization_theory.py
python 3_neural_theory.py
python 4_advanced_theory.py
python 5_research_frontiers.py
python 6_theoretical_limits.py
```

## Dedicated JAX Scripts

A full JAX replication of chapters `0`-`6` is available under `scripts/jax/`.

Setup:

```bash
python -m venv env
source env/bin/activate
pip install "jax[cpu]"
```

Validate JAX environment:

```bash
python scripts/jax/test_jax_setup.py
```

Run JAX chapters:

```bash
python scripts/jax/0_computational_primitives.py
python scripts/jax/1_automatic_differentiation.py
python scripts/jax/2_optimization_theory.py
python scripts/jax/3_neural_theory.py
python scripts/jax/4_advanced_theory.py
python scripts/jax/5_research_frontiers.py
python scripts/jax/6_theoretical_limits.py
```

## Dedicated PyTorch Scripts

A full PyTorch replication of chapters `0`-`6` is available under `scripts/pytorch/`.

Setup:

```bash
python -m venv env
source env/bin/activate
pip install torch
```

Validate PyTorch environment:

```bash
python scripts/pytorch/test_pytorch_setup.py
```

Run PyTorch chapters:

```bash
python scripts/pytorch/0_computational_primitives.py
python scripts/pytorch/1_automatic_differentiation.py
python scripts/pytorch/2_optimization_theory.py
python scripts/pytorch/3_neural_theory.py
python scripts/pytorch/4_advanced_theory.py
python scripts/pytorch/5_research_frontiers.py
python scripts/pytorch/6_theoretical_limits.py
```

## Dedicated NumPy Scripts

A full NumPy replication of chapters `0`-`6` is available under `scripts/numpy/`.

Setup:

```bash
python -m venv env
source env/bin/activate
pip install numpy
```

Validate NumPy environment:

```bash
python scripts/numpy/test_numpy_setup.py
```

Run NumPy chapters:

```bash
python scripts/numpy/0_computational_primitives.py
python scripts/numpy/1_automatic_differentiation.py
python scripts/numpy/2_optimization_theory.py
python scripts/numpy/3_neural_theory.py
python scripts/numpy/4_advanced_theory.py
python scripts/numpy/5_research_frontiers.py
python scripts/numpy/6_theoretical_limits.py
```

## Dedicated Keras Scripts

A full Keras replication of chapters `0`-`6` is available under `scripts/keras/`.

Setup:

```bash
python -m venv env
source env/bin/activate
pip install keras tensorflow
```

Validate Keras environment:

```bash
python scripts/keras/test_keras_setup.py
```

Run Keras chapters:

```bash
python scripts/keras/0_computational_primitives.py
python scripts/keras/1_automatic_differentiation.py
python scripts/keras/2_optimization_theory.py
python scripts/keras/3_neural_theory.py
python scripts/keras/4_advanced_theory.py
python scripts/keras/5_research_frontiers.py
python scripts/keras/6_theoretical_limits.py
```

## Dedicated CuPy Scripts

A full CuPy replication of chapters `0`-`6` is available under `scripts/cupy/`.

Setup:

```bash
python -m venv env
source env/bin/activate
pip install cupy-cuda12x
```

Validate CuPy environment:

```bash
python scripts/cupy/test_cupy_setup.py
```

Run CuPy chapters:

```bash
python scripts/cupy/0_computational_primitives.py
python scripts/cupy/1_automatic_differentiation.py
python scripts/cupy/2_optimization_theory.py
python scripts/cupy/3_neural_theory.py
python scripts/cupy/4_advanced_theory.py
python scripts/cupy/5_research_frontiers.py
python scripts/cupy/6_theoretical_limits.py
```

## Adding a New Backend Later

1. Add `t2c/frameworks/<name>_backend.py` with a `load()` function returning `Backend(name, core, nn)`.
2. Register it in `_AVAILABLE_BACKENDS` in `t2c/frameworks/__init__.py`.
3. Install that framework's dependency.
4. Run `python test_backend_setup.py` and chapter scripts with `T2C_BACKEND=<name>`.
