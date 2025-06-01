# tensors-to-consciousness

[![MLX v0.25.2](https://img.shields.io/badge/MLX-v0.25.2-blueviolet?logo=apple&logoColor=white)](https://github.com/ml-explore/mlx/releases/tag/v0.25.2)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/alexhraber/tensors-to-consciousness)

## Mathematical Foundations of AI/ML with MLX

A comprehensive journey through the theoretical foundations of artificial intelligence and machine learning, implemented using [Apple's MLX framework](https://ml-explore.github.io/mlx/build/html/index.html) on Apple Silicon.

### Overview

This repository contains a systematic exploration of computational theory, progressing through 7 levels:

0. **Computational Primitives** - Tensors, operations, reductions
1. **Automatic Differentiation** - Chain rule, gradients, backpropagation theory  
2. **Optimization Theory** - Gradient descent, momentum, adaptive methods
3. **Neural Network Theory** - Universal approximation, information flow
4. **Advanced Theory** - Manifold learning, attention mechanisms, Riemannian optimization
5. **Research Frontiers** - Meta-learning, scaling laws, lottery tickets, grokking
6. **Theoretical Limits** - Information geometry, consciousness, quantum computation

### Requirements

- Apple Silicon Mac (Metal GPU acceleration)
- Python 3.11+
- MLX framework
- Virtual environment recommended

### Setup

```bash
python -m venv mlx-env
source mlx-env/bin/activate
pip install mlx
```

then

```bash
python test_mlx_setup.py
```

### Usage

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

### Key Insights

- Modern AI emerges from elegant mathematical progressions
- Each level builds systematically on previous foundations
- Demonstrates the power of Apple Silicon and MLX for ML research
- Bridges multiple fields: mathematics, physics, neuroscience, computer science
