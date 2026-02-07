<div align="center">

<img src="assets/banner.svg" alt="tensors-to-consciousness banner" width="100%">

<p>
  <a href="https://pypi.org/project/mlx/"><img alt="MLX version" src="https://img.shields.io/pypi/v/mlx?label=MLX&logo=apple&logoColor=white&color=111111"></a>
  <a href="https://pypi.org/project/jax/"><img alt="JAX version" src="https://img.shields.io/pypi/v/jax?label=JAX&color=FE5F00"></a>
  <a href="https://pypi.org/project/torch/"><img alt="PyTorch version" src="https://img.shields.io/pypi/v/torch?label=PyTorch&logo=pytorch&logoColor=white&color=EE4C2C"></a>
  <a href="https://pypi.org/project/numpy/"><img alt="NumPy version" src="https://img.shields.io/pypi/v/numpy?label=NumPy&logo=numpy&logoColor=white&color=013243"></a>
  <a href="https://pypi.org/project/keras/"><img alt="Keras version" src="https://img.shields.io/pypi/v/keras?label=Keras&logo=keras&logoColor=white&color=D00000"></a>
  <a href="https://pypi.org/project/cupy-cuda12x/"><img alt="CuPy version" src="https://img.shields.io/pypi/v/cupy-cuda12x?label=CuPy&logo=nvidia&logoColor=white&color=76B900"></a>
</p>

<p>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <a href="https://github.com/alexhraber/tensors-to-consciousness/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/alexhraber/tensors-to-consciousness/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.14" src="https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white">
</p>

</div>

---

## Terminal Exploration Platform

`tensors-to-consciousness` is a framework-agnostic exploration system for mathematical transforms on tensor fields.

Core architecture:

1. `transforms/`: canonical transform definitions and catalog metadata.
2. `frameworks/`: execution engines that consume transform sequences.
3. `tools/shinkei.py`: rendering and terminal presentation path.
4. `tools/tui.py`: interactive operator surface for selection, ordering, and control.
5. `explorer.py`: unified entrypoint and runtime orchestration.

## Quickstart (Container-First)

```bash
docker compose build explorer
docker compose run --rm explorer
```

GPU profiles:

```bash
docker compose --profile nvidia run --rm explorer-nvidia
docker compose --profile amd run --rm explorer-amd
docker compose --profile intel run --rm explorer-intel
docker compose --profile apple run --rm explorer-apple
```

Local launch:

```bash
python explorer.py
```

First-run defaults:

- macOS: `mlx`
- other platforms: `numpy`

<p align="center">
  <img src="assets/render/tui_explorer.gif" alt="TUI explorer preview" width="86%">
</p>

## Visual Samples

Headless captures of three transform progression examples (real `explorer.py run` path):

<p align="center">
  <img src="assets/render/optimization_flow.gif" alt="Optimization flow sample" width="32%">
  <img src="assets/render/attention_dynamics.gif" alt="Attention dynamics sample" width="32%">
  <img src="assets/render/phase_portraits.gif" alt="Phase portrait sample" width="32%">
</p>

## Supported Frameworks

- `mlx`
- `jax`
- `pytorch`
- `numpy`
- `keras`
- `cupy`

## Top 5 Core Transforms

- `tensor_ops`
- `chain_rule`
- `gradient_descent`
- `momentum`
- `adam`

## References

- [Documentation Index](docs/README.md)
- [Architecture and Contracts](docs/reference/architecture.md)
- [TUI Guide](docs/usage/tui.md)
- [CLI Guide](docs/usage/cli.md)
- [Container Guide](docs/usage/container.md)
- [Diagnostics and Logging](docs/usage/diagnostics.md)
- [Render Asset Generation](docs/usage/assets.md)
- [Transform Catalog](docs/reference/transforms.md)
- [Supported Frameworks](docs/reference/frameworks.md)
- [Rust Core](docs/development/rust-core.md)
- [Contributing](CONTRIBUTING.md)

## Core Commands

```bash
python explorer.py --list-transforms
python explorer.py run --framework jax --transforms chain_rule,gradient_descent,adam
python explorer.py render --framework numpy --transforms default
python -m tests
```
