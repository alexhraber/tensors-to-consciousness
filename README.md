<div align="center">

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
  <a href="https://github.com/alexhraber/tensors-to-consciousness/actions/workflows/ci-tests.yml"><img alt="CI Tests" src="https://github.com/alexhraber/tensors-to-consciousness/actions/workflows/ci-tests.yml/badge.svg"></a>
  <a href="https://github.com/alexhraber/tensors-to-consciousness/actions/workflows/ci-contracts.yml"><img alt="CI Contracts" src="https://github.com/alexhraber/tensors-to-consciousness/actions/workflows/ci-contracts.yml/badge.svg"></a>
  <a href="https://github.com/alexhraber/tensors-to-consciousness/actions/workflows/ci-docs-sync.yml"><img alt="CI Docs" src="https://github.com/alexhraber/tensors-to-consciousness/actions/workflows/ci-docs-sync.yml/badge.svg"></a>
  <a href="https://github.com/alexhraber/tensors-to-consciousness/actions/workflows/ci-viz-assets.yml"><img alt="CI Viz Assets" src="https://github.com/alexhraber/tensors-to-consciousness/actions/workflows/ci-viz-assets.yml/badge.svg"></a>
  <img alt="Python 3.14" src="https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white">
  <a href="https://deepwiki.com/alexhraber/tensors-to-consciousness"><img alt="Ask DeepWiki" src="https://deepwiki.com/badge.svg"></a>
</p>

</div>

---

## Visualization Preview

<p align="center">
  <img src="assets/viz/optimization_flow.gif" alt="Optimization flow visualization" width="32%">
  <img src="assets/viz/attention_dynamics.gif" alt="Attention dynamics visualization" width="32%">
  <img src="assets/viz/phase_portraits.gif" alt="Phase portrait visualization" width="32%">
</p>

## TUI-First Research Studio

`tensors-to-consciousness` is built around an interactive terminal studio for mathematical AI/ML exploration.
Primary entrypoint:

```bash
python main.py
```

On first run it will:

1. Detect framework config
2. Use default framework `numpy` if missing
3. Auto-setup environment + latest available dependencies
4. Validate the framework track
5. Launch the interactive visualization studio

Detailed usage lives in:

- [TUI Guide](docs/tui.md)
- [CLI Guide](docs/cli.md)
- [Transforms Reference](docs/transforms.md)
- [Frameworks Reference](docs/frameworks.md)

Toolchain baseline: Python `3.14` + `uv` (latest stable).

## Supported Frameworks

The project currently supports the following frameworks:

- `mlx`
- `jax`
- `pytorch`
- `numpy`
- `keras`
- `cupy`

## Transforms

Transforms are framework-agnostic math definitions consumed by framework engines.
Canonical definitions live in `transforms/transforms.json`.

Top 5 core transforms:

- `tensor_ops`
- `chain_rule`
- `gradient_descent`
- `momentum`
- `adam`

Ultra imperative commands:

```bash
python main.py --list-transforms
python main.py run --framework jax --transforms chain_rule,gradient_descent,adam
python main.py viz --framework numpy --transforms default
```

For full transform details, presets, and nuanced examples, use:
- [Transforms Reference](docs/transforms.md)
- [Transform Playbook](docs/transform-playbook.md)

References:
- [Contributing Guide](CONTRIBUTING.md)
- [Tests Guide](tests/README.md)
- [CI Tests Workflow](.github/workflows/ci-tests.yml)
- [CI Contracts Workflow](.github/workflows/ci-contracts.yml)
- [CI Docs Sync Workflow](.github/workflows/ci-docs-sync.yml)
- [CI Viz Assets Workflow](.github/workflows/ci-viz-assets.yml)

## Notes

- `python -m tests` validates CLI/setup/runtime operations.
- `numpy` and `cupy` use finite-difference gradients in autodiff-heavy sections.
- `keras` mixes gradient tape and numerical approximations in selected sections.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
