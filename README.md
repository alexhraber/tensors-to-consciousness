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
  <img alt="Rust stable" src="https://img.shields.io/badge/Rust-stable-000000?logo=rust&logoColor=white">
</p>

</div>

---

## Tensor Exploration Platform

`tensors-to-consciousness` is an interactive terminal system for exploring tensor transforms across multiple compute frameworks.

## Quickstart

```bash
docker compose build explorer
docker compose run --rm explorer
```

Local:

```bash
mise install
mise run install-test-deps
mise run build
./target/debug/explorer
```

## Explorer Preview

<p align="center">
  <img src="assets/render/tui_explorer.gif" alt="TUI explorer preview" width="88%">
</p>

## Shinkei Render Samples

<p align="center">
  <img src="assets/render/optimization_flow.gif" alt="Optimization flow sample" width="32%">
  <img src="assets/render/attention_dynamics.gif" alt="Attention dynamics sample" width="32%">
  <img src="assets/render/phase_portraits.gif" alt="Phase portrait sample" width="32%">
</p>

Asset regeneration:

```bash
mise run assets-regenerate
```

## Documentation

- [Documentation Index](docs/README.md)
- [Architecture and Contracts](docs/reference/architecture.md)
- [TUI Guide](docs/usage/tui.md)
- [CLI Guide](docs/usage/cli.md)
- [Container Guide](docs/usage/container.md)
- [Transform Catalog](docs/reference/transforms.md)
- [Supported Frameworks](docs/reference/frameworks.md)
- [Contributing](CONTRIBUTING.md)

## Contributors

<p align="center">
  <img src="https://contrib.rocks/image?repo=alexhraber/tensors-to-consciousness" alt="Contributors screenshot" width="5%">
</p>

## License

[MIT](LICENSE)
