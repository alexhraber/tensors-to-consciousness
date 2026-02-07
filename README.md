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
  <a href="https://github.com/alexhraber/tensors-to-consciousness/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/alexhraber/tensors-to-consciousness/actions/workflows/ci.yml/badge.svg"></a>
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

- `docs/tui.md`
- `docs/cli.md`

Toolchain baseline: Python `3.14` + `uv` (latest stable).

## Frameworks

The entire research track is implemented in each of the following frameworks:

- `mlx`
- `jax`
- `pytorch`
- `numpy`
- `keras`
- `cupy`

## Algorithm Layout

Algorithm implementations live under `algos/<framework>/`.
Framework backends live under `frameworks/<framework>/` and provide reusable ops/models utilities.
Algorithm ordering, complexity ranking, defaults, and execution mapping are centralized in `algos/registry.py`.

To scaffold a new abstract algorithm + backend adapters:

```bash
python tools/scaffold_algo.py --complexity 2 --key rk4_solver --title "RK4 Solver" --formula "x_{t+1}=x_t+..." --description "Fourth-order integration"
```

Sandbox-style runs use algorithm combinations instead of fixed `0..6` module targets:

```bash
python main.py run --framework jax --algos chain_rule,gradient_descent,adam
python main.py viz --framework jax --algos all
python main.py --list-algos
```

## Notes

- `python -m tests` validates CLI/setup/runtime operations.
- `numpy` and `cupy` use finite-difference gradients in autodiff-heavy sections.
- `keras` mixes gradient tape and numerical approximations in selected sections.

## Contributing

See `CONTRIBUTING.md`.
