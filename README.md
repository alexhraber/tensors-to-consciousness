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
2. Prompt user for framework if missing
3. Prompt for optional input overrides (JSON path or inline JSON)
4. Auto-setup environment + latest available dependencies
5. Validate the framework track
6. Launch the interactive visualization studio

Mode behavior:

- `simple`: browse default/preset visualizations for each script profile with formula + description.
- `advanced`: same profile context, but with tunable algorithm parameters.
- `ultra`: advanced controls plus live transformation motion.

Detailed controls, direct module/algorithm fetch, and command examples live in:

- `docs/tui.md`
- `docs/cli.md`

Toolchain baseline: Python `3.14` + `uv` (latest stable).

### Studio + Inputs

Default behavior stays random, but you can override generated inputs per framework/script/call:

```bash
python main.py all --framework jax --inputs inputs.example.json
```

You can also pass raw JSON via `--inputs`.

Config keys:

- `seed`: global seed
- `frameworks.<name>.seed`: framework seed override
- `normal` / `uniform`: framework-wide distribution tuning
- `frameworks.<name>.scripts.<script>.calls.<label|line>`: per-call overrides

See `inputs.example.json` for a working template.

## Frameworks

The entire research track is implemented in each of the following frameworks:

- `mlx`
- `jax`
- `pytorch`
- `numpy`
- `keras`
- `cupy`

Advanced targets available through `main.py`:

- `validate`
- `viz`
- `0..6`
- `all`

## Notes

- `python -m tests` validates CLI/setup/runtime operations.
- `numpy` and `cupy` use finite-difference gradients in autodiff-heavy sections.
- `keras` mixes gradient tape and numerical approximations in selected sections.

## Contributing

See `CONTRIBUTING.md`.
