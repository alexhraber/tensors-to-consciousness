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
  <a href="https://github.com/alexhraber/tensors-to-consciousness/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/alexhraber/tensors-to-consciousness/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.14" src="https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white">
  <a href="https://deepwiki.com/alexhraber/tensors-to-consciousness"><img alt="Ask DeepWiki" src="https://deepwiki.com/badge.svg"></a>
</p>

</div>

---

## Rendering Preview

<p align="center">
  <img src="assets/render/optimization_flow.gif" alt="Optimization flow rendering" width="32%">
  <img src="assets/render/attention_dynamics.gif" alt="Attention dynamics rendering" width="32%">
  <img src="assets/render/phase_portraits.gif" alt="Phase portrait rendering" width="32%">
</p>

## Research-Grade Terminal Explorer

`tensors-to-consciousness` centers on an interactive terminal environment for rigorous mathematical rendering, comparative framework analysis, and exploratory AI/ML experimentation.

Primary entrypoint:

```bash
python main.py
```

<p align="center">
  <img src="assets/render/tui_explorer.gif" alt="TUI explorer preview" width="86%">
</p>

On first run it will:

1. Detect framework config
2. Use default framework `numpy` if missing
3. Auto-setup environment + latest available dependencies
4. Validate the framework track
5. Launch the interactive rendering explorer

Detailed usage lives in:

- [Docs Index](docs/README.md)
- [TUI Guide](docs/usage/tui.md)
- [CLI Guide](docs/usage/cli.md)
- [Container + SSH Guide](docs/usage/container.md)
- [Transforms Reference](docs/reference/transforms.md)
- [Frameworks Reference](docs/reference/frameworks.md)

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

## References

- [Transforms Reference](docs/reference/transforms.md)
- [Transform Playbook](docs/guides/transform-playbook.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Tests Guide](tests/README.md)
- [CI Workflow](.github/workflows/ci.yml)

Core exploration commands:

```bash
python main.py --list-transforms
python main.py run --framework jax --transforms chain_rule,gradient_descent,adam
python main.py render --framework numpy --transforms default
```

Diagnostics and introspection:

- Default logging level is `INFO`.
- Debug tracing is disabled by default.
- Set `DEBUG=1` to emit kernel-level transform/tooling events.
- Set `LOG_LEVEL` to one of: `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`.
- You can also set defaults in `.config/config.json` via `diagnostics.log_level` and `diagnostics.debug`.
- Environment variables override config values.
- The config file is normalized with sensible defaults for `platform` and `diagnostics` when written by tooling.

```json
{
  "framework": "numpy",
  "venv": ".venv-numpy",
  "diagnostics": {
    "log_level": "INFO",
    "debug": false
  }
}
```

```bash
DEBUG=1 LOG_LEVEL=DEBUG python main.py run --framework numpy --transforms default
```

## Shinkei Visual Samples

<p align="center"><strong>Sample A</strong></p>
<p align="center">
  <img src="assets/render/optimization_flow.gif" alt="Shinkei sample A - optimization flow" width="72%">
</p>

<p align="center"><strong>Sample B</strong></p>
<p align="center">
  <img src="assets/render/attention_dynamics.gif" alt="Shinkei sample B - attention dynamics" width="72%">
</p>

<p align="center"><strong>Sample C</strong></p>
<p align="center">
  <img src="assets/render/phase_portraits.gif" alt="Shinkei sample C - phase portraits" width="72%">
</p>

## Notes

- `python -m tests` validates CLI/setup/runtime operations.
- `numpy` and `cupy` use finite-difference gradients in autodiff-heavy sections.
- `keras` mixes gradient tape and numerical approximations in selected sections.

## Contributing

See [Contributing Guide](CONTRIBUTING.md).
