# CLI Guide

This project is operated through `main.py`.

For interactive studio usage, see `docs/usage/tui.md`.

## Core Commands

```bash
python main.py --help
python main.py -c
python main.py validate
python main.py run --framework jax --transforms default
python -m tools.playground --framework jax --transforms chain_rule,adam --viz
python main.py --list-transforms
```

Top 5 core transforms:
- `tensor_ops`
- `chain_rule`
- `gradient_descent`
- `momentum`
- `adam`

## Direct Visualization Fetch

```bash
python main.py viz --transforms gradient_descent,momentum --transform momentum
```

- `--transforms`: comma-separated transform keys, or `default`/`all`
- `--transform`: initial focused transform key/title for the TUI

## Input Overrides

```bash
python main.py run --framework jax --transforms chain_rule,adam --inputs examples/inputs.example.json
python main.py run --framework numpy --transforms spectral_filter,wave_propagation --inputs examples/inputs.spectral_sweep.json
python main.py run --framework pytorch --transforms constraint_projection,entropy_flow --inputs examples/inputs.stability_focus.json
python main.py run --framework keras --transforms reaction_diffusion,stochastic_process --inputs examples/inputs.noise_storm.json
```

`--inputs` supports:
- a JSON file path
- an inline JSON string

See `examples/README.md` for all curated presets.

## Tests

```bash
python -m tests
python -m tests --suite unit
python -m tests --suite integration
```
