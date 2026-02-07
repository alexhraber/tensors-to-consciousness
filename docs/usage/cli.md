# CLI Guide

`explorer.py` is the canonical CLI entrypoint.

## Core Commands

```bash
python explorer.py --help
python explorer.py -c
python explorer.py validate
python explorer.py run --framework jax --transforms default
python explorer.py render --framework numpy --transforms gradient_descent,momentum --transform momentum
python explorer.py --list-transforms
```

## Inputs and Presets

```bash
python explorer.py run --framework jax --transforms chain_rule,adam --inputs examples/inputs.example.json
python explorer.py run --framework numpy --transforms spectral_filter,wave_propagation --inputs examples/inputs.spectral_sweep.json
python explorer.py run --framework pytorch --transforms constraint_projection,entropy_flow --inputs examples/inputs.stability_focus.json
python explorer.py run --framework keras --transforms reaction_diffusion,stochastic_process --inputs examples/inputs.noise_storm.json
```

`--inputs` accepts:

- JSON file path
- inline JSON blob

## Test Commands

```bash
python -m tests
python -m tests --suite unit
python -m tests --suite integration
```
