# CLI Guide

`explorer` is the canonical entrypoint. Rust hosts the product surface; Python executes transforms and framework math.

## Core Commands

```bash
explorer --help
explorer list-transforms
explorer validate --framework jax
explorer run --framework jax --transforms default
```

## Inputs and Presets

```bash
explorer run --framework jax --transforms chain_rule,adam --inputs examples/inputs.example.json
explorer run --framework numpy --transforms spectral_filter,wave_propagation --inputs examples/inputs.spectral_sweep.json
explorer run --framework pytorch --transforms constraint_projection,entropy_flow --inputs examples/inputs.stability_focus.json
explorer run --framework keras --transforms reaction_diffusion,stochastic_process --inputs examples/inputs.noise_storm.json
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
