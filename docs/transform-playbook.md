# Transform Playbook

This page holds practical transform usage details that are intentionally kept out of `README.md`.

## Full Catalog Sources

- Transform catalog JSON: `transforms/transforms.json`
- Generated transform reference: [Transforms Reference](transforms.md)
- Example inputs catalog: [examples/README.md](../examples/README.md)

## Generate Docs

```bash
python tools/generate_catalog_docs.py
```

## Scaffold a New Transform

```bash
python tools/scaffold_algo.py --complexity 2 --key rk4_solver --title "RK4 Solver" --formula "x_{t+1}=x_t+..." --description "Fourth-order integration"
```

## Pipeline Examples

```bash
python main.py run --framework jax --transforms chain_rule,gradient_descent,adam
python main.py viz --framework jax --transforms all
python main.py --list-transforms
python -m tools.playground --framework jax --transforms chain_rule,adam --viz
```

## Nuanced Preset Examples

```bash
python main.py run --framework jax --transforms chain_rule,momentum,adam --inputs examples/inputs.example.json
python main.py run --framework numpy --transforms spectral_filter,wave_propagation --inputs examples/inputs.spectral_sweep.json
python main.py run --framework pytorch --transforms constraint_projection,entropy_flow --inputs examples/inputs.stability_focus.json
python main.py run --framework keras --transforms reaction_diffusion,stochastic_process --inputs examples/inputs.noise_storm.json
python main.py run --framework mlx --transforms tensor_ops,chain_rule,gradient_descent --inputs examples/inputs.framework_matrix.json
```
