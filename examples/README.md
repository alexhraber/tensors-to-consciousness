# Examples

Input presets for `--inputs` / `INPUTS`.

## Files

- `inputs.example.json`: balanced baseline for most runs.
- `inputs.spectral_sweep.json`: tighter ranges for spectral-style transforms.
- `inputs.stability_focus.json`: low-noise regime for stability and convergence checks.
- `inputs.noise_storm.json`: high-noise stress test.
- `inputs.framework_matrix.json`: deterministic per-framework comparison seeds.

## Usage

```bash
explorer run --framework jax --transforms chain_rule,momentum,adam --inputs examples/inputs.example.json
explorer run --framework numpy --transforms spectral_filter,wave_propagation --inputs examples/inputs.spectral_sweep.json
explorer run --framework pytorch --transforms constraint_projection,entropy_flow --inputs examples/inputs.stability_focus.json
explorer run --framework keras --transforms reaction_diffusion,stochastic_process --inputs examples/inputs.noise_storm.json
```

## Headless Progression Examples

These transform progressions back the three README render GIFs:

```bash
explorer run --framework numpy --transforms tensor_ops,chain_rule,gradient_descent,momentum,adam --inputs examples/inputs.example.json
explorer run --framework numpy --transforms forward_pass,activation_flow,attention_surface,attention_message_passing --inputs examples/inputs.framework_matrix.json
explorer run --framework numpy --transforms laplacian_diffusion,reaction_diffusion,spectral_filter,wave_propagation,entropy_flow --inputs examples/inputs.spectral_sweep.json
```

Inline JSON also works:

```bash
explorer run --framework numpy --transforms tensor_ops,chain_rule --inputs '{"seed": 9, "normal": {"std": 0.5}}'
```
