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
python explorer.py run --framework jax --transforms chain_rule,momentum,adam --inputs examples/inputs.example.json
python explorer.py run --framework numpy --transforms spectral_filter,wave_propagation --inputs examples/inputs.spectral_sweep.json
python explorer.py run --framework pytorch --transforms constraint_projection,entropy_flow --inputs examples/inputs.stability_focus.json
python explorer.py run --framework keras --transforms reaction_diffusion,stochastic_process --inputs examples/inputs.noise_storm.json
```

Inline JSON also works:

```bash
python explorer.py run --framework numpy --transforms tensor_ops,chain_rule --inputs '{"seed": 9, "normal": {"std": 0.5}}'
```
