# Render Asset Generation

Render assets are generated from the real runtime path.

## Reproducible Generation

Progression GIFs are generated directly from the Shinkei render path (framework engine -> transforms -> Shinkei plot PNG -> GIF), so the output is tightly framed on the rendered visual.

TUI GIF capture is headless from the real interactive session using:

- `Xvfb`
- `xterm`
- `xdotool`
- `ffmpeg`

Three progression examples (used in README):

```bash
python tools/generate_render_assets.py --only optimization_flow attention_dynamics phase_portraits --framework numpy
```

Interactive TUI example:

```bash
python tools/generate_render_assets.py --only tui_explorer --tui-capture headless --framework numpy
```

## Full Asset Refresh

```bash
python tools/generate_render_assets.py --framework numpy --tui-capture headless
```

## Notes

- Use `--tui-capture synthetic` only as a fallback when headless system dependencies are unavailable.
- CI runs progression generation checks plus a headless TUI capture smoke check.
- `.github/workflows/assets-readme-sync.yml` regenerates and syncs README render assets when render scripts/utilities/dependencies change.
