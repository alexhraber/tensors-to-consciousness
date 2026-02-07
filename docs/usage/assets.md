# Render Asset Generation

Render assets are generated from the real runtime path.

## Reproducible Headless Captures

All README GIFs are captured headlessly from real commands using:

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
- CI runs headless progression capture checks plus a headless TUI capture smoke check.
- `.github/workflows/assets-readme-sync.yml` regenerates and syncs README render assets when render scripts/utilities/dependencies change.
