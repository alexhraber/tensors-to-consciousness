# TUI Guide

The TUI is the primary operator interface.

Launch:

```bash
explorer
```

If no prior config exists, the default framework is:

- `mlx` on macOS
- `jax` on other platforms

## Operator Controls

- `Up/Down`: move transform cursor
- `Space`: include/exclude focused transform from the active chain
- `[` / `]`: shift selected transform precedence up/down
- `Left/Right`: switch framework
- `p`: run/pause (paused by default)
- `Esc`: return to landing
- `q`: quit

## Operational Pattern

The explorer is a continuous view:

1. select transforms
2. order precedence
3. tune parameters
4. execute and inspect output

## Start With a Preselected Chain

```bash
explorer tui --framework jax --transforms gradient_descent,momentum,adam
```

- `--transforms`: comma-separated keys
