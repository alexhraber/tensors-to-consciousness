# TUI Guide

The TUI is the primary operator interface.

Launch:

```bash
python explorer.py
```

If no prior config exists, the default framework is:

- `mlx` on macOS
- `jax` on other platforms

## Operator Controls

- `n` / `b`: move transform cursor next/back
- `x`: include or exclude focused transform from the active chain
- `[` / `]`: shift selected transform precedence up/down
- `f`: open framework selector
- `p`: open compute platform selector (`cpu` / `gpu`)
- `i`: guided parameter input
- `e`: quick `key=value` parameter edit
- `space`: pause/resume live dynamics
- `:`: command console (`run`, `set`, `show`, `help`)
- `h`: toggle compact/full detail layout
- `r`: reseed
- `q`: quit

## Operational Pattern

The explorer is a continuous view:

1. select transforms
2. order precedence
3. tune parameters
4. execute and inspect output

## Start With a Preselected Chain

```bash
python explorer.py render --transforms gradient_descent,momentum,adam --transform momentum
```

- `--transforms`: comma-separated keys, `default`, or `all`
- `--transform`: initial focused key or title fragment
