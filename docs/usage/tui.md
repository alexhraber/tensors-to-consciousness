# TUI Guide

The primary experience is the interactive explorer:

```bash
python main.py
```

`main.py` auto-detects setup, validates, and launches the TUI.
If no framework config exists yet, it defaults to `numpy` (no onboarding prompt).

## Controls

- `n` / `b`: move transform cursor next/back
- `x`: toggle transform include/exclude in active chain
- `[` / `]`: move selected transform up/down in chain precedence
- `f`: framework selector modal
- `p`: compute platform selector (`cpu` / `gpu`)
- `i`: guided parameter input
- `e`: quick `key=value` parameter edit
- `space`: pause/resume live dynamics motion
- `:`: command console (`run`, `set`, `show`, `help`)
- `h`: toggle compact/full detail layout
- `r`: reseed
- `q`: quit

## Explorer Flow

The explorer runs as one continuous view.
Transform chains, parameter edits, and live dynamics are toggles within the same screen.

## Start with a Specific Transform Set

```bash
python main.py viz --transforms gradient_descent,momentum,adam --transform momentum
```

Top 5 core transforms:
- `tensor_ops`
- `chain_rule`
- `gradient_descent`
- `momentum`
- `adam`

- `--transforms`: comma-separated transform keys, or `default`/`all`
- `--transform`: key/title to focus initially

For non-interactive command usage, see `docs/usage/cli.md`.
