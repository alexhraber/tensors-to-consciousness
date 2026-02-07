# TUI Guide

The primary experience is the interactive studio:

```bash
python main.py
```

`main.py` auto-detects setup, validates, and launches the TUI.
If no framework config exists yet, it defaults to `numpy` (no onboarding prompt).

## Controls

- `m` / `M`: cycle mode forward/reverse (`simple`, `advanced`, `ultra`)
- `n` / `b`: algorithm next/back within selected algo set
- `a` / `A`: alias next/back for rapid cycling
- `f`: framework selector modal
- `p`: compute platform selector (`cpu` / `gpu`)
- `i`: guided parameter input
- `e`: quick `key=value` parameter edit
- `space`: pause/resume ultra-mode live motion
- `:`: command console (`run`, `set`, `view`, `show`, `help`)
- `r`: reseed
- `q`: quit

## Modes

- `simple`: preset/default visualization browsing with formula + description.
- `advanced`: tunable algorithm parameters with formula + description.
- `ultra`: advanced controls + live transformation motion.

## Start with a Specific Algo Set

```bash
python main.py viz --transforms gradient_descent,momentum,adam --algorithm momentum
```

Top 5 core transforms:
- `tensor_ops`
- `chain_rule`
- `gradient_descent`
- `momentum`
- `adam`

- `--transforms`: comma-separated transform keys, or `default`/`all`
- `--algorithm`: key/title to focus initially

For non-interactive command usage, see `docs/cli.md`.
