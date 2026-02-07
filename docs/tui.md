# TUI Guide

The primary experience is the interactive studio:

```bash
python main.py
```

`main.py` auto-detects setup, validates, and launches the TUI.
If no framework config exists yet, it defaults to `numpy` (no onboarding prompt).

## Controls

- `m` / `M`: cycle mode forward/reverse (`simple`, `advanced`, `ultra`)
- `n` / `b`: module profile next/back (`0..6`)
- `a` / `A`: algorithm next/back within the active module
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

## Start at a Specific Module + Algorithm

```bash
python main.py viz --module 2 --algorithm momentum
```

- `--module`: `0..6` or title fragment
- `--algorithm`: algorithm key or title fragment in selected module

For non-interactive command usage, see `docs/cli.md`.
