# TUI + CLI Guide

This project is operated through `main.py`.

## Fast Start

```bash
python main.py
```

`main.py` will auto-detect framework setup, prompt if needed, validate, and launch the TUI studio.

## TUI Controls

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

## Direct Fetch (Module + Algorithm)

```bash
python main.py viz --module 2 --algorithm momentum
```

Selectors:
- `--module`: `0..6` or title fragment
- `--algorithm`: algorithm key or title fragment in selected module

## Inputs Override

```bash
python main.py all --framework jax --inputs inputs.example.json
```

`--inputs` supports:
- a JSON file path
- an inline JSON string

## CLI Operations

```bash
python main.py --help
python main.py -c
python main.py validate
python main.py viz --module 4 --algorithm attention_surface
python main.py all --framework jax --inputs inputs.example.json
```

## Tests

```bash
python -m tests
python -m tests --suite unit
python -m tests --suite integration
```

