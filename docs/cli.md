# CLI Guide

This project is operated through `main.py`.

For interactive studio usage, see `docs/tui.md`.

## Core Commands

```bash
python main.py --help
python main.py -c
python main.py validate
python main.py all --framework jax
```

## Direct Visualization Fetch

```bash
python main.py viz --module 2 --algorithm momentum
```

- `--module`: `0..6` or title fragment
- `--algorithm`: algorithm key or title fragment in selected module

## Input Overrides

```bash
python main.py all --framework jax --inputs inputs.example.json
```

`--inputs` supports:
- a JSON file path
- an inline JSON string

## Tests

```bash
python -m tests
python -m tests --suite unit
python -m tests --suite integration
```
