# CLI Guide

This project is operated through `main.py`.

For interactive studio usage, see `docs/tui.md`.

## Core Commands

```bash
python main.py --help
python main.py -c
python main.py validate
python main.py run --framework jax --algos default
python main.py --list-algos
```

## Direct Visualization Fetch

```bash
python main.py viz --algos gradient_descent,momentum --algorithm momentum
```

- `--algos`: comma-separated algorithm keys, or `default`/`all`
- `--algorithm`: initial focused algorithm key/title for the TUI

## Input Overrides

```bash
python main.py run --framework jax --algos chain_rule,adam --inputs examples/inputs.example.json
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
