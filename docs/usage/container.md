# Container + SSH Usage

This project can run fully inside Docker while keeping the interactive explorer usable over SSH terminal sessions.

## 1) Build the image

```bash
docker compose build explorer
```

## 2) Run the explorer in container

```bash
docker compose run --rm explorer python main.py
```

Notes:

- `stdin_open: true` + `tty: true` are enabled for interactive TUI operation.
- The repository is bind-mounted into `/workspace`.
- `.config/` is persisted via the `t2c_config` volume.

## 3) Run over SSH (including Telescope/terminal multiplexers)

From your local machine:

```bash
ssh <host> "cd /path/to/tensors-to-consciousness && docker compose run --rm explorer python main.py"
```

If your terminal path cannot display advanced render styles well, force an ASCII rendering fallback:

```bash
ssh <host> "cd /path/to/tensors-to-consciousness && RENDER_STYLE=ascii docker compose run --rm explorer python main.py"
```

Telescope note:

- This workflow is compatible with Telescope-driven remote terminals because the container run is TTY-attached (`stdin_open: true`, `tty: true`).

## 4) One-off command examples

```bash
docker compose run --rm explorer python main.py --list-transforms
docker compose run --rm explorer python main.py run --framework numpy --transforms default
docker compose run --rm explorer python -m tests --suite unit
```

## 5) Cleanup

```bash
docker compose down --volumes
```
