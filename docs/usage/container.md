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

## 3) Run with GPU passthrough

GPU-enabled service:

```bash
docker compose --profile gpu run --rm explorer-gpu python main.py
```

Pass-through details:

- `gpus: all` for NVIDIA container runtime support.
- `/dev/dri` device mapping for common direct-render paths.
- `NVIDIA_VISIBLE_DEVICES` and `NVIDIA_DRIVER_CAPABILITIES` are set with safe defaults and can be overridden.

Example with explicit device selection:

```bash
NVIDIA_VISIBLE_DEVICES=0 docker compose --profile gpu run --rm explorer-gpu python main.py
```

## 4) Run over SSH (including Telescope/terminal multiplexers)

From your local machine:

```bash
ssh <host> "cd /path/to/tensors-to-consciousness && docker compose run --rm explorer python main.py"
```

GPU over SSH:

```bash
ssh <host> "cd /path/to/tensors-to-consciousness && docker compose --profile gpu run --rm explorer-gpu python main.py"
```

If your terminal path cannot display advanced render styles well, force an ASCII rendering fallback:

```bash
ssh <host> "cd /path/to/tensors-to-consciousness && RENDER_STYLE=ascii docker compose run --rm explorer python main.py"
```

Telescope note:

- This workflow is compatible with Telescope-driven remote terminals because the container run is TTY-attached (`stdin_open: true`, `tty: true`).

## 5) One-off command examples

```bash
docker compose run --rm explorer python main.py --list-transforms
docker compose run --rm explorer python main.py run --framework numpy --transforms default
docker compose run --rm explorer python -m tests --suite unit
```

## 6) Cleanup

```bash
docker compose down --volumes
```
