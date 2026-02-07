# Container + SSH Usage

This project can run fully inside Docker while keeping the interactive explorer usable over SSH terminal sessions.

## 1) Build the image

```bash
docker compose build explorer
```

## 2) Run the explorer in container

```bash
docker compose run --rm explorer
```

Notes:

- `stdin_open: true` + `tty: true` are enabled for interactive TUI operation.
- The repository is bind-mounted into `/workspace`.
- `.config/` is persisted via the `t2c_config` volume.
- `explorer` service defaults to `python main.py`; passing extra commands overrides that default.

## 3) Run with GPU passthrough

NVIDIA:

```bash
docker compose --profile nvidia run --rm explorer-nvidia
```

AMD ROCm:

```bash
docker compose --profile amd run --rm explorer-amd
```

Intel iGPU:

```bash
docker compose --profile intel run --rm explorer-intel
```

Apple Silicon (MLX container path):

```bash
docker compose --profile apple run --rm explorer-apple
```

Pass-through details:

- `explorer-nvidia` uses `gpus: all` + `/dev/dri` with NVIDIA runtime env vars.
- `explorer-amd` uses `/dev/dri` + `/dev/kfd` (ROCm-compatible hosts).
- `explorer-intel` uses `/dev/dri`.
- `explorer-apple` runs `linux/arm64` and initializes MLX using `mlx[cpu]` inside container.
- Apple Metal passthrough is not currently available via Docker Desktop Linux containers, so MLX in container mode is CPU-backed.

Example with explicit device selection:

```bash
NVIDIA_VISIBLE_DEVICES=0 docker compose --profile nvidia run --rm explorer-nvidia
```

## 4) Run over SSH (including Telescope/terminal multiplexers)

From your local machine:

```bash
ssh <host> "cd /path/to/tensors-to-consciousness && docker compose run --rm explorer"
```

GPU over SSH:

```bash
ssh <host> "cd /path/to/tensors-to-consciousness && docker compose --profile nvidia run --rm explorer-nvidia"
```

If your terminal path cannot display advanced render styles well, force an ASCII rendering fallback:

```bash
ssh <host> "cd /path/to/tensors-to-consciousness && docker compose run --rm -e RENDER_STYLE=ascii explorer"
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
