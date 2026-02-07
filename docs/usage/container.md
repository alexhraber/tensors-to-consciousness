# Container Guide

Run the platform fully in Docker while preserving TTY interactivity for remote terminals.

## Build

```bash
docker compose build explorer
```

## Launch

```bash
docker compose run --rm explorer
```

Service behavior:

- interactive TTY is enabled (`stdin_open: true`, `tty: true`)
- repository is mounted at `/workspace`
- configuration persists in a named volume
- service default command is `explorer`

## Hardware Profiles

```bash
docker compose --profile nvidia run --rm explorer-nvidia
docker compose --profile amd run --rm explorer-amd
docker compose --profile intel run --rm explorer-intel
docker compose --profile apple run --rm explorer-apple
```

Notes:

- `explorer-nvidia`: `gpus: all` plus NVIDIA runtime vars
- `explorer-amd`: `/dev/dri` + `/dev/kfd` for ROCm-compatible hosts
- `explorer-intel`: `/dev/dri`
- `explorer-apple`: `linux/arm64` with CPU-backed MLX in container mode

## SSH Usage

```bash
ssh <host> "cd /path/to/tensors-to-consciousness && docker compose run --rm explorer"
```

ASCII fallback for constrained terminals:

```bash
ssh <host> "cd /path/to/tensors-to-consciousness && docker compose run --rm -e RENDER_STYLE=ascii explorer"
```

## One-Off Commands

```bash
docker compose run --rm explorer explorer list-transforms
docker compose run --rm explorer explorer run --framework numpy --transforms default
docker compose run --rm explorer cargo test -p explorer
docker compose run --rm explorer python -m tests --suite unit
```
