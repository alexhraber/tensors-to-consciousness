#!/usr/bin/env python3
"""Generate README GIF assets using reproducible headless captures."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.headless_capture import (
    HeadlessCaptureError,
    capture_transform_progression_gif,
    capture_tui_session_gif,
)
from tools import shinkei


def clamp(v: float) -> int:
    if v < 0:
        return 0
    if v > 255:
        return 255
    return int(v)


def write_ppm(path: Path, width: int, height: int, rgb: bytearray) -> None:
    with path.open("wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(rgb)


def _render_pipeline_png(
    engine: FrameworkEngine,
    *,
    pipeline: tuple[str, ...],
    size: int,
    steps: int,
    stage: str,
    width_px: int,
    height_px: int,
) -> bytes:
    import numpy as np

    result = engine.run_pipeline(pipeline, size=size, steps=steps)
    arr = engine.to_numpy(result.final_tensor)
    if arr is None:
        raise RuntimeError("framework engine did not provide numpy-convertible tensor")
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e4, neginf=-1e4)
    arr = np.clip(arr, -1e4, 1e4)

    # Tone-map and smooth to keep previews calm and readable.
    arr2 = arr if arr.ndim == 2 else arr.reshape(arr.shape[0], -1)
    scale = float(np.percentile(np.abs(arr2), 92))
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
    arr2 = np.tanh(arr2 / scale) * 2.0

    padded = np.pad(arr2, ((1, 1), (1, 1)), mode="edge")
    smooth = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + (2.0 * padded[1:-1, 1:-1])
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 10.0
    arr = smooth.astype(np.float32)

    png = shinkei._matplotlib_plot_png(
        arr,
        stage=stage,
        tensor_name=f"{engine.framework}:{'+'.join(pipeline)}",
        width_px=width_px,
        height_px=height_px,
    )
    if png is None:
        raise RuntimeError("shinkei matplotlib renderer unavailable")
    return png


def _fill_rect(
    buf: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return
    r, g, b = color
    for y in range(y0, y1):
        row = (y * width + x0) * 3
        for _ in range(x0, x1):
            buf[row : row + 3] = bytes((r, g, b))
            row += 3


def _draw_hline(buf: bytearray, width: int, height: int, x0: int, x1: int, y: int, color: tuple[int, int, int]) -> None:
    _fill_rect(buf, width, height, x0, y, x1, y + 1, color)


def _draw_vline(buf: bytearray, width: int, height: int, x: int, y0: int, y1: int, color: tuple[int, int, int]) -> None:
    _fill_rect(buf, width, height, x, y0, x + 1, y1, color)


def _draw_frame(buf: bytearray, width: int, height: int, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    _draw_hline(buf, width, height, x0, x1, y0, color)
    _draw_hline(buf, width, height, x0, x1, y1 - 1, color)
    _draw_vline(buf, width, height, x0, y0, y1, color)
    _draw_vline(buf, width, height, x1 - 1, y0, y1, color)


def render_tui_explorer(frame: int, width: int, height: int) -> bytearray:
    t = frame * 0.09
    buf = bytearray(width * height * 3)

    idx = 0
    for y in range(height):
        ny = y / max(1, height - 1)
        for x in range(width):
            nx = x / max(1, width - 1)
            glow = 0.5 + 0.5 * math.sin(3.2 * nx + 2.4 * ny + 0.7 * t)
            r = clamp(7 + 12 * ny + 10 * glow)
            g = clamp(10 + 14 * ny + 14 * glow)
            b = clamp(18 + 22 * ny + 26 * glow)
            buf[idx : idx + 3] = bytes((r, g, b))
            idx += 3

    mx = max(18, int(width * 0.05))
    my = max(14, int(height * 0.08))
    x0, y0 = mx, my
    x1, y1 = width - mx, height - my
    _fill_rect(buf, width, height, x0, y0, x1, y1, (7, 10, 16))
    _draw_frame(buf, width, height, x0, y0, x1, y1, (70, 92, 126))

    header_h = max(18, int((y1 - y0) * 0.08))
    _fill_rect(buf, width, height, x0 + 1, y0 + 1, x1 - 1, y0 + header_h, (18, 26, 40))
    _fill_rect(buf, width, height, x0 + 16, y0 + 7, x0 + int((x1 - x0) * 0.45), y0 + 11, (92, 150, 222))

    body_y0 = y0 + header_h + 2
    body_y1 = y1 - max(28, int((y1 - y0) * 0.1))
    left_w = max(120, int((x1 - x0) * 0.36))
    split_x = x0 + left_w
    _draw_vline(buf, width, height, split_x, body_y0, body_y1, (50, 68, 94))

    row_h = max(10, int((body_y1 - body_y0) / 13))
    selected = (frame // 5) % 10
    enabled_count = 1 + ((frame // 7) % 5)
    for i in range(10):
        ry = body_y0 + 8 + i * row_h
        if ry + row_h - 2 >= body_y1:
            break
        on = i < enabled_count
        row_color = (14, 20, 30) if i != selected else (20, 34, 52)
        _fill_rect(buf, width, height, x0 + 8, ry, split_x - 8, ry + row_h - 2, row_color)
        cb_color = (92, 200, 132) if on else (95, 104, 124)
        _draw_frame(buf, width, height, x0 + 12, ry + 2, x0 + 20, ry + 10, cb_color)
        if on:
            _fill_rect(buf, width, height, x0 + 14, ry + 4, x0 + 18, ry + 8, cb_color)
        order_col = (230, 198, 112) if on else (116, 126, 144)
        _fill_rect(buf, width, height, x0 + 26, ry + 3, x0 + 34, ry + 7, order_col)
        _fill_rect(buf, width, height, x0 + 40, ry + 3, split_x - 18, ry + 6, (128, 146, 174))

    rx0 = split_x + 10
    rx1 = x1 - 10
    ry0 = body_y0 + 8
    ry1 = body_y1 - 8
    _draw_frame(buf, width, height, rx0 - 2, ry0 - 2, rx1 + 2, ry1 + 2, (62, 84, 114))
    w = max(1, rx1 - rx0)
    h = max(1, ry1 - ry0)
    for yy in range(h):
        ny = (yy / h) * 2.0 - 1.0
        for xx in range(w):
            nx = (xx / w) * 2.0 - 1.0
            band = math.sin(5.2 * nx + 1.6 * t) * math.cos(4.4 * ny - 1.2 * t)
            wave = math.sin(8.0 * (nx * nx + ny * ny) - 1.8 * t)
            v = 0.5 + 0.5 * (0.55 * band + 0.45 * wave)
            r = clamp(20 + 140 * v)
            g = clamp(24 + 100 * (1.0 - abs(0.5 - v) * 2.0))
            b = clamp(48 + 170 * (1.0 - v))
            j = ((ry0 + yy) * width + (rx0 + xx)) * 3
            buf[j : j + 3] = bytes((r, g, b))

    foot_y0 = body_y1 + 6
    foot_y1 = y1 - 8
    _fill_rect(buf, width, height, x0 + 8, foot_y0, x1 - 8, foot_y1, (14, 20, 32))
    _fill_rect(buf, width, height, x0 + 18, foot_y0 + 4, x0 + 96, foot_y0 + 7, (90, 160, 230))
    _fill_rect(buf, width, height, x0 + 102, foot_y0 + 4, x0 + int((x1 - x0) * 0.78), foot_y0 + 7, (126, 136, 156))
    if frame % 8 < 4:
        cx = x0 + int((x1 - x0) * 0.78) + 4
        _fill_rect(buf, width, height, cx, foot_y0 + 3, cx + 2, foot_y0 + 9, (240, 244, 252))

    return buf


def encode_gif(frames_dir: Path, output_gif: Path, fps: int, pattern: str) -> None:
    palette = frames_dir / "palette.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / pattern),
            "-vf",
            f"fps={fps},scale=960:-1:flags=lanczos,palettegen",
            str(palette),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / pattern),
            "-i",
            str(palette),
            "-lavfi",
            f"fps={fps},scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse",
            str(output_gif),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GIF previews in assets/render.")
    parser.add_argument("--output-dir", default="assets/render", help="Output directory.")
    parser.add_argument("--frames", type=int, default=64, help="Frames per GIF.")
    parser.add_argument("--fps", type=int, default=18, help="GIF fps.")
    parser.add_argument("--width", type=int, default=480, help="Frame width.")
    parser.add_argument("--height", type=int, default=270, help="Frame height.")
    parser.add_argument("--framework", default="numpy", help="Framework backend used for pipeline renders.")
    parser.add_argument(
        "--tui-capture",
        choices=["headless", "synthetic"],
        default="headless",
        help="TUI asset source. 'headless' captures real TUI via Xvfb/ffmpeg (default).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["optimization_flow", "attention_dynamics", "phase_portraits", "tui_explorer"],
        help="Generate only selected assets.",
    )
    return parser.parse_args()


def main() -> int:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but not found in PATH.")

    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    progression_specs: dict[str, dict[str, str | None]] = {
        "optimization_flow": {
            "transforms": "tensor_ops,chain_rule,gradient_descent,momentum,adam",
            "inputs": "examples/inputs.example.json",
        },
        "attention_dynamics": {
            "transforms": "forward_pass,activation_flow,attention_surface,attention_message_passing",
            "inputs": "examples/inputs.framework_matrix.json",
        },
        "phase_portraits": {
            "transforms": "laplacian_diffusion,reaction_diffusion,spectral_filter,wave_propagation,entropy_flow",
            "inputs": "examples/inputs.spectral_sweep.json",
        },
    }

    all_assets = ["optimization_flow", "attention_dynamics", "phase_portraits", "tui_explorer"]
    selected = args.only if args.only else all_assets

    with tempfile.TemporaryDirectory(prefix="t2c_render_") as td:
        tmp = Path(td)

        for name in selected:
            if name == "tui_explorer":
                frames_dir = tmp / name
                frames_dir.mkdir(parents=True, exist_ok=True)
                output_gif = out_dir / f"{name}.gif"
                if args.tui_capture == "headless":
                    try:
                        capture_tui_session_gif(
                            output_gif=output_gif,
                            python_exe=sys.executable,
                            framework=args.framework,
                            transforms="default",
                            width=max(1280, args.width * 2),
                            height=max(720, args.height * 2),
                            fps=args.fps,
                            duration_s=max(6.0, min(14.0, args.frames / max(1, args.fps))),
                        )
                    except HeadlessCaptureError as exc:
                        raise RuntimeError(str(exc)) from exc
                else:
                    for i in range(args.frames):
                        rgb = render_tui_explorer(i, args.width, args.height)
                        write_ppm(frames_dir / f"{i:03d}.ppm", args.width, args.height, rgb)
                    encode_gif(frames_dir, output_gif, args.fps, "%03d.ppm")
                print(f"wrote {output_gif}")
                continue

            spec = progression_specs[name]
            try:
                capture_transform_progression_gif(
                    output_gif=out_dir / f"{name}.gif",
                    python_exe=sys.executable,
                    framework=args.framework,
                    transforms=str(spec["transforms"]),
                    inputs=spec["inputs"],
                    width=max(1280, args.width * 2),
                    height=max(720, args.height * 2),
                    fps=args.fps,
                    duration_s=max(6.0, min(14.0, args.frames / max(1, args.fps))),
                    title=f"ttc-{name}",
                )
            except HeadlessCaptureError as exc:
                raise RuntimeError(str(exc)) from exc
            print(f"wrote {out_dir / f'{name}.gif'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
