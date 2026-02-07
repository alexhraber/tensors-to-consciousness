#!/usr/bin/env python3
"""Generate README visualization GIF assets using only stdlib + ffmpeg."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from pathlib import Path


def _fill_rect(buf: bytearray, width: int, height: int, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
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


def render_optimization_flow(frame: int, width: int, height: int) -> bytearray:
    t = frame * 0.075
    buf = bytearray(width * height * 3)

    # Premium dark base with moving contour-like energy bands.
    idx = 0
    for y in range(height):
        ny = (y / height) * 2.0 - 1.0
        for x in range(width):
            nx = (x / width) * 2.0 - 1.0
            radial = math.sqrt(nx * nx + ny * ny)
            swirl = math.sin(7.5 * radial - 2.2 * t + 2.8 * math.atan2(ny, nx))
            waves = 0.45 * math.sin(4.6 * nx + 1.1 * t) + 0.55 * math.cos(5.0 * ny - 1.4 * t)
            field = 0.5 + 0.5 * (0.62 * swirl + 0.38 * waves)
            glow = math.exp(-3.2 * radial * radial)

            r = clamp(10 + 30 * glow + 130 * (field**1.7))
            gc = clamp(12 + 95 * glow + 160 * ((1.0 - abs(0.5 - field) * 2.0) ** 1.4))
            b = clamp(22 + 145 * glow + 165 * ((1.0 - field) ** 1.1))
            buf[idx : idx + 3] = bytes((r, gc, b))
            idx += 3

    # Multi-start optimizer trajectories flowing into the center basin.
    seeds = [(-0.92, 0.86), (-0.85, -0.78), (0.88, 0.74), (0.74, -0.88), (0.02, 0.92)]
    for sidx, (sx, sy) in enumerate(seeds):
        px, py = sx, sy
        for step in range(170):
            gx = (
                1.15 * px
                + 0.75 * math.cos(4.6 * px + 1.1 * t)
                - 0.48 * math.sin(2.8 * py - 1.0 * t)
            )
            gy = (
                1.05 * py
                - 0.72 * math.sin(5.0 * py - 1.4 * t)
                + 0.43 * math.cos(3.1 * px + 0.7 * t)
            )
            lr = 0.010 + 0.0015 * sidx
            px -= lr * gx
            py -= lr * gy
            xx = int((px + 1.0) * 0.5 * (width - 1))
            yy = int((py + 1.0) * 0.5 * (height - 1))
            if 1 <= xx < width - 1 and 1 <= yy < height - 1:
                j = (yy * width + xx) * 3
                age = step / 169.0
                trail = (
                    clamp(120 + 120 * (1.0 - age)),
                    clamp(205 + 40 * (1.0 - age)),
                    clamp(255),
                )
                buf[j] = max(buf[j], trail[0])
                buf[j + 1] = max(buf[j + 1], trail[1])
                buf[j + 2] = max(buf[j + 2], trail[2])

    return buf


def render_attention_dynamics(frame: int, width: int, height: int) -> bytearray:
    t = frame * 0.09
    centers = [
        (0.35 * math.cos(t), 0.25 * math.sin(1.6 * t), 0.28),
        (0.5 * math.cos(0.7 * t + 1.1), 0.35 * math.sin(1.2 * t + 2.4), 0.24),
        (0.6 * math.cos(1.2 * t + 3.0), 0.45 * math.sin(0.8 * t + 1.8), 0.20),
    ]

    buf = bytearray(width * height * 3)
    idx = 0
    for y in range(height):
        ny = (y / height) * 2.0 - 1.0
        for x in range(width):
            nx = (x / width) * 2.0 - 1.0
            val = 0.0
            for cx, cy, sig in centers:
                dx = nx - cx
                dy = ny - cy
                val += math.exp(-(dx * dx + dy * dy) / (2 * sig * sig))
            val = min(val, 1.6) / 1.6
            r = clamp(20 + 235 * val)
            gc = clamp(10 + 140 * (val**1.8))
            b = clamp(40 + 220 * ((1.0 - val) ** 1.2))
            buf[idx : idx + 3] = bytes((r, gc, b))
            idx += 3

    for y in range(0, height, 27):
        for x in range(width):
            j = (y * width + x) * 3
            buf[j] = min(255, buf[j] + 20)
            buf[j + 1] = min(255, buf[j + 1] + 20)
    for x in range(0, width, 32):
        for y in range(height):
            j = (y * width + x) * 3
            buf[j + 2] = min(255, buf[j + 2] + 24)
    return buf


def render_phase_portraits(frame: int, width: int, height: int) -> bytearray:
    t = frame * 0.08
    buf = bytearray([8, 10, 18] * (width * height))
    for k in range(6):
        color = (
            clamp(120 + 120 * math.sin(k * 0.8 + t)),
            clamp(120 + 120 * math.sin(k * 1.1 + t + 2.1)),
            clamp(120 + 120 * math.sin(k * 1.3 + t + 4.2)),
        )
        for i in range(900):
            u = i / 120.0
            x = math.sin((2.0 + k * 0.25) * u + t * (0.8 + 0.1 * k))
            y = math.sin((3.0 + k * 0.18) * u + 0.9 * t + k)
            xx = int((x + 1) * 0.5 * (width - 1))
            yy = int((y + 1) * 0.5 * (height - 1))
            j = (yy * width + xx) * 3
            buf[j] = max(buf[j], color[0])
            buf[j + 1] = max(buf[j + 1], color[1])
            buf[j + 2] = max(buf[j + 2], color[2])
    return buf


def render_tui_studio(frame: int, width: int, height: int) -> bytearray:
    t = frame * 0.09
    buf = bytearray(width * height * 3)

    # Background.
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

    # Terminal frame.
    mx = max(18, int(width * 0.05))
    my = max(14, int(height * 0.08))
    x0, y0 = mx, my
    x1, y1 = width - mx, height - my
    _fill_rect(buf, width, height, x0, y0, x1, y1, (7, 10, 16))
    _draw_frame(buf, width, height, x0, y0, x1, y1, (70, 92, 126))

    # Header bar.
    header_h = max(18, int((y1 - y0) * 0.08))
    _fill_rect(buf, width, height, x0 + 1, y0 + 1, x1 - 1, y0 + header_h, (18, 26, 40))
    _fill_rect(buf, width, height, x0 + 16, y0 + 7, x0 + int((x1 - x0) * 0.45), y0 + 11, (92, 150, 222))

    # Layout panes.
    body_y0 = y0 + header_h + 2
    body_y1 = y1 - max(28, int((y1 - y0) * 0.1))
    left_w = max(120, int((x1 - x0) * 0.36))
    split_x = x0 + left_w
    _draw_vline(buf, width, height, split_x, body_y0, body_y1, (50, 68, 94))

    # Left panel rows with moving cursor and checkboxes.
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

    # Right visualization panel with animated field.
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

    # Footer command bar.
    foot_y0 = body_y1 + 6
    foot_y1 = y1 - 8
    _fill_rect(buf, width, height, x0 + 8, foot_y0, x1 - 8, foot_y1, (14, 20, 32))
    _fill_rect(buf, width, height, x0 + 18, foot_y0 + 4, x0 + 96, foot_y0 + 7, (90, 160, 230))
    _fill_rect(buf, width, height, x0 + 102, foot_y0 + 4, x0 + int((x1 - x0) * 0.78), foot_y0 + 7, (126, 136, 156))
    if frame % 8 < 4:
        cx = x0 + int((x1 - x0) * 0.78) + 4
        _fill_rect(buf, width, height, cx, foot_y0 + 3, cx + 2, foot_y0 + 9, (240, 244, 252))

    return buf


def encode_gif(frames_dir: Path, output_gif: Path, fps: int) -> None:
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
            str(frames_dir / "%03d.ppm"),
            "-vf",
            "fps=%d,scale=960:-1:flags=lanczos,palettegen" % fps,
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
            str(frames_dir / "%03d.ppm"),
            "-i",
            str(palette),
            "-lavfi",
            "fps=%d,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse" % fps,
            str(output_gif),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GIF previews in assets/viz.")
    parser.add_argument("--output-dir", default="assets/viz", help="Output directory.")
    parser.add_argument("--frames", type=int, default=64, help="Frames per GIF.")
    parser.add_argument("--fps", type=int, default=18, help="GIF fps.")
    parser.add_argument("--width", type=int, default=480, help="Frame width.")
    parser.add_argument("--height", type=int, default=270, help="Frame height.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["optimization_flow", "attention_dynamics", "phase_portraits", "tui_studio"],
        help="Generate only selected assets.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but not found in PATH.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    renderers = {
        "optimization_flow": render_optimization_flow,
        "attention_dynamics": render_attention_dynamics,
        "phase_portraits": render_phase_portraits,
        "tui_studio": render_tui_studio,
    }
    if args.only:
        renderers = {name: renderers[name] for name in args.only}

    with tempfile.TemporaryDirectory(prefix="t2c_viz_") as td:
        tmp = Path(td)
        for name, renderer in renderers.items():
            frames_dir = tmp / name
            frames_dir.mkdir(parents=True, exist_ok=True)
            for i in range(args.frames):
                rgb = renderer(i, args.width, args.height)
                write_ppm(frames_dir / f"{i:03d}.ppm", args.width, args.height, rgb)
            encode_gif(frames_dir, out_dir / f"{name}.gif", args.fps)
            print(f"wrote {out_dir / f'{name}.gif'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
