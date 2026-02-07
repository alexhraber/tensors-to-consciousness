#!/usr/bin/env python3
"""Generate README visualization GIF assets using only stdlib + ffmpeg."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from pathlib import Path


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
    t = frame * 0.13
    buf = bytearray(width * height * 3)
    idx = 0
    for y in range(height):
        ny = (y / height) * 2.0 - 1.0
        for x in range(width):
            nx = (x / width) * 2.0 - 1.0
            v = math.sin(5.5 * nx + t) * math.cos(6.2 * ny - 0.7 * t)
            g = math.exp(-2.2 * (nx * nx + ny * ny))
            s = 0.5 + 0.5 * v
            r = clamp(25 + 180 * s + 50 * g)
            gc = clamp(18 + 110 * (1 - s) + 110 * g)
            b = clamp(40 + 180 * (0.5 + 0.5 * math.sin(3 * nx - 2 * ny + t)))
            buf[idx : idx + 3] = bytes((r, gc, b))
            idx += 3

    px, py = -0.9, 0.85
    for _ in range(120):
        gx = 5.5 * math.cos(5.5 * px + t) * math.cos(6.2 * py - 0.7 * t)
        gy = -6.2 * math.sin(5.5 * px + t) * math.sin(6.2 * py - 0.7 * t)
        px -= 0.012 * gx
        py -= 0.012 * gy
        xx = int((px + 1.0) * 0.5 * (width - 1))
        yy = int((py + 1.0) * 0.5 * (height - 1))
        if 1 <= xx < width - 1 and 1 <= yy < height - 1:
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    j = ((yy + oy) * width + (xx + ox)) * 3
                    buf[j : j + 3] = bytes((255, 245, 180))
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
    }

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
