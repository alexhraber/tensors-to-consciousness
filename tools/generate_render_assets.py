#!/usr/bin/env python3
"""Generate README GIF assets from Shinkei render output and headless TUI capture."""

from __future__ import annotations

import argparse
import math
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.headless_capture import (
    HeadlessCaptureError,
    capture_tui_session_gif,
)
from frameworks.engine import FrameworkEngine
from tools import input_controls
from transforms.contracts import TensorField
from transforms.definitions import get_transform_definition


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


def _render_pipeline_ppm(
    engine: FrameworkEngine,
    tensor: Any,
    *,
    pipeline: tuple[str, ...],
    stage: str,
    width_px: int,
    height_px: int,
) -> bytes:
    import numpy as np
    import base64

    arr = engine.to_numpy(tensor)
    if arr is None:
        raise RuntimeError("framework engine did not provide numpy-convertible tensor")
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e4, neginf=-1e4)
    arr = np.clip(arr, -1e4, 1e4)

    arr2 = arr if arr.ndim == 2 else arr.reshape(arr.shape[0], -1)
    h, w = arr2.shape
    data_b64 = base64.b64encode(arr2.astype(np.float32).tobytes(order="C")).decode("ascii")

    # Render via Rust (Shinkei) so the README assets always represent the actual product path.
    explorer_bin = os.environ.get("EXPLORER_BIN", "").strip() or shutil.which("explorer")
    if not explorer_bin:
        # Local dev fallback (repo checkout)
        candidate = ROOT / "target" / "debug" / "explorer"
        explorer_bin = str(candidate) if candidate.exists() else None
    if not explorer_bin:
        raise RuntimeError("explorer binary not found (set EXPLORER_BIN or build ./target/debug/explorer)")

    cmd = [
        explorer_bin,
        "render-tensor",
        "--h",
        str(int(h)),
        "--w",
        str(int(w)),
        "--data-b64",
        data_b64,
        "--width-px",
        str(int(width_px)),
        "--height-px",
        str(int(height_px)),
        "--out",
        "-",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.stdout


def _advance_tensor_field(
    engine: FrameworkEngine,
    *,
    field: TensorField,
    pipeline: tuple[str, ...],
    index: int,
    ops_per_frame: int,
) -> tuple[TensorField, int]:
    ops = engine._Ops(engine)
    engine._validate_ops_adapter(ops)
    for _ in range(max(1, int(ops_per_frame))):
        key = pipeline[index % len(pipeline)]
        definition = get_transform_definition(key)
        params = {**definition.defaults, **engine._params_for(key)}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow encountered.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")
            field = definition.transform(field, ops, params)
        if engine.framework == "numpy":
            import numpy as np

            arr = np.asarray(field.tensor, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e4, neginf=-1e4)
            field.tensor = np.clip(arr, -1e4, 1e4)
        index += 1
    return field, index


def _stable_seed(*parts: str) -> int:
    payload = "|".join(parts).encode("utf-8")
    # 32-bit seed for numpy default_rng portability.
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _reset_framework_rng(engine: FrameworkEngine, *, seed: int) -> None:
    # Asset generation must be deterministic for CI drift checks. The numpy backend
    # uses a module-global RNG; reset it per asset so outputs do not depend on
    # generation order or prior runs within the same Python process.
    if engine.framework != "numpy":
        return
    try:
        import numpy as np
        import importlib

        mod = importlib.import_module("frameworks.numpy.utils")
        if hasattr(mod, "RNG"):
            mod.RNG = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    except Exception:
        return


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


def encode_gif(frames_dir: Path, output_gif: Path, fps: int, pattern: str, *, width_px: int) -> None:
    width_px = max(160, int(width_px))
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
            f"fps={fps},scale={width_px}:-1:flags=lanczos,palettegen",
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
            f"fps={fps},scale={width_px}:-1:flags=lanczos[x];[x][1:v]paletteuse",
            str(output_gif),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GIF previews in assets/render.")
    parser.add_argument("--output-dir", default="assets/render", help="Output directory.")
    parser.add_argument("--frames", type=int, default=48, help="Frames per GIF.")
    parser.add_argument("--fps", type=int, default=12, help="GIF fps.")
    parser.add_argument("--width", type=int, default=320, help="Frame width.")
    parser.add_argument("--height", type=int, default=180, help="Frame height.")
    parser.add_argument("--framework", default="numpy", help="Default framework backend used for pipeline renders.")
    parser.add_argument(
        "--spec-dir",
        default="examples/progressions",
        help="Directory containing progression spec JSON files (one per asset).",
    )
    parser.add_argument(
        "--tui-capture",
        choices=["headless", "synthetic"],
        default="headless",
        help="TUI asset source. 'headless' captures real TUI via Xvfb/ffmpeg (default).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Generate only selected assets (by spec filename stem, plus 'tui_explorer').",
    )
    return parser.parse_args()


def _load_specs(spec_dir: Path) -> dict[str, dict[str, object]]:
    if not spec_dir.exists():
        raise RuntimeError(f"spec dir not found: {spec_dir}")
    out: dict[str, dict[str, object]] = {}
    for p in sorted(spec_dir.glob("*.json")):
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise RuntimeError(f"invalid spec (expected object): {p}")
        name = p.stem
        transforms = raw.get("transforms")
        if isinstance(transforms, str):
            transforms_list = [t.strip() for t in transforms.split(",") if t.strip()]
        elif isinstance(transforms, list) and all(isinstance(x, str) for x in transforms):
            transforms_list = [x.strip() for x in transforms if x.strip()]
        else:
            raise RuntimeError(f"invalid spec transforms in {p} (expected string or list[str])")
        if not transforms_list:
            raise RuntimeError(f"empty transforms in spec: {p}")
        inputs = raw.get("inputs")
        if inputs is not None and not isinstance(inputs, str):
            raise RuntimeError(f"invalid inputs in spec: {p} (expected string)")
        framework = raw.get("framework")
        if framework is not None and not isinstance(framework, str):
            raise RuntimeError(f"invalid framework in spec: {p} (expected string)")
        title = raw.get("title")
        if title is not None and not isinstance(title, str):
            raise RuntimeError(f"invalid title in spec: {p} (expected string)")
        out[name] = {
            "title": title or name.replace("_", " ").title(),
            "framework": framework,
            "inputs": inputs,
            "transforms": transforms_list,
        }
    if not out:
        raise RuntimeError(f"no specs found in {spec_dir}")
    return out


def _apply_inputs_blob(inputs: str | None) -> None:
    # tools.input_controls caches config; clear between assets so INPUTS changes take effect.
    input_controls._load_config.cache_clear()
    if not inputs:
        os.environ.pop("INPUTS", None)
        return
    os.environ["INPUTS"] = inputs


def main() -> int:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but not found in PATH.")

    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = _load_specs(Path(args.spec_dir))
    all_assets = sorted(specs.keys()) + ["tui_explorer"]
    selected = args.only if args.only else all_assets

    with tempfile.TemporaryDirectory(prefix="explorer_render_") as td:
        tmp = Path(td)
        gif_width = max(480, args.width * 2)
        explorer_bin = os.environ.get("EXPLORER_BIN", "").strip()
        if not explorer_bin:
            for candidate in (
                ROOT / "target" / "debug" / "explorer",
                ROOT / "target" / "release" / "explorer",
            ):
                if candidate.exists():
                    explorer_bin = candidate.as_posix()
                    break
        if not explorer_bin:
            explorer_bin = "explorer"

        for name in selected:
            if name == "tui_explorer":
                frames_dir = tmp / name
                frames_dir.mkdir(parents=True, exist_ok=True)
                output_gif = out_dir / f"{name}.gif"
                if args.tui_capture == "headless":
                    try:
                        capture_tui_session_gif(
                            output_gif=output_gif,
                            explorer_bin=explorer_bin,
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
                    encode_gif(frames_dir, output_gif, args.fps, "%03d.ppm", width_px=gif_width)
                print(f"wrote {output_gif}")
                continue

            if name not in specs:
                raise RuntimeError(f"unknown asset '{name}' (no spec found in {args.spec_dir})")
            spec = specs[name]
            pipeline = tuple(spec["transforms"])
            frames_dir = tmp / name
            frames_dir.mkdir(parents=True, exist_ok=True)

            # Apply spec inputs (path or raw JSON string) to influence distributions.
            inputs = spec.get("inputs")
            _apply_inputs_blob(str(inputs) if inputs else None)

            fw = str(spec.get("framework") or args.framework)
            engine = FrameworkEngine(fw)
            _reset_framework_rng(engine, seed=_stable_seed(fw, name, ",".join(pipeline)))
            n = max(48, min(192, int(max(args.width, args.height) * 0.24)))
            field = TensorField(tensor=engine._normal((n, n)))
            op_index = 0
            warmup = max(2, len(pipeline) // 2)
            for _ in range(warmup):
                field, op_index = _advance_tensor_field(
                    engine,
                    field=field,
                    pipeline=pipeline,
                    index=op_index,
                    ops_per_frame=1,
                )

            # Keep progression motion smooth and deterministic for GIF capture.
            ops_per_frame = 1
            width_px = max(640, args.width * 2)
            height_px = max(360, args.height * 2)
            for i in range(args.frames):
                field, op_index = _advance_tensor_field(
                    engine,
                    field=field,
                    pipeline=pipeline,
                    index=op_index,
                    ops_per_frame=ops_per_frame,
                )
                ppm = _render_pipeline_ppm(
                    engine,
                    field.tensor,
                    pipeline=pipeline,
                    stage="asset",
                    width_px=width_px,
                    height_px=height_px,
                )
                (frames_dir / f"{i:03d}.ppm").write_bytes(ppm)

            encode_gif(frames_dir, out_dir / f"{name}.gif", args.fps, "%03d.ppm", width_px=gif_width)
            print(f"wrote {out_dir / f'{name}.gif'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
