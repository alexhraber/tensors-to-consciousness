from __future__ import annotations

import os
import sys
import base64
from typing import Any, Callable

import numpy as np


def _supports_graphical_terminal() -> bool:
    if not sys.stdout.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    term = os.environ.get("TERM", "").lower()
    colorterm = os.environ.get("COLORTERM", "").lower()
    if "truecolor" in colorterm or "24bit" in colorterm:
        return True
    return "256color" in term or term.startswith("xterm") or term.startswith("screen")


def _supports_kitty_graphics() -> bool:
    if not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "").lower()
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    if "kitty" in term:
        return True
    if "ghostty" in term or term_program == "ghostty":
        return True
    if os.environ.get("KITTY_WINDOW_ID"):
        return True
    return False


def _supports_inline_image_graphics() -> bool:
    # Currently backed by kitty-compatible graphics protocol.
    # Over SSH, TERM/TERM_PROGRAM may be lossy; keep this permissive.
    if os.environ.get("T2C_VIZ_DISABLE_INLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if _supports_kitty_graphics():
        return True
    if os.environ.get("T2C_VIZ_FORCE_INLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    # If we're on a real color TTY, try inline protocol first.
    return _supports_graphical_terminal()


def _viridis(v: float) -> tuple[int, int, int]:
    # Coarse viridis-like gradient stops for terminal rendering.
    stops = [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ]
    v = min(1.0, max(0.0, v))
    x = v * (len(stops) - 1)
    i = int(x)
    if i >= len(stops) - 1:
        return stops[-1]
    t = x - i
    a = stops[i]
    b = stops[i + 1]
    r = int(a[0] + t * (b[0] - a[0]))
    g = int(a[1] + t * (b[1] - a[1]))
    bch = int(a[2] + t * (b[2] - a[2]))
    return r, g, bch


def _ansi_heatmap(arr: np.ndarray, width: int = 36, height: int = 12) -> str:
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return "(empty)"

    h, w = arr.shape
    ys = np.linspace(0, h - 1, num=min(height, h), dtype=int)
    xs = np.linspace(0, w - 1, num=min(width, w), dtype=int)
    sampled = arr[np.ix_(ys, xs)]
    mn = float(np.min(sampled))
    mx = float(np.max(sampled))
    if mx - mn < 1e-12:
        norm = np.zeros_like(sampled, dtype=np.float32)
    else:
        norm = (sampled - mn) / (mx - mn)

    lines: list[str] = []
    for row in norm:
        cells = []
        for v in row:
            r, g, b = _viridis(float(v))
            cells.append(f"\x1b[48;2;{r};{g};{b}m  \x1b[0m")
        lines.append("".join(cells))
    return "\n".join(lines)


def _full_block_heatmap(arr: np.ndarray, width: int = 56, height: int = 20) -> str:
    """Truecolor full-block renderer (one sample per terminal cell)."""
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return "(empty)"

    h, w = arr.shape
    ys = np.linspace(0, h - 1, num=min(height, h), dtype=int)
    xs = np.linspace(0, w - 1, num=min(width, w), dtype=int)
    sampled = arr[np.ix_(ys, xs)]
    mn = float(np.min(sampled))
    mx = float(np.max(sampled))
    if mx - mn < 1e-12:
        norm = np.full_like(sampled, 0.5, dtype=np.float32)
    else:
        norm = (sampled - mn) / (mx - mn)

    lines: list[str] = []
    for row in norm:
        cells = []
        for v in row:
            r, g, b = _viridis(float(v))
            cells.append(f"\x1b[38;2;{r};{g};{b}m█\x1b[0m")
        lines.append("".join(cells))
    return "\n".join(lines)


def _braille_heatmap(arr: np.ndarray, width: int = 72, height: int = 24) -> str:
    """High-density truecolor renderer using Unicode Braille cells (2x4 pixels/cell)."""
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return "(empty)"

    px_h = max(4, height * 4)
    px_w = max(2, width * 2)
    sampled = _upsample_interp(arr, out_h=px_h, out_w=px_w)
    if sampled.shape[0] < 4:
        sampled = np.vstack([sampled] * (4 // sampled.shape[0] + 1))[:4, :]
    if sampled.shape[1] < 2:
        sampled = np.hstack([sampled] * (2 // sampled.shape[1] + 1))[:, :2]

    mn = float(np.min(sampled))
    mx = float(np.max(sampled))
    if mx - mn < 1e-12:
        # Keep constant tensors visible as a mid-tone tile rather than blank space.
        norm = np.full_like(sampled, 0.5, dtype=np.float32)
    else:
        norm = (sampled - mn) / (mx - mn)

    dot_map = {
        (0, 0): 1,
        (1, 0): 2,
        (2, 0): 4,
        (3, 0): 64,
        (0, 1): 8,
        (1, 1): 16,
        (2, 1): 32,
        (3, 1): 128,
    }

    lines: list[str] = []
    for r in range(0, norm.shape[0], 4):
        block_rows = norm[r : r + 4, :]
        if block_rows.shape[0] < 4:
            block_rows = np.vstack([block_rows, block_rows[-1:, :].repeat(4 - block_rows.shape[0], axis=0)])
        row_chars = []
        for c in range(0, norm.shape[1], 2):
            block = block_rows[:, c : c + 2]
            if block.shape[1] < 2:
                block = np.hstack([block, block[:, -1:]])
            bits = 0
            for (rr, cc), bit in dot_map.items():
                if block[rr, cc] > 0.45:
                    bits |= bit
            mean_v = float(np.mean(block))
            if bits == 0 and mean_v > 0.05:
                bits = 1
            ch = " " if bits == 0 else chr(0x2800 + bits)
            cr, cg, cb = _viridis(mean_v)
            row_chars.append(f"\x1b[38;2;{cr};{cg};{cb}m{ch}\x1b[0m")
        lines.append("".join(row_chars))
    return "\n".join(lines)


def _pixel_heatmap(arr: np.ndarray, width: int = 48, height: int = 20) -> str:
    """High-density terminal renderer using truecolor half blocks.

    Each character encodes two vertical pixels via foreground/background:
    - upper pixel -> fg color
    - lower pixel -> bg color
    """
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return "(empty)"

    # We need an even number of rows because each terminal row shows two pixels.
    px_h = max(2, height * 2)
    sampled = _upsample_interp(arr, out_h=px_h, out_w=max(2, width))
    if sampled.shape[0] == 1:
        sampled = np.vstack([sampled, sampled])
    elif sampled.shape[0] % 2 == 1:
        sampled = np.vstack([sampled, sampled[-1:, :]])
    mn = float(np.min(sampled))
    mx = float(np.max(sampled))
    if mx - mn < 1e-12:
        norm = np.full_like(sampled, 0.5, dtype=np.float32)
    else:
        norm = (sampled - mn) / (mx - mn)

    lines: list[str] = []
    for r in range(0, norm.shape[0], 2):
        top = norm[r]
        bottom = norm[r + 1]
        cells = []
        for vt, vb in zip(top, bottom):
            tr, tg, tb = _viridis(float(vt))
            br, bg, bb = _viridis(float(vb))
            cells.append(f"\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m▀\x1b[0m")
        lines.append("".join(cells))
    return "\n".join(lines)


def _kitty_from_rgb(rgb: np.ndarray, cells_w: int = 48, cells_h: int = 12) -> str:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return "(empty)"
    if rgb.size == 0:
        return "(empty)"

    payload = base64.b64encode(rgb.tobytes()).decode("ascii")

    # Kitty/ghostty graphics protocol in chunked mode.
    # Large payloads must be split across multiple escape sequences.
    chunks = []
    chunk_size = 4096
    first = True
    i = 0
    while i < len(payload):
        part = payload[i : i + chunk_size]
        i += chunk_size
        more = 1 if i < len(payload) else 0
        if first:
            prefix = (
                f"\x1b_Ga=T,f=24,s={rgb.shape[1]},v={rgb.shape[0]},"
                f"c={cells_w},r={cells_h},m={more};"
            )
            first = False
        else:
            prefix = f"\x1b_Gm={more};"
        seq = f"{prefix}{part}\x1b\\"
        if os.environ.get("TMUX"):
            # tmux passthrough envelope so kitty APC survives multiplexing.
            escaped_seq = seq.replace("\x1b", "\x1b\x1b")
            seq = f"\x1bPtmux;{escaped_seq}\x1b\\"
        chunks.append(seq)
    return "".join(chunks)


def _upsample_interp(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Simple separable interpolation to avoid blocky visual output."""
    arr = np.asarray(arr, dtype=np.float32)
    in_h, in_w = arr.shape
    y_old = np.linspace(0.0, 1.0, in_h, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, in_w, dtype=np.float32)
    y_new = np.linspace(0.0, 1.0, out_h, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, out_w, dtype=np.float32)

    tmp = np.empty((in_h, out_w), dtype=np.float32)
    for i in range(in_h):
        tmp[i, :] = np.interp(x_new, x_old, arr[i, :]).astype(np.float32)

    out = np.empty((out_h, out_w), dtype=np.float32)
    for j in range(out_w):
        out[:, j] = np.interp(y_new, y_old, tmp[:, j]).astype(np.float32)
    return out


def _box_blur(arr: np.ndarray, rounds: int = 2) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    for _ in range(max(1, rounds)):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="reflect")
        out = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0
    return out.astype(np.float32)


def _fluid_rgb(arr: np.ndarray, width: int = 320, height: int = 180) -> np.ndarray:
    """Generate a smooth, non-blocky color field suitable for inline image rendering."""
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    sampled = _upsample_interp(arr, out_h=max(24, height), out_w=max(24, width))
    if sampled.shape[0] < 2:
        sampled = np.vstack([sampled, sampled])
    if sampled.shape[1] < 2:
        sampled = np.hstack([sampled, sampled])
    sampled = _box_blur(sampled, rounds=3)
    # Add soft ridges so tiny tensors still get visible, fluid-looking structure.
    yy = np.linspace(0.0, 1.0, sampled.shape[0], dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, sampled.shape[1], dtype=np.float32)[None, :]
    sampled = sampled + 0.08 * np.sin(2.0 * np.pi * (1.3 * xx + 0.7 * yy))

    mn = float(np.min(sampled))
    mx = float(np.max(sampled))
    if mx - mn < 1e-12:
        norm = np.full_like(sampled, 0.5, dtype=np.float32)
    else:
        norm = (sampled - mn) / (mx - mn)

    rgb = np.zeros((norm.shape[0], norm.shape[1], 3), dtype=np.uint8)
    for y in range(norm.shape[0]):
        for x in range(norm.shape[1]):
            rgb[y, x] = _viridis(float(norm[y, x]))
    return rgb


def _fluid_inline_render(arr: np.ndarray, width: int = 320, height: int = 180) -> str:
    rgb = _fluid_rgb(arr, width=width, height=height)
    return _kitty_from_rgb(rgb, cells_w=64, cells_h=18)


def _ascii_heatmap(arr: np.ndarray, width: int = 36, height: int = 12) -> str:
    ramp = " .:-=+*#%@"
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return "(empty)"

    h, w = arr.shape
    ys = np.linspace(0, h - 1, num=min(height, h), dtype=int)
    xs = np.linspace(0, w - 1, num=min(width, w), dtype=int)
    sampled = arr[np.ix_(ys, xs)]
    mn = float(np.min(sampled))
    mx = float(np.max(sampled))
    if mx - mn < 1e-12:
        norm = np.full_like(sampled, 0.5, dtype=np.float32)
    else:
        norm = (sampled - mn) / (mx - mn)
    idx = np.clip((norm * (len(ramp) - 1)).astype(int), 0, len(ramp) - 1)
    lines = ["".join(ramp[i] for i in row) for row in idx]
    return "\n".join(lines)


def viz_stage(
    stage: str,
    scope: dict[str, Any],
    to_numpy: Callable[[Any], np.ndarray | None],
    framework: str,
    limit: int = 3,
) -> None:
    if os.environ.get("T2C_VIZ", "1").strip().lower() in {"0", "false", "off", "no"}:
        return

    candidates: list[tuple[str, np.ndarray]] = []
    for name, value in scope.items():
        if name.startswith("_"):
            continue
        arr = to_numpy(value)
        if arr is None:
            continue
        if arr.ndim == 0:
            continue
        if arr.size < 4:
            continue
        candidates.append((name, arr))

    if not candidates:
        return

    # Prefer larger tensors to show the most informative surfaces.
    candidates.sort(key=lambda item: item[1].size, reverse=True)
    style = os.environ.get("T2C_VIZ_STYLE", "fluid-render").strip().lower()
    use_graphics = style != "ascii" and _supports_graphical_terminal()
    kitty_ok = _supports_kitty_graphics()
    inline_image_ok = _supports_inline_image_graphics()
    chosen = "ascii"
    if style in {"fluid-render", "fluid-image", "auto", "fluid"}:
        # Ordered fallback chain:
        # 1) fluid-render (inline raster image)
        # 2) gpu-render (braille truecolor)
        # 3) kitty protocol image
        # 4) half-cubes (truecolor)
        # 5) full-cubes (truecolor)
        # 6) ascii
        if inline_image_ok:
            chosen = "fluid-render"
        elif use_graphics:
            chosen = "gpu-render"
        elif kitty_ok:
            chosen = "kitty"
        elif sys.stdout.isatty():
            chosen = "half-cubes"
        else:
            chosen = "ascii"
    elif style in {"gpu-render", "gpu", "braille"}:
        # Skip fluid render in this explicit mode.
        if use_graphics:
            chosen = "gpu-render"
        elif kitty_ok:
            chosen = "kitty"
        elif sys.stdout.isatty():
            chosen = "half-cubes"
        else:
            chosen = "ascii"
    elif style in {"half-cubes", "half", "pixel"}:
        if use_graphics:
            chosen = "half-cubes"
        elif sys.stdout.isatty():
            chosen = "full-cubes"
        else:
            chosen = "ascii"
    elif style in {"full-cubes", "full", "ansi"}:
        if use_graphics:
            chosen = "full-cubes"
        else:
            chosen = "ascii"
    elif style in {"ascii", "text"}:
        chosen = "ascii"
    elif style in {"kitty", "image"}:
        if kitty_ok:
            chosen = "kitty"
        elif use_graphics:
            chosen = "half-cubes"
        elif sys.stdout.isatty():
            chosen = "full-cubes"
        else:
            chosen = "ascii"
    if os.environ.get("T2C_VIZ_TRACE", "0").strip() in {"1", "true", "yes", "on"}:
        print(f"[VIS renderer={chosen}]")
        if os.environ.get("SSH_TTY") and not inline_image_ok:
            print("[VIS hint] SSH detected. Set T2C_VIZ_FORCE_INLINE=1 to force inline image protocol.")

    print(f"\n[VIS:{framework}] {stage}")
    for name, arr in candidates[:limit]:
        arr_f = np.asarray(arr, dtype=np.float32)
        print(
            f"- {name}: shape={arr_f.shape} mean={arr_f.mean():.4f} "
            f"std={arr_f.std():.4f} min={arr_f.min():.4f} max={arr_f.max():.4f}"
        )
        if chosen == "fluid-render":
            print(_fluid_inline_render(arr_f, width=360, height=200))
        elif chosen == "gpu-render":
            print(_braille_heatmap(arr_f, width=72, height=24))
        elif chosen == "kitty":
            print(_kitty_from_rgb(_fluid_rgb(arr_f, width=220, height=120), cells_w=52, cells_h=14))
        elif chosen == "half-cubes":
            print(_pixel_heatmap(arr_f, width=72, height=30))
        elif chosen == "full-cubes":
            print(_full_block_heatmap(arr_f, width=72, height=30))
        else:
            print(_ascii_heatmap(arr_f))
