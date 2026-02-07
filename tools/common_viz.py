from __future__ import annotations

import os
import sys
import base64
import io
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
    # Keep detection strict to avoid "blank" output on unsupported terminals.
    if os.environ.get("T2C_VIZ_DISABLE_INLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if os.environ.get("T2C_VIZ_FORCE_INLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if not sys.stdout.isatty():
        return False

    # Strong positive signals for kitty graphics support.
    term = os.environ.get("TERM", "").lower()
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    if os.environ.get("KITTY_WINDOW_ID"):
        return True
    if "kitty" in term:
        return True
    if term_program == "kitty":
        return True

    # Ghostty compatibility can vary by environment/proxy/multiplexer.
    # Require explicit opt-in to avoid blank frames.
    if term_program == "ghostty" or "ghostty" in term:
        return os.environ.get("T2C_VIZ_GHOSTTY_INLINE", "").strip().lower() in {"1", "true", "yes", "on"}

    return False


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


def _kitty_transmit_payload(
    payload_b64: str,
    *,
    format_code: int,
    pixel_w: int,
    pixel_h: int,
    cells_w: int,
    cells_h: int,
) -> str:
    if not payload_b64:
        return "(empty)"

    # Kitty/ghostty graphics protocol in chunked mode.
    # Large payloads must be split across multiple escape sequences.
    chunks = []
    chunk_size = 4096
    first = True
    i = 0
    while i < len(payload_b64):
        part = payload_b64[i : i + chunk_size]
        i += chunk_size
        more = 1 if i < len(payload_b64) else 0
        if first:
            if pixel_w > 0 and pixel_h > 0:
                prefix = (
                    f"\x1b_Ga=T,f={format_code},s={pixel_w},v={pixel_h},"
                    f"c={cells_w},r={cells_h},m={more};"
                )
            else:
                prefix = f"\x1b_Ga=T,f={format_code},c={cells_w},r={cells_h},m={more};"
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


def _kitty_from_png_bytes(png_bytes: bytes, cells_w: int = 64, cells_h: int = 18) -> str:
    if not png_bytes:
        return "(empty)"
    payload_b64 = base64.b64encode(png_bytes).decode("ascii")
    # PNG dimensions are self-described; s/v can be omitted for f=100.
    return _kitty_transmit_payload(
        payload_b64,
        format_code=100,
        pixel_w=0,
        pixel_h=0,
        cells_w=cells_w,
        cells_h=cells_h,
    )


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


def _matplotlib_plot_png(
    arr: np.ndarray,
    *,
    stage: str,
    tensor_name: str,
    width_px: int = 1080,
    height_px: int = 620,
) -> bytes | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 0 or arr.size == 0:
        return None

    # High-complexity rendering for later module stages.
    high_stage = stage.endswith("4") or stage.endswith("5") or stage.endswith("final")
    dpi = 120
    fig_w = max(6.0, width_px / dpi)
    fig_h = max(3.5, height_px / dpi)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#080b15")

    title = f"{tensor_name} · {stage} · {arr.shape}"
    fig.suptitle(title, color="#dbe6ff", fontsize=12, y=0.98)

    # Shared style language inspired by README GIF aesthetics.
    grid_color = "#1e2840"
    tick_color = "#92a3c6"
    line_primary = "#57d8ff"
    line_secondary = "#ff6bb3"
    heat_cmap = "magma"

    if arr.ndim == 1 or (arr.ndim == 2 and 1 in arr.shape):
        y = np.ravel(arr).astype(np.float32)
        if high_stage:
            gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)
            axes = [
                fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[1, 1]),
            ]
            x = np.arange(y.size, dtype=np.int32)
            w = max(3, min(31, y.size // 12 if y.size >= 12 else 3))
            kernel = np.ones(w, dtype=np.float32) / w
            smooth = np.convolve(y, kernel, mode="same")

            axes[0].plot(x, y, color=line_primary, linewidth=1.3, alpha=0.9)
            axes[0].plot(x, smooth, color=line_secondary, linewidth=1.6, alpha=0.85)
            axes[0].fill_between(x, smooth, np.min(y), color=line_secondary, alpha=0.08)
            axes[0].set_title("Signal + Trend", color="#c8d6f2", fontsize=9)

            axes[1].hist(y, bins=min(48, max(12, y.size // 8)), color="#47c8ff", alpha=0.85)
            axes[1].set_title("Distribution", color="#c8d6f2", fontsize=9)

            spec = np.abs(np.fft.rfft(y - np.mean(y)))
            fx = np.arange(spec.size, dtype=np.int32)
            axes[2].plot(fx, spec, color="#a080ff", linewidth=1.4)
            axes[2].set_title("Frequency Spectrum", color="#c8d6f2", fontsize=9)

            if y.size > 1:
                axes[3].scatter(y[:-1], y[1:], s=4, alpha=0.5, c="#7af0a7")
            axes[3].set_title("Phase Portrait", color="#c8d6f2", fontsize=9)
            styled_axes = axes
        else:
            ax = fig.add_subplot(111)
            x = np.arange(y.size, dtype=np.int32)
            ax.plot(x, y, color=line_primary, linewidth=1.8)
            ax.fill_between(x, y, np.min(y), color=line_primary, alpha=0.12)
            ax.set_title("Tensor Signal", color="#c8d6f2", fontsize=10)
            styled_axes = [ax]
    else:
        arr2 = arr.reshape(arr.shape[0], -1) if arr.ndim > 2 else arr
        arr2 = np.asarray(arr2, dtype=np.float32)
        if high_stage:
            gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.24)
            ax_main = fig.add_subplot(gs[:, 0])
            ax_prof = fig.add_subplot(gs[0, 1])
            ax_hist = fig.add_subplot(gs[1, 1])

            im = ax_main.imshow(arr2, cmap=heat_cmap, aspect="auto", interpolation="bilinear")
            levels = np.linspace(np.min(arr2), np.max(arr2), num=8, dtype=np.float32)
            if np.max(levels) > np.min(levels):
                ax_main.contour(arr2, levels=levels, colors="#7fffd4", linewidths=0.4, alpha=0.55)
            ax_main.set_title("Field + Contours", color="#c8d6f2", fontsize=9)
            cbar = fig.colorbar(im, ax=ax_main, fraction=0.035, pad=0.02)
            cbar.ax.tick_params(colors=tick_color, labelsize=7)

            row_mean = np.mean(arr2, axis=1)
            col_mean = np.mean(arr2, axis=0)
            ax_prof.plot(np.arange(row_mean.size), row_mean, color=line_primary, linewidth=1.4, label="row mean")
            ax_prof.plot(np.arange(col_mean.size), col_mean, color=line_secondary, linewidth=1.2, label="col mean")
            ax_prof.legend(loc="upper right", fontsize=7, frameon=False, labelcolor=tick_color)
            ax_prof.set_title("Marginal Profiles", color="#c8d6f2", fontsize=9)

            ax_hist.hist(arr2.reshape(-1), bins=42, color="#6cf7b9", alpha=0.8)
            ax_hist.set_title("Value Density", color="#c8d6f2", fontsize=9)
            styled_axes = [ax_main, ax_prof, ax_hist]
        else:
            ax = fig.add_subplot(111)
            im = ax.imshow(arr2, cmap=heat_cmap, aspect="auto", interpolation="bilinear")
            ax.set_title("Tensor Field", color="#c8d6f2", fontsize=10)
            cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
            cbar.ax.tick_params(colors=tick_color, labelsize=8)
            styled_axes = [ax]

    for ax in styled_axes:
        ax.set_facecolor("#0d1325")
        ax.tick_params(colors=tick_color, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#2a3455")
        ax.grid(color=grid_color, linewidth=0.35, alpha=0.55)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


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


def _viz_caption(mode: str, arr: np.ndarray) -> str:
    vals = np.asarray(arr, dtype=np.float32).reshape(-1)
    spread = float(np.std(vals))
    span = float(np.max(vals) - np.min(vals))
    zeros = float(np.mean(np.isclose(vals, 0.0)))

    if mode == "plots":
        if arr.ndim == 1 or (arr.ndim == 2 and 1 in arr.shape):
            start = float(vals[0])
            end = float(vals[-1])
            drift = end - start
            if drift > max(1e-6, 0.05 * spread):
                trend = "upward"
            elif drift < -max(1e-6, 0.05 * spread):
                trend = "downward"
            else:
                trend = "flat"
            tone = "volatile" if spread > 0.75 else "stable"
            return f"Signal is {trend} with {tone} variation."
        contrast = "high-contrast" if span > 2.0 * max(1e-6, spread) else "low-contrast"
        return f"Tensor field shows {contrast} structure across dimensions."

    if mode == "heatmap":
        pocket = "outlier pockets" if span > 3.0 * max(1e-6, spread) else "smooth gradients"
        return f"Synthesized map highlights {pocket} and intensity flow."

    if mode == "matrix":
        density = "sparse" if zeros > 0.5 else "dense"
        return f"Matrix bins preserve rank intensity in a {density} field."

    return "ASCII retains coarse structure for non-graphics terminals."


def _format_caption(text: str) -> str:
    if _supports_graphical_terminal():
        return f"\x1b[38;2;139;233;253m╰─ ⟡ Insight\x1b[0m \x1b[3m{text}\x1b[0m"
    return f"Insight: {text}"


def viz_stage(
    stage: str,
    scope: dict[str, Any],
    to_numpy: Callable[[Any], np.ndarray | None],
    framework: str,
    metadata: dict[str, str] | None = None,
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
    style = os.environ.get("T2C_VIZ_STYLE", "plots").strip().lower()
    use_graphics = style != "ascii" and _supports_graphical_terminal()
    kitty_ok = _supports_kitty_graphics()
    inline_image_ok = _supports_inline_image_graphics()
    chosen = "ascii"
    # Canonical fallback chain:
    # plots -> heatmap -> matrix -> ascii
    if style == "plots":
        if inline_image_ok and kitty_ok:
            chosen = "plots"
        elif use_graphics:
            chosen = "heatmap"
        elif sys.stdout.isatty():
            chosen = "matrix"
        else:
            chosen = "ascii"
    elif style == "heatmap":
        if use_graphics:
            chosen = "heatmap"
        elif sys.stdout.isatty():
            chosen = "matrix"
        else:
            chosen = "ascii"
    elif style == "matrix":
        if use_graphics:
            chosen = "matrix"
        else:
            chosen = "ascii"
    elif style == "ascii":
        chosen = "ascii"
    else:
        chosen = "plots" if inline_image_ok and kitty_ok else ("heatmap" if use_graphics else "ascii")
    if os.environ.get("T2C_VIZ_TRACE", "1").strip().lower() in {"1", "true", "yes", "on"}:
        print(f"[VIS renderer={chosen}]")
        if chosen != "plots":
            print("[VIS hint] Plot rendering unavailable; using synthesized fallback.")

    print(f"\n[VIS:{framework}] {stage}")
    for name, arr in candidates[:limit]:
        arr_f = np.asarray(arr, dtype=np.float32)
        print(
            f"- {name}: shape={arr_f.shape} mean={arr_f.mean():.4f} "
            f"std={arr_f.std():.4f} min={arr_f.min():.4f} max={arr_f.max():.4f}"
        )
        if metadata and name in metadata:
            if _supports_graphical_terminal():
                print(f"\x1b[38;2;255;203;107m  ↳ {metadata[name]}\x1b[0m")
            else:
                print(f"  -> {metadata[name]}")
        if chosen == "plots":
            png_bytes = _matplotlib_plot_png(
                arr_f,
                stage=stage,
                tensor_name=name,
                width_px=1080,
                height_px=620,
            )
            if png_bytes is not None:
                print(_kitty_from_png_bytes(png_bytes, cells_w=72, cells_h=22))
            else:
                print(_pixel_heatmap(arr_f, width=72, height=30))
        elif chosen == "heatmap":
            print(_pixel_heatmap(arr_f, width=72, height=30))
        elif chosen == "matrix":
            print(_full_block_heatmap(arr_f, width=72, height=30))
        else:
            print(_ascii_heatmap(arr_f))
        print(_format_caption(_viz_caption(chosen, arr_f)))
