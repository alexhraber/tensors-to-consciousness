#!/usr/bin/env python3
"""Terminal visualization utility using matplotlib rasterized to ASCII."""

from __future__ import annotations

import argparse
import io
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a terminal-safe visualization using matplotlib."
    )
    parser.add_argument(
        "--framework",
        default="unknown",
        help="Framework label displayed in the visualization header.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=96,
        help="Output character width (default: 96).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=28,
        help="Output character height (default: 28).",
    )
    return parser.parse_args()


def _to_ascii(rgba, width: int, height: int) -> str:
    # Lightweight grayscale ramp with good terminal contrast.
    ramp = " .:-=+*#%@"
    r = rgba[..., 0].astype("float32")
    g = rgba[..., 1].astype("float32")
    b = rgba[..., 2].astype("float32")
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    luminance = 255.0 - luminance
    if luminance.max() > 0:
        luminance = luminance / luminance.max()

    y_idx = [int(i * luminance.shape[0] / height) for i in range(height)]
    x_idx = [int(i * luminance.shape[1] / width) for i in range(width)]
    lines: list[str] = []
    for yy in y_idx:
        row = []
        for xx in x_idx:
            v = luminance[yy, xx]
            row.append(ramp[min(int(v * (len(ramp) - 1)), len(ramp) - 1)])
        lines.append("".join(row))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError:
        print(
            "matplotlib is not installed in the active environment. "
            "Run `python -m tools.setup <framework>` first.",
            file=sys.stderr,
        )
        return 1

    x = np.linspace(0.0, 12.0, 600)
    y1 = np.sin(x)
    y2 = np.cos(1.5 * x) * 0.6
    y3 = np.exp(-x / 8.0) * np.sin(2.2 * x)

    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=120)
    ax.plot(x, y1, linewidth=2.0, label="signal_a")
    ax.plot(x, y2, linewidth=2.0, label="signal_b")
    ax.plot(x, y3, linewidth=2.0, label="signal_c")
    ax.set_title(f"tensors-to-consciousness | terminal viz | {args.framework}")
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)

    from matplotlib import image as mpimg

    rgba = mpimg.imread(buffer)
    if rgba.dtype != "uint8":
        rgba = (rgba * 255).astype("uint8")
    ascii_plot = _to_ascii(rgba, width=args.width, height=args.height)
    print(ascii_plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
