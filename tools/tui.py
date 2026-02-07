#!/usr/bin/env python3
"""Interactive terminal UI for tensors-to-consciousness visualization."""

from __future__ import annotations

import argparse
import os
import select
import subprocess
import sys
import termios
import tty
from types import ModuleType

from tools import shinkei


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive full-screen visualization studio."
    )
    parser.add_argument(
        "--framework",
        default="unknown",
        help="Framework label shown in the visualization header.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=96,
        help="Fallback output character width (default: 96).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=28,
        help="Fallback output character height (default: 28).",
    )
    parser.add_argument(
        "--view",
        choices=["simplified", "advanced", "ultra"],
        default="advanced",
        help="Initial view depth for interactive mode.",
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Run a single static render (no full-screen interaction).",
    )
    parser.add_argument(
        "--inputs",
        help="JSON file path or inline JSON to seed the parameter state.",
    )
    return parser.parse_args()


def _clear_screen() -> None:
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()


def _enter_alt() -> None:
    sys.stdout.write("\x1b[?1049h\x1b[?25l")
    sys.stdout.flush()


def _leave_alt() -> None:
    sys.stdout.write("\x1b[?25h\x1b[?1049l")
    sys.stdout.flush()


def _read_char(fd: int, timeout_s: float = 0.15) -> str:
    ready, _, _ = select.select([fd], [], [], timeout_s)
    if not ready:
        return ""
    return os.read(fd, 1).decode(errors="ignore")


def _edit_state(fd: int, state: shinkei.VizState, old: list[int]) -> None:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    try:
        sys.stdout.write("\x1b[2K\rEdit (key=value, e.g. freq=2.7, samples=2400): ")
        sys.stdout.flush()
        line = input().strip()
    finally:
        tty.setcbreak(fd)
    if "=" not in line:
        return
    key, value = [x.strip() for x in line.split("=", 1)]
    edits: dict[str, tuple[object, str]] = {
        "seed": (int, "seed"),
        "samples": (int, "samples"),
        "freq": (float, "freq"),
        "amplitude": (float, "amplitude"),
        "damping": (float, "damping"),
        "noise": (float, "noise"),
        "phase": (float, "phase"),
        "grid": (int, "grid"),
    }
    if key not in edits:
        return
    caster, attr = edits[key]
    try:
        casted = caster(value)
    except (TypeError, ValueError):
        return
    setattr(state, attr, casted)
    shinkei.normalize_state(state)


def _command_console(fd: int, old: list[int], state: shinkei.VizState, framework: str) -> bool:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    _leave_alt()
    print("\nCommand console (type `help`). `back` returns to studio, `quit` exits.")
    while True:
        try:
            raw = input("t2c> ").strip()
        except EOFError:
            raw = "back"
        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0].lower()
        if cmd in {"back", "b"}:
            _enter_alt()
            tty.setcbreak(fd)
            return True
        if cmd in {"quit", "exit"}:
            return False
        if cmd in {"help", "h", "?"}:
            print(
                "Commands: help, show, view <simplified|advanced|ultra>, "
                "set <k=v>, run <validate|all|0..6>, back, quit"
            )
            continue
        if cmd == "show":
            print(
                f"view={state.view} seed={state.seed} samples={state.samples} grid={state.grid} "
                f"freq={state.freq:.3f} amp={state.amplitude:.3f} damping={state.damping:.3f} noise={state.noise:.3f} phase={state.phase:.3f}"
            )
            continue
        if cmd == "view" and len(parts) == 2:
            if parts[1] in {"simplified", "advanced", "ultra"}:
                state.view = parts[1]
            else:
                print("Invalid view. Use simplified, advanced, or ultra.")
            continue
        if cmd == "set":
            expr = raw[len("set") :].strip()
            if "=" not in expr:
                print("Use: set key=value")
                continue
            key, value = [x.strip() for x in expr.split("=", 1)]
            edits: dict[str, tuple[object, str]] = {
                "seed": (int, "seed"),
                "samples": (int, "samples"),
                "freq": (float, "freq"),
                "amplitude": (float, "amplitude"),
                "damping": (float, "damping"),
                "noise": (float, "noise"),
                "phase": (float, "phase"),
                "grid": (int, "grid"),
            }
            if key not in edits:
                print(f"Unknown key: {key}")
                continue
            caster, attr = edits[key]
            try:
                setattr(state, attr, caster(value))
                shinkei.normalize_state(state)
            except (TypeError, ValueError):
                print("Invalid value type.")
            continue
        if cmd == "run" and len(parts) == 2:
            target = parts[1]
            if target not in {"validate", "all", "0", "1", "2", "3", "4", "5", "6"}:
                print("Use one of: validate, all, 0..6")
                continue
            cmdline = [
                sys.executable,
                "main.py",
                target,
                "--framework",
                framework,
                "--inputs",
                shinkei.state_json(state),
            ]
            print("+ " + " ".join(cmdline))
            subprocess.run(cmdline, check=False)
            continue
        print("Unknown command. Type `help`.")


def _render_interactive(np: ModuleType, state: shinkei.VizState, framework: str) -> int:
    common_viz = shinkei.load_common_viz()
    if not sys.stdin.isatty():
        return shinkei.render_static(np=np, state=state, framework=framework, width=96, height=28)

    use_plots = common_viz._supports_kitty_graphics()
    use_heatmap = common_viz._supports_graphical_terminal()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    _enter_alt()
    tty.setcbreak(fd)
    try:
        while True:
            _clear_screen()
            arr, stage, caption = shinkei.stage_payload(np, state)
            arr_f = np.asarray(arr, dtype=np.float32)
            renderer = shinkei.renderer_name(use_plots, use_heatmap)
            print(
                "\x1b[38;2;120;231;255mtensors-to-consciousness · visualization studio\x1b[0m "
                f"\x1b[38;2;255;184;108m[{framework}]\x1b[0m"
            )
            print(
                f"view={stage}  renderer={renderer}  seed={state.seed}  samples={state.samples}  grid={state.grid}"
            )
            print(
                f"freq={state.freq:.3f}  amplitude={state.amplitude:.3f}  damping={state.damping:.3f}  noise={state.noise:.3f}  phase={state.phase:.3f}"
            )
            print()

            if use_plots:
                png = common_viz._matplotlib_plot_png(arr_f, stage=stage, tensor_name=framework)
                if png:
                    print(common_viz._kitty_from_png_bytes(png, cells_w=80, cells_h=22))
                elif use_heatmap:
                    print(common_viz._pixel_heatmap(arr_f, width=80, height=30))
                else:
                    print(common_viz._ascii_heatmap(arr_f, width=96, height=28))
            elif use_heatmap:
                print(common_viz._pixel_heatmap(arr_f, width=80, height=30))
            else:
                print(common_viz._ascii_heatmap(arr_f, width=96, height=28))

            print()
            print(common_viz._format_caption(caption))
            print(
                "Controls: [1] simplified  [2] advanced  [3] ultra  [e] edit  [:] console  [r] reseed  [q] quit"
            )
            sys.stdout.flush()

            ch = _read_char(fd)
            if ch == "q":
                break
            if ch == "1":
                state.view = "simplified"
            elif ch == "2":
                state.view = "advanced"
            elif ch == "3":
                state.view = "ultra"
            elif ch == "r":
                state.seed = (state.seed + 1) % 2_000_000_000
            elif ch == "e":
                _edit_state(fd, state, old)
            elif ch == ":":
                if not _command_console(fd, old, state, framework):
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        _leave_alt()
    return 0


def main() -> int:
    args = parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import image as mpimg
        import numpy as np
    except ModuleNotFoundError:
        print(
            "matplotlib is not installed in the active environment. "
            "Run `python -m tools.setup <framework>` first.",
            file=sys.stderr,
        )
        return 1

    state = shinkei.build_state(
        view=getattr(args, "view", "advanced"),
        inputs=getattr(args, "inputs", None),
    )
    framework = getattr(args, "framework", "unknown")
    width = getattr(args, "width", 96)
    height = getattr(args, "height", 28)

    if getattr(args, "no_tui", False):
        return shinkei.render_static(
            np=np,
            state=state,
            framework=framework,
            width=width,
            height=height,
        )

    if not sys.stdout.isatty():
        ascii_plot, caption = shinkei.render_non_tty_ascii(
            mpimg=mpimg,
            np=np,
            state=state,
            width=width,
            height=height,
        )
        if ascii_plot:
            print(ascii_plot)
            print(shinkei.load_common_viz()._format_caption(caption))
            return 0
        return shinkei.render_static(
            np=np,
            state=state,
            framework=framework,
            width=width,
            height=height,
        )

    return _render_interactive(np=np, state=state, framework=framework)


if __name__ == "__main__":
    raise SystemExit(main())
