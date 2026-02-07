#!/usr/bin/env python3
"""Interactive terminal UI for tensors-to-consciousness rendering."""

from __future__ import annotations

import argparse
import os
import re
import select
import signal
import shutil
import subprocess
import sys
import termios
import tty
from types import ModuleType

from transforms.registry import build_tui_profiles
from transforms.registry import resolve_transform_keys
from frameworks.engine import FrameworkEngine
from tools import runtime
from tools import shinkei

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
LIVE_TRANSFORMS = (
    "signal warp",
    "field coupling",
    "spectral lift",
    "phase sweep",
)
LIVE_REFRESH_SECONDS = 0.45
SCRIPT_PROFILES = build_tui_profiles(resolve_transform_keys("all"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive full-screen rendering explorer."
    )
    parser.add_argument(
        "--framework",
        default="unknown",
        help="Framework label shown in the rendering header.",
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
        type=shinkei.parse_view_arg,
        choices=[shinkei.EXPLORER_VIEW],
        default=shinkei.EXPLORER_VIEW,
        help="Explorer view (single-mode).",
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
    parser.add_argument(
        "--transforms",
        help="Comma-separated transform keys, or 'default'/'all' (preferred).",
    )
    parser.add_argument(
        "--transform",
        help="Initial transform selector key/title within the active transform set.",
    )
    return parser.parse_args()


def _clear_screen() -> None:
    # Stable redraw without hard screen flash.
    sys.stdout.write("\x1b[H\x1b[J")
    sys.stdout.flush()


def _cursor_up(lines: int) -> None:
    if lines > 0:
        sys.stdout.write(f"\x1b[{lines}A")


def _clear_to_end() -> None:
    sys.stdout.write("\x1b[J")


def _line_count(text: str) -> int:
    if not text:
        return 1
    return text.count("\n") + 1


def _enter_alt() -> None:
    sys.stdout.write("\x1b[?1049h\x1b[?25l")
    sys.stdout.flush()


def _leave_alt() -> None:
    sys.stdout.write("\x1b[?25h\x1b[?1049l")
    sys.stdout.flush()


def _read_char(fd: int, timeout_s: float = 5.0) -> str:
    ready, _, _ = select.select([fd], [], [], timeout_s)
    if not ready:
        return ""
    return os.read(fd, 1).decode(errors="ignore")


def _field_specs() -> list[tuple[str, str, object]]:
    return [
        ("seed", "Random seed", int),
        ("samples", "Sample count", int),
        ("grid", "Field resolution", int),
        ("freq", "Signal frequency", float),
        ("amplitude", "Signal amplitude", float),
        ("phase", "Phase shift", float),
        ("damping", "Damping factor", float),
        ("noise", "Noise level", float),
    ]


def _guided_edit_state(fd: int, state: shinkei.RenderState, old: list[int]) -> None:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    try:
        print("\nParameter Input")
        print("Explorer controls")
        for key, label, caster in _field_specs():
            current = getattr(state, key)
            raw = input(f"- {label} [{key}] ({current}): ").strip()
            if not raw:
                continue
            try:
                setattr(state, key, caster(raw))
            except (TypeError, ValueError):
                print(f"  invalid value for {key}; keeping {current}")
    finally:
        tty.setcbreak(fd)
    shinkei.normalize_state(state)


def _quick_edit_state(fd: int, state: shinkei.RenderState, old: list[int]) -> None:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    try:
        line = input("Quick edit key=value (blank to cancel): ").strip()
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
        setattr(state, attr, caster(value))
        shinkei.normalize_state(state)
    except (TypeError, ValueError):
        return


def _command_console(
    fd: int,
    old: list[int],
    state: shinkei.RenderState,
    framework: str,
    platform: str,
) -> bool:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    _leave_alt()
    print("\nCommand console (type `help`). `back` returns to explorer, `quit` exits.")
    while True:
        try:
            raw = input("explore> ").strip()
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
                "Commands: help, show, "
                "set <k=v>, run <validate|transform1,transform2,...>, back, quit"
            )
            continue
        if cmd == "show":
            print(
                f"seed={state.seed} samples={state.samples} grid={state.grid} "
                f"freq={state.freq:.3f} amp={state.amplitude:.3f} damping={state.damping:.3f} noise={state.noise:.3f} phase={state.phase:.3f} platform={platform}"
            )
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
            if target == "validate":
                cmdline = [
                    sys.executable,
                    "explorer.py",
                    "validate",
                    "--framework",
                    framework,
                    "--inputs",
                    shinkei.state_json(state),
                ]
            else:
                cmdline = [
                    sys.executable,
                    "explorer.py",
                    "run",
                    "--framework",
                    framework,
                    "--transforms",
                    target,
                    "--inputs",
                    shinkei.state_json(state),
                ]
            print("+ " + " ".join(cmdline))
            env = os.environ.copy()
            env.update(_platform_env(framework, platform))
            subprocess.run(cmdline, check=False, env=env)
            continue
        print("Unknown command. Type `help`.")


def _persist_framework(framework: str) -> None:
    config = runtime.load_config_optional()
    config["framework"] = framework
    config["venv"] = f".venv-{framework}"
    runtime.save_config(config)


def _load_platform() -> str:
    try:
        cfg = runtime.load_config()
        platform = str(cfg.get("platform", "gpu")).lower()
        if platform in {"cpu", "gpu"}:
            return platform
    except RuntimeError:
        pass
    return "gpu"


def _persist_platform(platform: str) -> None:
    config = runtime.load_config_optional()
    config["platform"] = platform
    runtime.save_config(config)


def _framework_selector(fd: int, current: str) -> str | None:
    options = list(runtime.SUPPORTED_FRAMEWORKS)
    idx = options.index(current) if current in options else 0
    while True:
        _clear_screen()
        print("┌──────────────────────────────────────────────┐")
        print("│ Switch Framework                             │")
        print("├──────────────────────────────────────────────┤")
        for i, fw in enumerate(options):
            marker = ">" if i == idx else " "
            current_tag = " (current)" if fw == current else ""
            print(f"│ {marker} {fw}{current_tag}".ljust(47) + "│")
        print("├──────────────────────────────────────────────┤")
        print("│ Controls: [n]/[j] next  [p]/[k] prev         │")
        print("│           [Enter] select  [q] cancel         │")
        print("└──────────────────────────────────────────────┘")
        sys.stdout.flush()

        ch = _read_char(fd, timeout_s=30.0)
        if ch in {"n", "j", "\t"}:
            idx = (idx + 1) % len(options)
            continue
        if ch in {"p", "k"}:
            idx = (idx - 1) % len(options)
            continue
        if ch in {"\r", "\n"}:
            return options[idx]
        if ch in {"q", "\x1b"}:
            return None


def _platform_selector(fd: int, current: str) -> str | None:
    options = ["cpu", "gpu"]
    idx = options.index(current) if current in options else 1
    while True:
        _clear_screen()
        print("┌──────────────────────────────────────────────┐")
        print("│ Compute Platform                             │")
        print("├──────────────────────────────────────────────┤")
        for i, mode in enumerate(options):
            marker = ">" if i == idx else " "
            current_tag = " (current)" if mode == current else ""
            print(f"│ {marker} {mode}{current_tag}".ljust(47) + "│")
        print("├──────────────────────────────────────────────┤")
        print("│ Controls: [n]/[j] next  [p]/[k] prev         │")
        print("│           [Enter] select  [q] cancel         │")
        print("└──────────────────────────────────────────────┘")
        sys.stdout.flush()

        ch = _read_char(fd, timeout_s=30.0)
        if ch in {"n", "j", "\t"}:
            idx = (idx + 1) % len(options)
            continue
        if ch in {"p", "k"}:
            idx = (idx - 1) % len(options)
            continue
        if ch in {"\r", "\n"}:
            return options[idx]
        if ch in {"q", "\x1b"}:
            return None


def _platform_env(framework: str, platform: str) -> dict[str, str]:
    env: dict[str, str] = {"PLATFORM": platform}
    if framework == "jax":
        env["JAX_PLATFORM_NAME"] = platform
    if framework in {"pytorch", "keras", "cupy"} and platform == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    if framework == "keras":
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    return env


def _handoff_framework_switch(framework: str) -> int:
    cmdline = [
        sys.executable,
        "explorer.py",
        "render",
        "--framework",
        framework,
    ]
    return subprocess.run(cmdline, check=False).returncode


def _frame_line(text: str, width: int = 104) -> str:
    if len(text) >= width - 2:
        text = text[: width - 5] + "..."
    return "│" + text.ljust(width - 2) + "│"


def _theme_enabled() -> bool:
    return shinkei._supports_graphical_terminal() and not os.environ.get("NO_COLOR")


def _style(
    text: str,
    *,
    fg: tuple[int, int, int] | None = None,
    bg: tuple[int, int, int] | None = None,
    bold: bool = False,
    dim: bool = False,
) -> str:
    if not _theme_enabled():
        return text
    parts: list[str] = []
    if bold:
        parts.append("1")
    if dim:
        parts.append("2")
    if fg is not None:
        r, g, b = fg
        parts.append(f"38;2;{r};{g};{b}")
    if bg is not None:
        r, g, b = bg
        parts.append(f"48;2;{r};{g};{b}")
    if not parts:
        return text
    return f"\x1b[{';'.join(parts)}m{text}\x1b[0m"


def _visible_len(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


def _center_text(text: str, width: int) -> str:
    visible = _visible_len(text)
    if visible >= width:
        return text
    return (" " * ((width - visible) // 2)) + text


def _print_centered(lines: list[str], term_cols: int) -> None:
    for line in lines:
        print(_center_text(line, term_cols))


def _keycap(label: str, tone: tuple[int, int, int] = (210, 224, 245)) -> str:
    return _style(f" {label} ", fg=(18, 24, 34), bg=tone, bold=True)


def _render_header(
    framework: str,
    platform: str,
    state: shinkei.RenderState,
    renderer: str,
    width: int = 104,
) -> None:
    top = "┌" + ("─" * (width - 2)) + "┐"
    bot = "└" + ("─" * (width - 2)) + "┘"
    header_line = _frame_line(
        f" explorer · {framework} · {platform} · view={state.view} · renderer={renderer} ",
        width=width,
    )
    stats_line = _frame_line(
        f" seed={state.seed} samples={state.samples} grid={state.grid} freq={state.freq:.3f} amp={state.amplitude:.3f} damp={state.damping:.3f} noise={state.noise:.3f} phase={state.phase:.3f} ",
        width=width,
    )
    print(_style(top, fg=(83, 109, 154)))
    print(_style(header_line, fg=(208, 224, 252), bold=True))
    print(_style(stats_line, fg=(142, 166, 206)))
    print(_style(bot, fg=(83, 109, 154)))


def _render_landing(framework: str, platform: str, width: int, term_cols: int, term_rows: int) -> None:
    card_w = max(58, min(width, 86))
    top = "┌" + ("─" * (card_w - 2)) + "┐"
    bot = "└" + ("─" * (card_w - 2)) + "┘"
    pad_top = max(1, min(6, (term_rows - 14) // 3))
    print("\n" * pad_top, end="")

    framework_text = _style(framework, fg=(126, 214, 255), bold=True)
    platform_text = _style(platform, fg=(153, 231, 173), bold=True)
    lines: list[str] = [
        _style(top, fg=(83, 109, 154)),
        _frame_line(_style(" TENSORS TO CONSCIOUSNESS ", fg=(234, 241, 255), bold=True), width=card_w),
        _frame_line(_style(" terminal transform exploration ", fg=(150, 178, 224), dim=True), width=card_w),
        _frame_line("", width=card_w),
        _frame_line(f" {framework_text}  ·  {platform_text} ", width=card_w),
        _frame_line("", width=card_w),
        _frame_line(f" {_keycap('ENTER', (255, 214, 136))} begin exploring ", width=card_w),
        _frame_line(
            f" {_keycap('F')} framework   {_keycap('P')} platform   {_keycap('Q', (255, 189, 189))} quit ",
            width=card_w,
        ),
        _style(bot, fg=(83, 109, 154)),
    ]
    _print_centered(lines, term_cols=term_cols)


def _layout_for_view() -> dict[str, int]:
    size = shutil.get_terminal_size(fallback=(140, 44))
    cols = max(100, size.columns)
    rows = max(30, size.lines)

    header_w = max(92, min(cols - 2, 150))
    plot_w = max(72, min(cols - 4, 120))
    plot_h = max(20, min(rows - 12, 32))

    return {
        "cols": cols,
        "rows": rows,
        "header_w": header_w,
        "plot_w": plot_w,
        "plot_h": plot_h,
        "ascii_w": max(72, min(plot_w, 120)),
        "ascii_h": max(16, min(plot_h, 30)),
    }


def _profile_state(index: int, transform_index: int, seed: int = 7) -> shinkei.RenderState:
    profile = SCRIPT_PROFILES[index % len(SCRIPT_PROFILES)]
    transform = profile["transforms"][0]
    preset = transform["preset"]
    return shinkei.RenderState(
        seed=seed,
        samples=int(preset["samples"]),
        freq=float(preset["freq"]),
        amplitude=float(preset["amplitude"]),
        damping=float(preset["damping"]),
        noise=float(preset["noise"]),
        phase=float(preset["phase"]),
        grid=int(preset["grid"]),
        view=shinkei.EXPLORER_VIEW,
    )


def _apply_profile_to_state(state: shinkei.RenderState, index: int) -> None:
    preset = _profile_state(index, 0, seed=state.seed)
    state.samples = preset.samples
    state.freq = preset.freq
    state.amplitude = preset.amplitude
    state.damping = preset.damping
    state.noise = preset.noise
    state.phase = preset.phase
    state.grid = preset.grid
    shinkei.normalize_state(state)


def _next_script_index(index: int, direction: int = 1) -> int:
    return (index + direction) % len(SCRIPT_PROFILES)


def _resolve_script_index(selector: str | None) -> int:
    if selector is None:
        return 0
    raw = selector.strip().lower()
    if not raw:
        return 0
    for i, profile in enumerate(SCRIPT_PROFILES):
        if raw == str(profile["id"]).lower():
            return i
    for i, profile in enumerate(SCRIPT_PROFILES):
        if raw in profile["title"].lower():
            return i
    return 0


def _toggle_pipeline_key(pipeline: list[str], key: str) -> None:
    if key in pipeline:
        pipeline.remove(key)
    else:
        pipeline.append(key)


def _move_pipeline_key(pipeline: list[str], key: str, direction: int) -> None:
    if key not in pipeline:
        return
    idx = pipeline.index(key)
    dst = idx + direction
    if dst < 0 or dst >= len(pipeline):
        return
    pipeline[idx], pipeline[dst] = pipeline[dst], pipeline[idx]


def _render_pipeline_selector(
    transforms: tuple[dict[str, object], ...],
    pipeline: list[str],
    cursor_index: int,
    *,
    max_lines: int = 8,
) -> list[str]:
    total = len(transforms)
    if total <= max_lines:
        start = 0
        end = total
    else:
        half = max_lines // 2
        start = max(0, min(cursor_index - half, total - max_lines))
        end = start + max_lines

    lines = ["transforms (cursor + checkbox + precedence):"]
    for i in range(start, end):
        transform = transforms[i]
        key = str(transform["key"])
        title = str(transform["title"])
        selected = key in pipeline
        marker = ">" if i == cursor_index else " "
        check = "[x]" if selected else "[ ]"
        order = pipeline.index(key) + 1 if selected else 0
        order_text = f"{order:02d}" if order else "--"
        lines.append(f" {marker} {check} {order_text} {key} · {title}")
    if total > max_lines:
        lines.append(f" showing {start + 1}-{end} of {total}")
    return lines


def _live_motion_state(base: shinkei.RenderState, tick: int) -> tuple[shinkei.RenderState, str]:
    phase_idx = (tick // 5) % len(LIVE_TRANSFORMS)
    wave = (tick % 40) / 40.0
    if wave > 0.5:
        wave = 1.0 - wave
    wave = wave * 2.0

    state = shinkei.RenderState(
        seed=base.seed + tick,
        samples=max(256, int(base.samples * (0.85 + 0.30 * wave))),
        freq=base.freq * (0.9 + 0.35 * wave),
        amplitude=base.amplitude * (0.8 + 0.4 * wave),
        damping=max(0.0, base.damping * (0.7 + 0.7 * wave)),
        noise=base.noise * (0.6 + 0.9 * (1.0 - wave)),
        phase=base.phase + (tick * 0.18),
        grid=max(32, int(base.grid * (0.85 + 0.35 * wave))),
        view=base.view,
    )
    return state, LIVE_TRANSFORMS[phase_idx]


def _compact_controls_line() -> str:
    return (
        "Controls: [n/b] cursor  [x] toggle  [[/]] order  [space] live  "
        "[i/e] params  [f/p] runtime  [:] console  [h] details  [q] quit"
    )


def _detailed_controls_line() -> str:
    return (
        "Controls: [n]/[b] cursor next/back  [x] toggle transform  ['[']/[']'] precedence up/down  "
        "[f] framework  [p] cpu/gpu  [i] guided input  [e] quick key=value  [space] pause/resume live dynamics  "
        "[:] command mode  [h] compact mode  [r] reseed  [q] quit"
    )


def _render_interactive(
    np: ModuleType,
    state: shinkei.RenderState,
    framework: str,
    initial_pipeline: tuple[str, ...],
    transform_selector: str | None = None,
) -> int:
    state.view = shinkei.normalize_view(state.view)
    if not sys.stdin.isatty():
        return shinkei.render_static(np=np, state=state, framework=framework, width=96, height=28)

    use_plots = shinkei._supports_kitty_graphics()
    use_heatmap = shinkei._supports_graphical_terminal()
    fd = sys.stdin.fileno()
    active_framework = framework
    active_platform = _load_platform()
    motion_enabled = False
    motion_tick = 0
    script_index = _resolve_script_index(transform_selector)
    transforms = tuple(profile["transforms"][0] for profile in SCRIPT_PROFILES)
    transform_keys = {str(t["key"]) for t in transforms}
    pipeline: list[str] = [key for key in initial_pipeline if key in transform_keys]
    if not pipeline and transforms:
        pipeline = [str(transforms[script_index]["key"])]
    _apply_profile_to_state(state, script_index)
    old = termios.tcgetattr(fd)
    prev_sigint = signal.getsignal(signal.SIGINT)
    interrupt_requested = False

    def _handle_sigint(_signum, _frame) -> None:
        nonlocal interrupt_requested
        interrupt_requested = True

    signal.signal(signal.SIGINT, _handle_sigint)
    _enter_alt()
    tty.setcbreak(fd)
    needs_render = True
    tick_refresh = False
    dynamic_lines = 0
    show_details = False
    on_landing = True
    engine_cache: dict[str, FrameworkEngine] = {}

    def _engine_for(framework_name: str) -> FrameworkEngine:
        if framework_name not in engine_cache:
            engine_cache[framework_name] = FrameworkEngine(framework_name)
        return engine_cache[framework_name]

    try:
        while True:
            try:
                if interrupt_requested:
                    if on_landing:
                        break
                    interrupt_requested = False
                    on_landing = True
                    motion_enabled = False
                    show_details = False
                    needs_render = True
                    tick_refresh = False
                    dynamic_lines = 0
                    continue

                if needs_render:
                    if on_landing:
                        _clear_screen()
                        layout = _layout_for_view()
                        _render_landing(
                            framework=active_framework,
                            platform=active_platform,
                            width=layout["header_w"],
                            term_cols=layout["cols"],
                            term_rows=layout["rows"],
                        )
                        sys.stdout.flush()
                        needs_render = False
                        tick_refresh = False
                        dynamic_lines = 0
                        continue

                    profile = SCRIPT_PROFILES[script_index]
                    transform = transforms[script_index]
                    render_state = state
                    transform_label = "steady"
                    if motion_enabled:
                        render_state, transform_label = _live_motion_state(state, motion_tick)
                    engine = _engine_for(active_framework)
                    active_pipeline = tuple(pipeline)
                    result = engine.run_pipeline(active_pipeline, size=render_state.grid, steps=1)
                    arr = engine.to_numpy(result.final_tensor)
                    if arr is None:
                        arr, stage, caption = shinkei.stage_payload(np, render_state)
                        arr_f = np.asarray(arr, dtype=np.float32)
                    else:
                        stage = "+".join(active_pipeline) if active_pipeline else "field_init"
                        caption = (
                            f"{len(active_pipeline)} transform(s) executed on {active_framework} backend: "
                            f"{' -> '.join(active_pipeline) if active_pipeline else '(none)'}."
                        )
                        arr_f = np.asarray(arr, dtype=np.float32)
                    renderer = shinkei.renderer_name(use_plots, use_heatmap)
                    layout = _layout_for_view()

                    if use_plots:
                        png = shinkei._matplotlib_plot_png(arr_f, stage=stage, tensor_name=active_framework)
                        if png:
                            render = shinkei._kitty_from_png_bytes(
                                png,
                                cells_w=layout["plot_w"],
                                cells_h=layout["plot_h"],
                            )
                        elif use_heatmap:
                            render = shinkei._pixel_heatmap(
                                arr_f,
                                width=layout["plot_w"],
                                height=layout["plot_h"],
                            )
                        else:
                            render = shinkei._ascii_heatmap(
                                arr_f,
                                width=layout["ascii_w"],
                                height=layout["ascii_h"],
                            )
                    elif use_heatmap:
                        render = shinkei._pixel_heatmap(
                            arr_f,
                            width=layout["plot_w"],
                            height=layout["plot_h"],
                        )
                    else:
                        render = shinkei._ascii_heatmap(
                            arr_f,
                            width=layout["ascii_w"],
                            height=layout["ascii_h"],
                        )
                    caption_line = shinkei._format_caption(caption)
                    controls_line = _detailed_controls_line() if show_details else _compact_controls_line()
                    selector_lines = _render_pipeline_selector(transforms, pipeline, script_index)

                    partial_render_only = tick_refresh and motion_enabled and dynamic_lines > 0
                    if partial_render_only:
                        _cursor_up(dynamic_lines)
                        _clear_to_end()
                    else:
                        _clear_screen()
                        _render_header(
                            framework=active_framework,
                            platform=active_platform,
                            state=state,
                            renderer=renderer,
                            width=layout["header_w"],
                        )
                    print()
                    print(
                        _style(
                            f"transform [{transform['key']}] {transform['title']} · complexity={profile.get('complexity', '?')}",
                            fg=(200, 220, 255),
                            bold=True,
                        )
                    )
                    if show_details:
                        print(_style(f"formula: {transform['formula']}", fg=(159, 184, 228)))
                        print(_style(f"description: {transform['description']}", fg=(159, 184, 228)))
                        print()
                    live = "live" if motion_enabled else "paused"
                    live_color = (153, 231, 173) if motion_enabled else (255, 214, 136)
                    print(_style(f"live-dynamics: {live} · phase={transform_label}", fg=live_color, bold=True))
                    print()
                    if show_details:
                        for line in selector_lines:
                            print(_style(line, fg=(155, 180, 222)))
                    print(_style("pipeline: " + (" -> ".join(pipeline) if pipeline else "(none)"), fg=(136, 208, 251)))
                    if not show_details:
                        print(_style("hint: press [h] for transform details and full controls", fg=(140, 160, 192), dim=True))
                    print()

                    print(render)
                    if use_plots:
                        # Keep cursor below kitty image payload before footer text.
                        sys.stdout.write("\n")
                    print()
                    print(caption_line)
                    print(_style(controls_line, fg=(136, 160, 202)))
                    sys.stdout.flush()
                    dynamic_lines = _line_count(render) + 1 + _line_count(caption_line) + 1
                    if use_plots:
                        dynamic_lines += 1
                    needs_render = False
                    tick_refresh = False
                timeout = LIVE_REFRESH_SECONDS if motion_enabled else 5.0
                ch = _read_char(fd, timeout_s=timeout)
                if not ch:
                    if motion_enabled:
                        motion_tick += 1
                        needs_render = True
                        tick_refresh = True
                    continue
                if ch == "q":
                    break
                if ch == "\x03":
                    if on_landing:
                        break
                    on_landing = True
                    motion_enabled = False
                    show_details = False
                    needs_render = True
                    tick_refresh = False
                    dynamic_lines = 0
                    continue
                if on_landing:
                    if ch in {"\r", "\n"}:
                        on_landing = False
                        needs_render = True
                        tick_refresh = False
                    elif ch == "f":
                        chosen = _framework_selector(fd, active_framework)
                        if chosen and chosen != active_framework:
                            _persist_framework(chosen)
                            return _handoff_framework_switch(chosen)
                        needs_render = True
                        tick_refresh = False
                    elif ch == "p":
                        chosen_platform = _platform_selector(fd, active_platform)
                        if chosen_platform and chosen_platform != active_platform:
                            active_platform = chosen_platform
                            _persist_platform(active_platform)
                        needs_render = True
                        tick_refresh = False
                    continue
                if ch == "n":
                    script_index = _next_script_index(script_index, direction=1)
                    _apply_profile_to_state(state, script_index)
                    needs_render = True
                    tick_refresh = False
                elif ch == "b":
                    script_index = _next_script_index(script_index, direction=-1)
                    _apply_profile_to_state(state, script_index)
                    needs_render = True
                    tick_refresh = False
                elif ch == "x":
                    key = str(transforms[script_index]["key"])
                    _toggle_pipeline_key(pipeline, key)
                    needs_render = True
                    tick_refresh = False
                elif ch == "[":
                    key = str(transforms[script_index]["key"])
                    _move_pipeline_key(pipeline, key, direction=-1)
                    needs_render = True
                    tick_refresh = False
                elif ch == "]":
                    key = str(transforms[script_index]["key"])
                    _move_pipeline_key(pipeline, key, direction=1)
                    needs_render = True
                    tick_refresh = False
                elif ch == "r":
                    state.seed = (state.seed + 1) % 2_000_000_000
                    needs_render = True
                    tick_refresh = False
                elif ch == " ":
                    motion_enabled = not motion_enabled
                    needs_render = True
                    tick_refresh = False
                elif ch == "h":
                    show_details = not show_details
                    needs_render = True
                    tick_refresh = False
                elif ch == "i":
                    _guided_edit_state(fd, state, old)
                    needs_render = True
                    tick_refresh = False
                elif ch == "f":
                    chosen = _framework_selector(fd, active_framework)
                    if chosen and chosen != active_framework:
                        _persist_framework(chosen)
                        return _handoff_framework_switch(chosen)
                    needs_render = True
                    tick_refresh = False
                elif ch == "p":
                    chosen_platform = _platform_selector(fd, active_platform)
                    if chosen_platform and chosen_platform != active_platform:
                        active_platform = chosen_platform
                        _persist_platform(active_platform)
                    needs_render = True
                    tick_refresh = False
                elif ch == "e":
                    _quick_edit_state(fd, state, old)
                    needs_render = True
                    tick_refresh = False
                elif ch == ":":
                    if not _command_console(fd, old, state, active_framework, active_platform):
                        break
                    needs_render = True
                    tick_refresh = False
            except KeyboardInterrupt:
                if on_landing:
                    break
                on_landing = True
                motion_enabled = False
                show_details = False
                needs_render = True
                tick_refresh = False
                dynamic_lines = 0
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
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
        view=getattr(args, "view", shinkei.EXPLORER_VIEW),
        inputs=getattr(args, "inputs", None),
    )
    state.view = shinkei.normalize_view(state.view)
    transforms_selector = getattr(args, "transforms", None)
    transform_selector = getattr(args, "transform", None)
    try:
        transform_keys = resolve_transform_keys("all" if transforms_selector is None else transforms_selector)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    global SCRIPT_PROFILES
    SCRIPT_PROFILES = build_tui_profiles(transform_keys)

    script_index = _resolve_script_index(transform_selector)
    _apply_profile_to_state(state, script_index)
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
            print(shinkei._format_caption(caption))
            return 0
        return shinkei.render_static(
            np=np,
            state=state,
            framework=framework,
            width=width,
            height=height,
        )

    return _render_interactive(
        np=np,
        state=state,
        framework=framework,
        initial_pipeline=transform_keys,
        transform_selector=transform_selector,
    )


if __name__ == "__main__":
    raise SystemExit(main())
