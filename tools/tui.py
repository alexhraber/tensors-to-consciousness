#!/usr/bin/env python3
"""Interactive terminal UI for tensors-to-consciousness visualization."""

from __future__ import annotations

import argparse
import json
import os
import select
import shutil
import subprocess
import sys
import termios
import tty
from types import ModuleType

from tools import runtime
from tools import shinkei

ULTRA_TRANSFORMS = (
    "signal warp",
    "field coupling",
    "spectral lift",
    "phase sweep",
)
ULTRA_REFRESH_SECONDS = 0.45
VIEW_ORDER = ("simplified", "advanced", "ultra")
SCRIPT_PROFILES = (
    {
        "id": "0",
        "title": "Computational Primitives",
        "algorithms": (
            {
                "key": "tensor_ops",
                "title": "Tensor Operations",
                "description": "Elementwise and matrix operations over structured tensors.",
                "formula": "C = A ⊙ B,  M = A @ B",
                "preset": {"samples": 900, "freq": 1.4, "amplitude": 0.9, "damping": 0.05, "noise": 0.06, "phase": 0.2, "grid": 72},
            },
            {
                "key": "norms",
                "title": "Norm Geometry",
                "description": "Magnitude and stability readout across tensor trajectories.",
                "formula": "||x||₂ = sqrt(sum_i x_i²)",
                "preset": {"samples": 980, "freq": 1.6, "amplitude": 1.0, "damping": 0.06, "noise": 0.05, "phase": 0.3, "grid": 76},
            },
        ),
    },
    {
        "id": "1",
        "title": "Automatic Differentiation",
        "algorithms": (
            {
                "key": "chain_rule",
                "title": "Chain Rule Field",
                "description": "Derivative amplification through nested nonlinearities.",
                "formula": "d/dx sin(x²) = 2x cos(x²)",
                "preset": {"samples": 1100, "freq": 1.9, "amplitude": 1.1, "damping": 0.08, "noise": 0.04, "phase": 0.55, "grid": 88},
            },
            {
                "key": "jacobian",
                "title": "Jacobian Sensitivity",
                "description": "Multivariate gradient coupling across dimensions.",
                "formula": "J_ij = ∂f_i/∂x_j",
                "preset": {"samples": 1180, "freq": 2.05, "amplitude": 1.05, "damping": 0.09, "noise": 0.05, "phase": 0.7, "grid": 90},
            },
        ),
    },
    {
        "id": "2",
        "title": "Optimization Theory",
        "algorithms": (
            {
                "key": "gradient_descent",
                "title": "Gradient Descent",
                "description": "Iterative descent dynamics over a curved objective.",
                "formula": "xₜ₊₁ = xₜ - η∇f(xₜ)",
                "preset": {"samples": 1200, "freq": 2.2, "amplitude": 1.0, "damping": 0.13, "noise": 0.10, "phase": 0.9, "grid": 96},
            },
            {
                "key": "momentum",
                "title": "Momentum Descent",
                "description": "Velocity-augmented traversal with inertia memory.",
                "formula": "vₜ₊₁ = βvₜ + ∇f(xₜ),  xₜ₊₁ = xₜ - ηvₜ₊₁",
                "preset": {"samples": 1280, "freq": 2.35, "amplitude": 1.1, "damping": 0.11, "noise": 0.12, "phase": 1.05, "grid": 98},
            },
            {
                "key": "adam",
                "title": "Adam Dynamics",
                "description": "Adaptive first/second moment optimization geometry.",
                "formula": "xₜ₊₁ = xₜ - η m̂ₜ / (sqrt(v̂ₜ)+ε)",
                "preset": {"samples": 1320, "freq": 2.5, "amplitude": 1.05, "damping": 0.12, "noise": 0.13, "phase": 1.15, "grid": 102},
            },
        ),
    },
    {
        "id": "3",
        "title": "Neural Theory",
        "algorithms": (
            {
                "key": "forward_pass",
                "title": "Forward Composition",
                "description": "Layered nonlinear transformation and feature shaping.",
                "formula": "y = σ(W₂ σ(W₁x + b₁) + b₂)",
                "preset": {"samples": 1300, "freq": 2.5, "amplitude": 1.2, "damping": 0.09, "noise": 0.12, "phase": 1.2, "grid": 104},
            },
            {
                "key": "activation_flow",
                "title": "Activation Flow",
                "description": "Activation distribution drift across depth.",
                "formula": "a_l = φ(W_l a_{l-1} + b_l)",
                "preset": {"samples": 1360, "freq": 2.7, "amplitude": 1.18, "damping": 0.10, "noise": 0.13, "phase": 1.35, "grid": 108},
            },
        ),
    },
    {
        "id": "4",
        "title": "Advanced Computational Theory",
        "algorithms": (
            {
                "key": "manifold_field",
                "title": "Manifold Field",
                "description": "Curved latent field with coupled oscillatory terms.",
                "formula": "z = exp(-λ||x||²) · sin(ωx) · cos(ωy)",
                "preset": {"samples": 1450, "freq": 2.9, "amplitude": 1.25, "damping": 0.11, "noise": 0.14, "phase": 1.55, "grid": 116},
            },
            {
                "key": "attention_surface",
                "title": "Attention Surface",
                "description": "Softmax geometry over query-key interactions.",
                "formula": "Attn(Q,K,V) = softmax(QK^T/√d)V",
                "preset": {"samples": 1500, "freq": 3.05, "amplitude": 1.2, "damping": 0.12, "noise": 0.15, "phase": 1.7, "grid": 120},
            },
        ),
    },
    {
        "id": "5",
        "title": "Research Frontiers",
        "algorithms": (
            {
                "key": "scaling_laws",
                "title": "Scaling Laws",
                "description": "Performance trends across model/data scale.",
                "formula": "L(N,D) ≈ A N^-α + B D^-β + C",
                "preset": {"samples": 1550, "freq": 3.2, "amplitude": 1.3, "damping": 0.15, "noise": 0.18, "phase": 1.9, "grid": 124},
            },
            {
                "key": "grokking",
                "title": "Grokking Transition",
                "description": "Delayed generalization phase shift dynamics.",
                "formula": "gen_gap(t) = L_test(t) - L_train(t)",
                "preset": {"samples": 1620, "freq": 3.35, "amplitude": 1.28, "damping": 0.16, "noise": 0.20, "phase": 2.05, "grid": 128},
            },
        ),
    },
    {
        "id": "6",
        "title": "Theoretical Limits",
        "algorithms": (
            {
                "key": "information_bound",
                "title": "Information Bound",
                "description": "Information transfer upper bounds under entropy constraints.",
                "formula": "I(X;Y) ≤ min(H(X), H(Y))",
                "preset": {"samples": 1700, "freq": 3.6, "amplitude": 1.35, "damping": 0.17, "noise": 0.20, "phase": 2.2, "grid": 132},
            },
            {
                "key": "thermo_learning",
                "title": "Thermodynamics of Learning",
                "description": "Energy/work dynamics across optimization states.",
                "formula": "ΔE = W - Q",
                "preset": {"samples": 1760, "freq": 3.75, "amplitude": 1.32, "damping": 0.18, "noise": 0.22, "phase": 2.35, "grid": 136},
            },
        ),
    },
)


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
    parser.add_argument(
        "--module",
        help="Initial module/profile selector (0..6 or partial module title).",
    )
    parser.add_argument(
        "--algorithm",
        help="Initial algorithm selector key/title for the selected module.",
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


def _field_specs(view: str) -> list[tuple[str, str, object]]:
    if view == "simplified":
        return [
            ("seed", "Random seed", int),
            ("freq", "Signal frequency", float),
            ("amplitude", "Signal amplitude", float),
        ]
    if view == "advanced":
        return [
            ("seed", "Random seed", int),
            ("samples", "Sample count", int),
            ("freq", "Signal frequency", float),
            ("amplitude", "Signal amplitude", float),
            ("damping", "Damping factor", float),
            ("noise", "Noise level", float),
        ]
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


def _guided_edit_state(fd: int, state: shinkei.VizState, old: list[int]) -> None:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    try:
        print("\nParameter Input")
        print(f"View: {state.view} (complexity-aware fields)")
        for key, label, caster in _field_specs(state.view):
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


def _quick_edit_state(fd: int, state: shinkei.VizState, old: list[int]) -> None:
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
    state: shinkei.VizState,
    framework: str,
    platform: str,
) -> bool:
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
                f"freq={state.freq:.3f} amp={state.amplitude:.3f} damping={state.damping:.3f} noise={state.noise:.3f} phase={state.phase:.3f} platform={platform}"
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
            env = os.environ.copy()
            env.update(_platform_env(framework, platform))
            subprocess.run(cmdline, check=False, env=env)
            continue
        print("Unknown command. Type `help`.")


def _persist_framework(framework: str) -> None:
    config: dict[str, str] = {}
    try:
        config = runtime.load_config()
    except RuntimeError:
        config = {"venv": ".venv"}
    config["framework"] = framework
    cfg_path = runtime.CONFIG_FILE
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


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
    config: dict[str, str] = {}
    try:
        config = runtime.load_config()
    except RuntimeError:
        config = {"venv": ".venv"}
    config["platform"] = platform
    cfg_path = runtime.CONFIG_FILE
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


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
    env: dict[str, str] = {"T2C_PLATFORM": platform}
    if framework == "jax":
        env["JAX_PLATFORM_NAME"] = platform
    if framework in {"pytorch", "keras", "cupy"} and platform == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    if framework == "keras":
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    return env


def _frame_line(text: str, width: int = 104) -> str:
    if len(text) >= width - 2:
        text = text[: width - 5] + "..."
    return "│" + text.ljust(width - 2) + "│"


def _render_header(
    framework: str,
    platform: str,
    state: shinkei.VizState,
    renderer: str,
    width: int = 104,
) -> None:
    top = "┌" + ("─" * (width - 2)) + "┐"
    bot = "└" + ("─" * (width - 2)) + "┘"
    print(top)
    print(
        _frame_line(
            f" t2c studio · {framework} · {platform} · view={state.view} · renderer={renderer} ",
            width=width,
        )
    )
    print(
        _frame_line(
            f" seed={state.seed} samples={state.samples} grid={state.grid} freq={state.freq:.3f} amp={state.amplitude:.3f} damp={state.damping:.3f} noise={state.noise:.3f} phase={state.phase:.3f} ",
            width=width,
        )
    )
    print(bot)


def _layout_for_view(view: str) -> dict[str, int]:
    size = shutil.get_terminal_size(fallback=(140, 44))
    cols = max(100, size.columns)
    rows = max(30, size.lines)

    header_w = max(92, min(cols - 2, 150))
    plot_w = max(72, min(cols - 4, 120))
    if view == "simplified":
        plot_h = max(16, min(rows - 14, 24))
    elif view == "advanced":
        plot_h = max(18, min(rows - 13, 28))
    else:
        plot_h = max(20, min(rows - 12, 32))

    return {
        "header_w": header_w,
        "plot_w": plot_w,
        "plot_h": plot_h,
        "ascii_w": max(72, min(plot_w, 120)),
        "ascii_h": max(16, min(plot_h, 30)),
    }


def _next_view(view: str, direction: int = 1) -> str:
    try:
        idx = VIEW_ORDER.index(view)
    except ValueError:
        idx = 0
    return VIEW_ORDER[(idx + direction) % len(VIEW_ORDER)]


def _profile_state(index: int, algo_index: int, seed: int = 7) -> shinkei.VizState:
    profile = SCRIPT_PROFILES[index % len(SCRIPT_PROFILES)]
    algo = profile["algorithms"][algo_index % len(profile["algorithms"])]
    preset = algo["preset"]
    return shinkei.VizState(
        seed=seed,
        samples=int(preset["samples"]),
        freq=float(preset["freq"]),
        amplitude=float(preset["amplitude"]),
        damping=float(preset["damping"]),
        noise=float(preset["noise"]),
        phase=float(preset["phase"]),
        grid=int(preset["grid"]),
        view="simplified",
    )


def _apply_profile_to_state(state: shinkei.VizState, index: int, algo_index: int) -> None:
    preset = _profile_state(index, algo_index, seed=state.seed)
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


def _next_algo_index(script_index: int, algo_index: int, direction: int = 1) -> int:
    n = len(SCRIPT_PROFILES[script_index]["algorithms"])
    return (algo_index + direction) % n


def _resolve_script_index(selector: str | None) -> int:
    if selector is None:
        return 0
    raw = selector.strip().lower()
    if not raw:
        return 0
    for i, profile in enumerate(SCRIPT_PROFILES):
        if raw == profile["id"]:
            return i
    for i, profile in enumerate(SCRIPT_PROFILES):
        if raw in profile["title"].lower():
            return i
    return 0


def _resolve_algo_index(script_index: int, selector: str | None) -> int:
    if selector is None:
        return 0
    raw = selector.strip().lower()
    if not raw:
        return 0
    algos = SCRIPT_PROFILES[script_index]["algorithms"]
    for i, algo in enumerate(algos):
        if raw == algo["key"]:
            return i
    for i, algo in enumerate(algos):
        if raw in algo["title"].lower():
            return i
    return 0


def _ultra_motion_state(base: shinkei.VizState, tick: int) -> tuple[shinkei.VizState, str]:
    phase_idx = (tick // 5) % len(ULTRA_TRANSFORMS)
    wave = (tick % 40) / 40.0
    if wave > 0.5:
        wave = 1.0 - wave
    wave = wave * 2.0

    state = shinkei.VizState(
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
    return state, ULTRA_TRANSFORMS[phase_idx]


def _render_interactive(
    np: ModuleType,
    state: shinkei.VizState,
    framework: str,
    module_selector: str | None = None,
    algo_selector: str | None = None,
) -> int:
    if not sys.stdin.isatty():
        return shinkei.render_static(np=np, state=state, framework=framework, width=96, height=28)

    use_plots = shinkei._supports_kitty_graphics()
    use_heatmap = shinkei._supports_graphical_terminal()
    fd = sys.stdin.fileno()
    active_framework = framework
    active_platform = _load_platform()
    motion_enabled = state.view != "ultra"
    motion_tick = 0
    script_index = _resolve_script_index(module_selector)
    algo_index = _resolve_algo_index(script_index, algo_selector)
    if state.view in {"advanced", "ultra"}:
        _apply_profile_to_state(state, script_index, algo_index)
    old = termios.tcgetattr(fd)
    _enter_alt()
    tty.setcbreak(fd)
    needs_render = True
    tick_refresh = False
    dynamic_lines = 0
    try:
        while True:
            if needs_render:
                profile = SCRIPT_PROFILES[script_index]
                algo = profile["algorithms"][algo_index]
                render_state = (
                    _profile_state(script_index, algo_index, seed=state.seed)
                    if state.view == "simplified"
                    else state
                )
                transform_label = "steady"
                if state.view == "ultra" and motion_enabled:
                    render_state, transform_label = _ultra_motion_state(state, motion_tick)
                arr, stage, caption = shinkei.stage_payload(np, render_state)
                arr_f = np.asarray(arr, dtype=np.float32)
                renderer = shinkei.renderer_name(use_plots, use_heatmap)
                layout = _layout_for_view(state.view)

                if use_plots:
                    png = shinkei._matplotlib_plot_png(arr_f, stage=stage, tensor_name=active_framework)
                    if png:
                        viz = shinkei._kitty_from_png_bytes(
                            png,
                            cells_w=layout["plot_w"],
                            cells_h=layout["plot_h"],
                        )
                    elif use_heatmap:
                        viz = shinkei._pixel_heatmap(
                            arr_f,
                            width=layout["plot_w"],
                            height=layout["plot_h"],
                        )
                    else:
                        viz = shinkei._ascii_heatmap(
                            arr_f,
                            width=layout["ascii_w"],
                            height=layout["ascii_h"],
                        )
                elif use_heatmap:
                    viz = shinkei._pixel_heatmap(
                        arr_f,
                        width=layout["plot_w"],
                        height=layout["plot_h"],
                    )
                else:
                    viz = shinkei._ascii_heatmap(
                        arr_f,
                        width=layout["ascii_w"],
                        height=layout["ascii_h"],
                    )
                caption_line = shinkei._format_caption(caption)
                controls_line = (
                    "Controls: [m] mode cycle  [n]/[b] module next/back  [a]/[A] algorithm next/back  [f] framework  [p] cpu/gpu  [i] guided input  [e] quick key=value  [space] pause/resume ultra motion  [:] command mode  [r] reseed  [q] quit"
                )

                partial_viz_only = (
                    tick_refresh and state.view == "ultra" and motion_enabled and dynamic_lines > 0
                )
                if partial_viz_only:
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
                    print(f"module [{profile['id']}] {profile['title']} · algorithm [{algo['key']}] {algo['title']}")
                    print(f"formula: {algo['formula']}")
                    print(f"description: {algo['description']}")
                    print()
                    if state.view == "ultra":
                        live = "live" if motion_enabled else "paused"
                        print(f"ultra-motion: {live} · transform={transform_label}")
                        print()

                print(viz)
                if use_plots:
                    # Keep cursor below kitty image payload before footer text.
                    sys.stdout.write("\n")
                print()
                print(caption_line)
                print(controls_line)
                sys.stdout.flush()
                dynamic_lines = _line_count(viz) + 1 + _line_count(caption_line) + 1
                if use_plots:
                    dynamic_lines += 1
                needs_render = False
                tick_refresh = False

            timeout = ULTRA_REFRESH_SECONDS if state.view == "ultra" and motion_enabled else 5.0
            ch = _read_char(fd, timeout_s=timeout)
            if not ch:
                if state.view == "ultra" and motion_enabled:
                    motion_tick += 1
                    needs_render = True
                    tick_refresh = True
                continue
            if ch == "q":
                break
            if ch == "m":
                state.view = _next_view(state.view, direction=1)
                if state.view == "ultra":
                    motion_enabled = False
                needs_render = True
                tick_refresh = False
            elif ch == "M":
                state.view = _next_view(state.view, direction=-1)
                if state.view == "ultra":
                    motion_enabled = False
                needs_render = True
                tick_refresh = False
            elif ch == "n":
                script_index = _next_script_index(script_index, direction=1)
                algo_index = 0
                if state.view in {"advanced", "ultra"}:
                    _apply_profile_to_state(state, script_index, algo_index)
                needs_render = True
                tick_refresh = False
            elif ch == "b":
                script_index = _next_script_index(script_index, direction=-1)
                algo_index = 0
                if state.view in {"advanced", "ultra"}:
                    _apply_profile_to_state(state, script_index, algo_index)
                needs_render = True
                tick_refresh = False
            elif ch == "a":
                algo_index = _next_algo_index(script_index, algo_index, direction=1)
                if state.view in {"advanced", "ultra"}:
                    _apply_profile_to_state(state, script_index, algo_index)
                needs_render = True
                tick_refresh = False
            elif ch == "A":
                algo_index = _next_algo_index(script_index, algo_index, direction=-1)
                if state.view in {"advanced", "ultra"}:
                    _apply_profile_to_state(state, script_index, algo_index)
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
            elif ch == "i":
                if state.view != "simplified":
                    _guided_edit_state(fd, state, old)
                needs_render = True
                tick_refresh = False
            elif ch == "f":
                chosen = _framework_selector(fd, active_framework)
                if chosen and chosen != active_framework:
                    active_framework = chosen
                    _persist_framework(active_framework)
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
                if state.view != "simplified":
                    _quick_edit_state(fd, state, old)
                needs_render = True
                tick_refresh = False
            elif ch == ":":
                if not _command_console(fd, old, state, active_framework, active_platform):
                    break
                needs_render = True
                tick_refresh = False
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
    module_selector = getattr(args, "module", None)
    algo_selector = getattr(args, "algorithm", None)
    script_index = _resolve_script_index(module_selector)
    algo_index = _resolve_algo_index(script_index, algo_selector)
    if state.view in {"advanced", "ultra"}:
        _apply_profile_to_state(state, script_index, algo_index)
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
        module_selector=module_selector,
        algo_selector=algo_selector,
    )


if __name__ == "__main__":
    raise SystemExit(main())
