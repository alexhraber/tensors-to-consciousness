from __future__ import annotations

import argparse
import importlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


@dataclass
class VizState:
    seed: int = 7
    samples: int = 1200
    freq: float = 1.9
    amplitude: float = 1.0
    damping: float = 0.09
    noise: float = 0.13
    phase: float = 0.6
    grid: int = 96
    view: str = "advanced"


def to_ascii(rgba, width: int, height: int) -> str:
    ramp = " .:-=+*#%@"
    r = rgba[..., 0].astype("float32")
    g = rgba[..., 1].astype("float32")
    b = rgba[..., 2].astype("float32")
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    luminance = 255.0 - luminance
    if luminance.max() > 0:
        luminance = luminance / luminance.max()

    h = max(1, height)
    w = max(1, width)
    y_idx = [int(i * luminance.shape[0] / h) for i in range(h)]
    x_idx = [int(i * luminance.shape[1] / w) for i in range(w)]
    lines: list[str] = []
    for yy in y_idx:
        row = []
        for xx in x_idx:
            v = luminance[yy, xx]
            row.append(ramp[min(int(v * (len(ramp) - 1)), len(ramp) - 1)])
        lines.append("".join(row))
    return "\n".join(lines)


def _load_inputs_blob(raw: str | None) -> dict[str, object]:
    if not raw:
        return {}
    raw_s = raw.strip()
    if not raw_s:
        return {}
    try:
        p = Path(raw_s)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    try:
        return json.loads(raw_s)
    except json.JSONDecodeError:
        return {}


def _coerce_int(data: dict[str, object], key: str, default: int, lo: int, hi: int) -> int:
    val = data.get(key, default)
    try:
        out = int(val)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, out))


def _coerce_float(data: dict[str, object], key: str, default: float, lo: float, hi: float) -> float:
    val = data.get(key, default)
    try:
        out = float(val)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, out))


def build_state(view: str, inputs: str | None = None) -> VizState:
    merged = _load_inputs_blob(inputs)
    env_blob = os.environ.get("T2C_INPUTS", "").strip()
    if env_blob:
        merged = {**merged, **_load_inputs_blob(env_blob)}
    return VizState(
        seed=_coerce_int(merged, "seed", 7, 0, 2_000_000_000),
        samples=_coerce_int(merged, "samples", 1200, 64, 12000),
        freq=_coerce_float(merged, "freq", 1.9, 0.1, 20.0),
        amplitude=_coerce_float(merged, "amplitude", 1.0, 0.01, 20.0),
        damping=_coerce_float(merged, "damping", 0.09, 0.0, 2.0),
        noise=_coerce_float(merged, "noise", 0.13, 0.0, 3.0),
        phase=_coerce_float(merged, "phase", 0.6, -6.28, 6.28),
        grid=_coerce_int(merged, "grid", 96, 24, 320),
        view=view,
    )


def normalize_state(state: VizState) -> None:
    clamped = build_state(view=state.view, inputs=json.dumps(state.__dict__))
    state.seed = clamped.seed
    state.samples = clamped.samples
    state.freq = clamped.freq
    state.amplitude = clamped.amplitude
    state.damping = clamped.damping
    state.noise = clamped.noise
    state.phase = clamped.phase
    state.grid = clamped.grid


def state_json(state: VizState) -> str:
    payload = {
        "seed": state.seed,
        "samples": state.samples,
        "freq": state.freq,
        "amplitude": state.amplitude,
        "damping": state.damping,
        "noise": state.noise,
        "phase": state.phase,
        "grid": state.grid,
    }
    return json.dumps(payload)


def stage_payload(np: ModuleType, state: VizState) -> tuple[object, str, str]:
    rng = np.random.default_rng(state.seed)
    x = np.linspace(0.0, 4.0 * np.pi, state.samples, dtype=np.float32)
    envelope = np.exp(-state.damping * x).astype(np.float32)
    base = (state.amplitude * np.sin(state.freq * x + state.phase)).astype(np.float32)
    mod = (0.42 * np.cos(0.5 * state.freq * x - state.phase)).astype(np.float32)
    noisy = (base * envelope + mod + state.noise * rng.normal(size=state.samples)).astype(np.float32)

    if state.view == "simplified":
        caption = "Damped oscillation with controlled noise envelope."
        return noisy, "simplified", caption

    g = state.grid
    y = np.linspace(-2.0, 2.0, g, dtype=np.float32)
    xx, yy = np.meshgrid(y, y)
    field = (
        np.sin((state.freq + 0.35) * xx + state.phase)
        * np.cos((state.freq - 0.2) * yy - state.phase)
        * np.exp(-state.damping * (xx**2 + yy**2))
    ).astype(np.float32)
    field += (state.noise * 0.2 * rng.normal(size=(g, g))).astype(np.float32)
    if state.view == "advanced":
        caption = "Tensor field showing coupled wave interference and decay."
        return field, "advanced", caption

    spec = np.abs(np.fft.rfft(noisy - np.mean(noisy))).astype(np.float32)
    take = min(g, spec.shape[0])
    profile = spec[:take]
    ultra = np.outer(np.linspace(0.2, 1.0, g, dtype=np.float32), profile).astype(np.float32)
    ultra += field[:g, :take]
    caption = "Multi-scale energy map combining spatial field and frequency response."
    return ultra, "ultra", caption


def renderer_name(use_plots: bool, use_heatmap: bool) -> str:
    if use_plots:
        return "plots"
    if use_heatmap:
        return "synthesized-heatmap"
    return "ascii"


def load_common_viz() -> ModuleType:
    return importlib.import_module("tools.common_viz")


def render_static(
    *,
    np: ModuleType,
    state: VizState,
    framework: str,
    width: int,
    height: int,
) -> int:
    common_viz = load_common_viz()
    arr, stage, caption = stage_payload(np, state)
    arr_f = np.asarray(arr, dtype=np.float32)
    use_plots = common_viz._supports_kitty_graphics()
    use_heatmap = common_viz._supports_graphical_terminal()
    print(f"[VIS renderer={renderer_name(use_plots, use_heatmap)}]")
    print(f"[VIS:{framework}] {stage}")

    if use_plots:
        png = common_viz._matplotlib_plot_png(arr_f, stage=stage, tensor_name="interactive")
        if png:
            print(common_viz._kitty_from_png_bytes(png, cells_w=72, cells_h=22))
        elif use_heatmap:
            print(common_viz._pixel_heatmap(arr_f, width=72, height=30))
        else:
            print(common_viz._ascii_heatmap(arr_f, width=width, height=height))
    elif use_heatmap:
        print(common_viz._pixel_heatmap(arr_f, width=72, height=30))
    else:
        print(common_viz._ascii_heatmap(arr_f, width=width, height=height))
    print(common_viz._format_caption(caption))
    return 0


def render_non_tty_ascii(
    *,
    mpimg: ModuleType,
    np: ModuleType,
    state: VizState,
    width: int,
    height: int,
) -> tuple[str, str]:
    arr, stage, caption = stage_payload(np, state)
    common_viz = load_common_viz()
    png = common_viz._matplotlib_plot_png(
        np.asarray(arr, dtype=np.float32),
        stage=stage,
        tensor_name="static",
    )
    if png is None:
        return "", caption
    rgba = mpimg.imread(io.BytesIO(png))
    if rgba.dtype != "uint8":
        rgba = (rgba * 255).astype("uint8")
    return to_ascii(rgba, width=width, height=height), caption
