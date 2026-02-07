from __future__ import annotations

from dataclasses import dataclass

from tools import diagnostics
from transforms.catalog import catalog_default_keys
from transforms.catalog import catalog_transforms

_LOGGER = diagnostics.get_logger("transforms.registry")


@dataclass(frozen=True)
class TransformPreset:
    samples: int
    freq: float
    amplitude: float
    damping: float
    noise: float
    phase: float
    grid: int


@dataclass(frozen=True)
class TransformSpec:
    key: str
    title: str
    description: str
    formula: str
    complexity: int
    source_module: str
    preset: TransformPreset


def _spec_from_entry(entry: dict[str, object]) -> TransformSpec:
    preset = entry["preset"]
    if not isinstance(preset, dict):
        raise ValueError(f"Invalid preset for transform {entry.get('key', '<unknown>')}")
    return TransformSpec(
        key=str(entry["key"]),
        title=str(entry["title"]),
        description=str(entry["description"]),
        formula=str(entry["formula"]),
        complexity=int(entry["complexity"]),
        source_module=str(entry["source_module"]),
        preset=TransformPreset(
            samples=int(preset["samples"]),
            freq=float(preset["freq"]),
            amplitude=float(preset["amplitude"]),
            damping=float(preset["damping"]),
            noise=float(preset["noise"]),
            phase=float(preset["phase"]),
            grid=int(preset["grid"]),
        ),
    )


TRANSFORM_SPECS: tuple[TransformSpec, ...] = tuple(_spec_from_entry(entry) for entry in catalog_transforms())
TRANSFORM_MAP = {spec.key: spec for spec in TRANSFORM_SPECS}
DEFAULT_TRANSFORM_KEYS: tuple[str, ...] = catalog_default_keys() or ("tensor_ops", "chain_rule", "gradient_descent")


def list_transform_keys() -> tuple[str, ...]:
    return tuple(spec.key for spec in TRANSFORM_SPECS)


def resolve_transform_keys(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        diagnostics.kernel_event(_LOGGER, "transforms.resolve", raw=raw, resolved=DEFAULT_TRANSFORM_KEYS)
        return DEFAULT_TRANSFORM_KEYS
    s = raw.strip().lower()
    if not s or s == "default":
        diagnostics.kernel_event(_LOGGER, "transforms.resolve", raw=raw, resolved=DEFAULT_TRANSFORM_KEYS)
        return DEFAULT_TRANSFORM_KEYS
    if s == "all":
        resolved = list_transform_keys()
        diagnostics.kernel_event(_LOGGER, "transforms.resolve", raw=raw, resolved=resolved)
        return resolved

    out: list[str] = []
    for part in s.split(","):
        key = part.strip()
        if not key:
            continue
        if key not in TRANSFORM_MAP:
            allowed = ", ".join(list_transform_keys())
            raise ValueError(f"Unknown transform '{key}'. Available: {allowed}")
        if key not in out:
            out.append(key)
    if not out:
        diagnostics.kernel_event(_LOGGER, "transforms.resolve", raw=raw, resolved=DEFAULT_TRANSFORM_KEYS)
        return DEFAULT_TRANSFORM_KEYS
    resolved = tuple(out)
    diagnostics.kernel_event(_LOGGER, "transforms.resolve", raw=raw, resolved=resolved)
    return resolved


def specs_for_keys(keys: tuple[str, ...]) -> tuple[TransformSpec, ...]:
    return tuple(TRANSFORM_MAP[k] for k in keys)


def build_tui_transforms(keys: tuple[str, ...] | None = None) -> tuple[dict[str, object], ...]:
    selected = specs_for_keys(keys if keys is not None else DEFAULT_TRANSFORM_KEYS)
    return tuple(
        {
            "key": spec.key,
            "title": spec.title,
            "description": spec.description,
            "formula": spec.formula,
            "complexity": spec.complexity,
            "source_module": spec.source_module,
            "preset": {
                "samples": spec.preset.samples,
                "freq": spec.preset.freq,
                "amplitude": spec.preset.amplitude,
                "damping": spec.preset.damping,
                "noise": spec.preset.noise,
                "phase": spec.preset.phase,
                "grid": spec.preset.grid,
            },
        }
        for spec in selected
    )


def build_tui_profiles(keys: tuple[str, ...] | None = None) -> tuple[dict[str, object], ...]:
    transforms = build_tui_transforms(keys)
    return tuple(
        {
            "id": t["key"],
            "title": t["title"],
            "complexity": t["complexity"],
            "transforms": (t,),
        }
        for t in transforms
    )
