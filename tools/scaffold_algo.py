#!/usr/bin/env python3
"""Scaffold an abstract math transform and optional framework adapter stubs."""

from __future__ import annotations

import argparse
from pathlib import Path

FRAMEWORKS = ("numpy", "jax", "pytorch", "keras", "cupy", "mlx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create transform scaffold files.")
    parser.add_argument("--complexity", type=int, choices=range(0, 10), required=True, help="Complexity ordering level.")
    parser.add_argument("--key", required=True, help="Stable transform key (example: rk4_solver).")
    parser.add_argument("--title", required=True, help="Human title shown in registry/TUI.")
    parser.add_argument("--formula", required=True, help="Core equation/formula string.")
    parser.add_argument("--description", required=True, help="One-line conceptual description.")
    parser.add_argument("--root", default=".", help="Repository root (default: current directory).")
    return parser.parse_args()


def to_slug(key: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in key.strip().lower()).strip("_")


def write_once(path: Path, content: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def ensure_package(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    init_file = path / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    slug = to_slug(args.key)
    mod_prefix = f"c{args.complexity}_{slug}"

    ensure_package(root / "transforms" / "abstract")

    abstract_path = root / "transforms" / "abstract" / f"{mod_prefix}.py"
    abstract_content = f'''from __future__ import annotations

"""
Abstract math core for "{args.title}".
Formula: {args.formula}
"""

TRANSFORM_KEY = "{args.key}"
TITLE = "{args.title}"
FORMULA = "{args.formula}"
DESCRIPTION = "{args.description}"
COMPLEXITY = {args.complexity}


def math_core(ops, *, params: dict[str, object]) -> object:
    """
    Framework-agnostic core.
    - `ops` provides framework primitives (matmul, exp, sum, etc.).
    - `params` contains typed inputs/hyperparameters.
    """
    raise NotImplementedError("Implement abstract math_core first.")
'''
    write_once(abstract_path, abstract_content)

    for fw in FRAMEWORKS:
        ensure_package(root / "frameworks" / fw / "transforms")
        adapter_path = root / "frameworks" / fw / "transforms" / f"{mod_prefix}.py"
        adapter_content = f'''from __future__ import annotations

from transforms.abstract.{mod_prefix} import math_core


def run(*, params: dict[str, object]) -> object:
    """
    Framework adapter for {fw}.
    Wire an `ops` surface from frameworks.{fw}.utils into math_core().
    """
    raise NotImplementedError("Map frameworks.{fw}.utils primitives into an ops adapter.")
'''
        write_once(adapter_path, adapter_content)

    print("Scaffold created:")
    print(f"- {abstract_path.relative_to(root)}")
    for fw in FRAMEWORKS:
        print(f"- frameworks/{fw}/transforms/{mod_prefix}.py")

    print("\nTransform catalog snippet (append into transforms/transforms.json -> transforms):")
    print(
        "{"
        f'"key": "{args.key}", "title": "{args.title}", "description": "{args.description}", '
        f'"formula": "{args.formula}", "complexity": {args.complexity}, '
        '"source_module": "custom", "transform": "custom_impl", '
        '"preset": {"samples": 1200, "freq": 2.0, "amplitude": 1.0, "damping": 0.1, "noise": 0.1, "phase": 0.0, "grid": 96}'
        "},"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
