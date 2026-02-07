#!/usr/bin/env python3
"""Algorithm playground runner using framework engines."""

from __future__ import annotations

import argparse

from algos.registry import resolve_algorithm_keys
from frameworks.engine import FrameworkEngine
from tools import shinkei
from tools.runtime import SUPPORTED_FRAMEWORKS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a framework engine over an ordered algorithm pipeline.")
    parser.add_argument("--framework", choices=list(SUPPORTED_FRAMEWORKS), required=True)
    parser.add_argument("--algos", default="default", help="Comma-separated algo keys, or 'default'/'all'.")
    parser.add_argument("--size", type=int, default=96, help="Square tensor size for sandbox execution.")
    parser.add_argument("--steps", type=int, default=1, help="Pipeline passes to apply.")
    parser.add_argument("--viz", action="store_true", help="Render visualization for final tensor.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keys = resolve_algorithm_keys(args.algos)
    engine = FrameworkEngine(args.framework)
    result = engine.run_pipeline(keys, size=args.size, steps=args.steps)

    print(f"framework={args.framework}")
    print(f"algos={','.join(keys)}")
    if result.trace:
        last = result.trace[-1]
        print(f"final shape={last['shape']} mean={last['mean']:.4f} std={last['std']:.4f}")

    if args.viz:
        scope = {
            "pipeline_output": result.final_tensor,
            "VIZ_META": {
                "pipeline_output": f"Ordered pipeline on {args.framework}: {', '.join(keys)}",
            },
        }
        shinkei.viz_stage(
            stage="playground_final",
            scope=scope,
            to_numpy=engine.to_numpy,
            framework=args.framework,
            metadata=scope["VIZ_META"],
            limit=1,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
