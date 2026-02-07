from __future__ import annotations

import importlib
import inspect
import json
import os
import unittest
from unittest.mock import patch


def _load_input_controls():
    return importlib.import_module("tools.input_controls")


class InputControlsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.input_controls = _load_input_controls()
        self.input_controls._META_REGISTRY.clear()
        self.input_controls._load_config.cache_clear()

    def test_contract_requires_viz_meta_dict(self) -> None:
        with self.assertRaises(ValueError):
            self.input_controls.metadata_for_scope("jax", {"x": object()})

    def test_generated_metadata_and_viz_meta_override(self) -> None:
        with patch.dict(os.environ, {"T2C_INPUTS": "{}"}, clear=False):
            _ = self.input_controls.tune_normal("jax", (2, 2))
        generated = None

        def _make() -> None:
            nonlocal generated
            generated = self.input_controls.tune_normal("jax", (2, 2))

        _make()
        self.assertIsNotNone(generated)
        meta = self.input_controls.metadata_for_scope("jax", {"generated": 1, "VIZ_META": {}})
        self.assertIn("generated", meta)
        self.assertIn("Generated via normal(", meta["generated"])

        meta2 = self.input_controls.metadata_for_scope(
            "jax",
            {"generated": 1, "VIZ_META": {"generated": "Manual transform annotation."}},
        )
        self.assertEqual(meta2["generated"], "Manual transform annotation.")

    def test_override_precedence_global_framework_script_call(self) -> None:
        line = inspect.currentframe().f_lineno + 2
        config = {
            "normal": {"std": 1.0},
            "frameworks": {
                "jax": {
                    "normal": {"std": 2.0},
                    "scripts": {
                        "test_input_controls.py": {
                            "normal": {"std": 3.0},
                            "calls": {
                                "sample_cfg": {"std": 4.0, "shape": [7]},
                                str(line): {"std": 5.0},
                            },
                        }
                    },
                }
            },
        }
        with patch.dict(os.environ, {"T2C_INPUTS": json.dumps(config)}, clear=False):
            self.input_controls._load_config.cache_clear()
            sample_cfg = self.input_controls.tune_normal("jax", (2, 2))
        self.assertEqual(sample_cfg["std"], 4.0)
        self.assertEqual(sample_cfg["shape"], (7,))

    def test_line_override_applies_when_label_missing(self) -> None:
        def _call_without_lhs():
            return self.input_controls.tune_normal("jax", (2, 2))

        line = _call_without_lhs.__code__.co_firstlineno + 1
        cfg = {
            "frameworks": {
                "jax": {
                    "scripts": {
                        "test_input_controls.py": {
                            "calls": {str(line): {"shape": [11], "std": 0.25}}
                        }
                    }
                }
            }
        }
        with patch.dict(os.environ, {"T2C_INPUTS": json.dumps(cfg)}, clear=False):
            self.input_controls._load_config.cache_clear()
            tuned = _call_without_lhs()
        self.assertEqual(tuned["shape"], (11,))
        self.assertEqual(tuned["std"], 0.25)


if __name__ == "__main__":
    unittest.main()

