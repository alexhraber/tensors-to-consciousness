from __future__ import annotations

import unittest

from transforms import registry


class TransformRegistryTests(unittest.TestCase):
    def test_registry_has_unique_transform_keys(self) -> None:
        keys = [spec.key for spec in registry.TRANSFORM_SPECS]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertGreater(len(keys), 0)

    def test_default_transform_keys_resolve(self) -> None:
        keys = registry.resolve_transform_keys("default")
        self.assertEqual(keys, registry.DEFAULT_TRANSFORM_KEYS)
        self.assertTrue(all(k in registry.TRANSFORM_MAP for k in keys))

    def test_tui_profiles_align_with_selection(self) -> None:
        keys = ("tensor_ops", "adam")
        profiles = registry.build_tui_profiles()
        self.assertGreaterEqual(len(profiles), 1)
        selected_profiles = registry.build_tui_profiles(keys)
        self.assertEqual(tuple(p["id"] for p in selected_profiles), keys)
        self.assertTrue(all(p["transforms"] for p in selected_profiles))


if __name__ == "__main__":
    unittest.main()
