from __future__ import annotations

import unittest

from algos import registry


class AlgoRegistryTests(unittest.TestCase):
    def test_registry_has_unique_algo_keys(self) -> None:
        keys = [spec.key for spec in registry.ALGORITHM_SPECS]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertGreater(len(keys), 0)

    def test_default_algo_keys_resolve(self) -> None:
        keys = registry.resolve_algorithm_keys("default")
        self.assertEqual(keys, registry.DEFAULT_ALGO_KEYS)
        self.assertTrue(all(k in registry.ALGO_MAP for k in keys))

    def test_tui_profiles_align_with_selection(self) -> None:
        keys = ("tensor_ops", "adam")
        profiles = registry.build_tui_profiles()
        self.assertGreaterEqual(len(profiles), 1)
        selected_profiles = registry.build_tui_profiles(keys)
        self.assertEqual(tuple(p["id"] for p in selected_profiles), keys)
        self.assertTrue(all(p["algorithms"] for p in selected_profiles))


if __name__ == "__main__":
    unittest.main()
