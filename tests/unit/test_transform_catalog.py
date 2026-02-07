from __future__ import annotations

import unittest

from algos.catalog import catalog_default_keys
from algos.catalog import catalog_framework_interface
from algos.catalog import catalog_transforms
from algos.definitions import TRANSFORM_IMPLS


class TransformCatalogTests(unittest.TestCase):
    def test_catalog_defaults_exist_in_transforms(self) -> None:
        keys = {str(entry["key"]) for entry in catalog_transforms()}
        for key in catalog_default_keys():
            self.assertIn(key, keys)

    def test_catalog_entries_define_known_transform_impl(self) -> None:
        for entry in catalog_transforms():
            with self.subTest(key=entry.get("key")):
                impl = str(entry.get("transform", ""))
                self.assertIn(impl, TRANSFORM_IMPLS)

    def test_framework_interface_contract_shape(self) -> None:
        contract = catalog_framework_interface()
        self.assertIn("utils_entrypoints", contract)
        self.assertIn("ops_adapter", contract)
        self.assertIn("normal", contract["utils_entrypoints"])
        self.assertIn("add", contract["ops_adapter"])
        self.assertIn("normal_like", contract["ops_adapter"])


if __name__ == "__main__":
    unittest.main()
