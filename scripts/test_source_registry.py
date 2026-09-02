#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from source_registry import (  # noqa: E402
    LOCK_FILE_NAME,
    SourceRegistryError,
    install_items,
    load_json,
    provenance_status,
    resolve_items,
    sha256_file,
)


REVISION = "a" * 40


class SourceRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "registry"
        self.consumer = Path(self.temporary.name) / "consumer"
        (self.root / "src").mkdir(parents=True)
        (self.root / "src" / "base.rs").write_text("pub fn base() {}\n", encoding="utf-8")
        (self.root / "src" / "leaf.rs").write_text("pub fn leaf() {}\n", encoding="utf-8")

        self.registry = {
            "$schema": "./registry.schema.json",
            "name": "rust-kernels",
            "version": 2,
            "description": "fixture",
            "source": {
                "repository": "https://github.com/example/rust-kernels",
                "revisionType": "git-commit",
            },
            "provenance": {
                "schema": "https://raw.githubusercontent.com/moritzbrantner/rust-kernels/main/provenance.schema.json",
                "lockFile": ".rust-kernels.lock.json",
            },
            "items": [
                {
                    "name": "base",
                    "type": "registry:file",
                    "title": "Base",
                    "description": "base fixture",
                    "registryDependencies": [],
                    "files": [
                        {
                            "path": "src/base.rs",
                            "target": "vendor/base.rs",
                        }
                    ],
                },
                {
                    "name": "leaf",
                    "type": "registry:file",
                    "title": "Leaf",
                    "description": "leaf fixture",
                    "registryDependencies": ["base"],
                    "files": [
                        {
                            "path": "src/leaf.rs",
                            "target": "vendor/leaf.rs",
                        }
                    ],
                },
            ],
        }
        self.registry_path = self.root / "registry.json"
        self.registry_path.write_text(
            json.dumps(self.registry, indent=2) + "\n",
            encoding="utf-8",
        )

    def test_dependencies_install_before_requested_item(self) -> None:
        self.assertEqual(
            [item["name"] for item in resolve_items(self.registry, ["leaf"])],
            ["base", "leaf"],
        )

    def test_install_records_revision_hashes_and_clean_status(self) -> None:
        installed = install_items(
            self.root,
            self.registry_path,
            self.registry,
            ["leaf"],
            self.consumer,
            REVISION,
        )

        self.assertEqual(installed, ["base", "leaf"])
        lock = load_json(self.consumer / LOCK_FILE_NAME)
        self.assertEqual(
            lock["$schema"],
            "https://raw.githubusercontent.com/moritzbrantner/rust-kernels/"
            f"{REVISION}/provenance.schema.json",
        )
        self.assertEqual(lock["registry"]["name"], "rust-kernels")
        self.assertEqual([item["name"] for item in lock["items"]], ["base", "leaf"])
        self.assertTrue(all(item["revision"] == REVISION for item in lock["items"]))
        self.assertTrue(
            all(
                item["registrySha256"] == sha256_file(self.registry_path)
                for item in lock["items"]
            )
        )
        self.assertEqual(
            provenance_status(self.consumer, lock),
            [
                ("base", "vendor/base.rs", "clean"),
                ("leaf", "vendor/leaf.rs", "clean"),
            ],
        )

    def test_local_changes_are_reported_not_rewritten(self) -> None:
        install_items(
            self.root,
            self.registry_path,
            self.registry,
            ["leaf"],
            self.consumer,
            REVISION,
        )
        target = self.consumer / "vendor" / "leaf.rs"
        target.write_text("pub fn locally_specialized() {}\n", encoding="utf-8")
        lock = load_json(self.consumer / LOCK_FILE_NAME)

        self.assertIn(
            ("leaf", "vendor/leaf.rs", "modified"),
            provenance_status(self.consumer, lock),
        )

        with self.assertRaisesRegex(
            SourceRegistryError, "refusing to overwrite locally divergent file"
        ):
            install_items(
                self.root,
                self.registry_path,
                self.registry,
                ["leaf"],
                self.consumer,
                REVISION,
            )
        self.assertEqual(
            target.read_text(encoding="utf-8"),
            "pub fn locally_specialized() {}\n",
        )

    def test_missing_files_are_distinct_from_modified_files(self) -> None:
        install_items(
            self.root,
            self.registry_path,
            self.registry,
            ["base"],
            self.consumer,
            REVISION,
        )
        (self.consumer / "vendor" / "base.rs").unlink()
        lock = load_json(self.consumer / LOCK_FILE_NAME)
        self.assertEqual(
            provenance_status(self.consumer, lock),
            [("base", "vendor/base.rs", "missing")],
        )

    def test_invalid_revision_is_rejected_before_copying(self) -> None:
        with self.assertRaisesRegex(SourceRegistryError, "full lowercase 40-character"):
            install_items(
                self.root,
                self.registry_path,
                self.registry,
                ["base"],
                self.consumer,
                "main",
            )
        self.assertFalse((self.consumer / LOCK_FILE_NAME).exists())


if __name__ == "__main__":
    unittest.main()
