#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from source_registry import (  # noqa: E402
    LOCK_FILE_NAME,
    install_items,
    load_json,
    provenance_status,
)
from source_resolution import (  # noqa: E402
    MERGE_BUNDLE_FILE,
    SourceRegistryError,
    accept_merge_bundle,
    export_merge_bundle,
)


class SourceResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.registry_root = root / "registry"
        self.consumer = root / "consumer"
        self.bundle_root = root / "bundle"
        self.registry_root.mkdir()
        (self.registry_root / "src").mkdir()
        (self.registry_root / "src" / "conflict.rs").write_text(
            "pub fn conflict() -> u32 { 1 }\n",
            encoding="utf-8",
        )
        (self.registry_root / "src" / "sibling.rs").write_text(
            "pub fn sibling() -> u32 { 1 }\n",
            encoding="utf-8",
        )
        (self.registry_root / "provenance.schema.json").write_text(
            "{}\n", encoding="utf-8"
        )
        (self.registry_root / "merge-bundle.schema.json").write_text(
            "{}\n", encoding="utf-8"
        )

        self.registry = {
            "$schema": "./registry.schema.json",
            "name": "rust-kernels",
            "version": 2,
            "description": "resolution fixture",
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
                    "name": "kernel",
                    "type": "registry:file",
                    "title": "Kernel",
                    "description": "fixture kernel",
                    "dependencies": [],
                    "registryDependencies": [],
                    "files": [
                        {
                            "path": "src/conflict.rs",
                            "target": "vendor/conflict.rs",
                        },
                        {
                            "path": "src/sibling.rs",
                            "target": "vendor/sibling.rs",
                        },
                    ],
                    "provides": [],
                }
            ],
        }
        self.registry_path = self.registry_root / "registry.json"
        self.write_registry()

        self.git("init", "-q")
        self.git("config", "user.name", "rust-kernels tests")
        self.git("config", "user.email", "tests@example.invalid")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "base")
        self.base_revision = self.git("rev-parse", "HEAD")

        install_items(
            self.registry_root,
            self.registry_path,
            self.registry,
            ["kernel"],
            self.consumer,
            self.base_revision,
        )
        self.generation = 0

    def git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.registry_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def write_registry(self) -> None:
        self.registry_path.write_text(
            json.dumps(self.registry, indent=2) + "\n",
            encoding="utf-8",
        )

    def candidate(
        self,
        *,
        conflict: str = "pub fn conflict() -> u32 { 2 }\n",
        sibling: str = "pub fn sibling() -> u32 { 2 }\n",
    ) -> str:
        (self.registry_root / "src" / "conflict.rs").write_text(
            conflict, encoding="utf-8"
        )
        (self.registry_root / "src" / "sibling.rs").write_text(
            sibling, encoding="utf-8"
        )
        self.write_registry()
        self.generation += 1
        (self.registry_root / "generation.txt").write_text(
            f"{self.generation}\n", encoding="utf-8"
        )
        self.git("add", ".")
        self.git("commit", "-q", "-m", f"candidate {self.generation}")
        return self.git("rev-parse", "HEAD")

    def make_conflict(self) -> None:
        (self.consumer / "vendor" / "conflict.rs").write_text(
            "pub fn local_conflict() -> u32 { 7 }\n",
            encoding="utf-8",
        )

    def lock(self) -> dict:
        return load_json(self.consumer / LOCK_FILE_NAME)

    def export(self, revision: str) -> dict:
        return export_merge_bundle(
            self.registry_root,
            self.registry_path,
            load_json(self.registry_path),
            self.lock(),
            self.consumer,
            self.bundle_root,
            revision,
            ["kernel"],
        )

    def accept(self, *, allow_ours: bool = False) -> None:
        accept_merge_bundle(
            self.registry_root,
            self.registry_path,
            load_json(self.registry_path),
            self.lock(),
            self.consumer,
            self.bundle_root / MERGE_BUNDLE_FILE,
            ["kernel"],
            allow_ours=allow_ours,
        )

    def test_export_materializes_exact_three_way_evidence(self) -> None:
        self.make_conflict()
        revision = self.candidate()

        manifest = self.export(revision)

        self.assertEqual(manifest["candidateRevision"], revision)
        self.assertTrue(
            manifest["$schema"].endswith(
                f"/{revision}/merge-bundle.schema.json"
            )
        )
        self.assertEqual(len(manifest["conflicts"]), 1)
        conflict = manifest["conflicts"][0]
        self.assertEqual(conflict["item"], "kernel")
        self.assertEqual(conflict["source"], "src/conflict.rs")
        self.assertEqual(conflict["target"], "vendor/conflict.rs")
        self.assertEqual(conflict["baseRevision"], self.base_revision)

        artifacts = conflict["artifacts"]
        self.assertEqual(
            (self.bundle_root / artifacts["base"]).read_text(encoding="utf-8"),
            "pub fn conflict() -> u32 { 1 }\n",
        )
        self.assertEqual(
            (self.bundle_root / artifacts["ours"]).read_text(encoding="utf-8"),
            "pub fn local_conflict() -> u32 { 7 }\n",
        )
        self.assertEqual(
            (self.bundle_root / artifacts["theirs"]).read_text(encoding="utf-8"),
            "pub fn conflict() -> u32 { 2 }\n",
        )

    def test_accept_preserves_merged_conflict_and_applies_safe_sibling(self) -> None:
        self.make_conflict()
        revision = self.candidate()
        self.export(revision)
        conflict_target = self.consumer / "vendor" / "conflict.rs"
        conflict_target.write_text(
            "pub fn merged_conflict() -> u32 { 9 }\n",
            encoding="utf-8",
        )

        self.accept()

        self.assertEqual(
            conflict_target.read_text(encoding="utf-8"),
            "pub fn merged_conflict() -> u32 { 9 }\n",
        )
        self.assertEqual(
            (self.consumer / "vendor" / "sibling.rs").read_text(encoding="utf-8"),
            "pub fn sibling() -> u32 { 2 }\n",
        )
        lock = self.lock()
        self.assertEqual(lock["items"][0]["revision"], revision)
        self.assertEqual(
            provenance_status(self.consumer, lock),
            [
                ("kernel", "vendor/conflict.rs", "modified"),
                ("kernel", "vendor/sibling.rs", "clean"),
            ],
        )

    def test_accept_converged_conflict_becomes_clean(self) -> None:
        self.make_conflict()
        revision = self.candidate()
        manifest = self.export(revision)
        conflict = manifest["conflicts"][0]
        theirs = self.bundle_root / conflict["artifacts"]["theirs"]
        (self.consumer / "vendor" / "conflict.rs").write_bytes(theirs.read_bytes())

        self.accept()

        lock = self.lock()
        self.assertEqual(lock["items"][0]["revision"], revision)
        self.assertEqual(
            provenance_status(self.consumer, lock),
            [
                ("kernel", "vendor/conflict.rs", "clean"),
                ("kernel", "vendor/sibling.rs", "clean"),
            ],
        )

    def test_accept_refuses_untouched_local_conflict_without_explicit_override(self) -> None:
        self.make_conflict()
        revision = self.candidate()
        self.export(revision)
        lock_before = (self.consumer / LOCK_FILE_NAME).read_text(encoding="utf-8")
        sibling_before = (self.consumer / "vendor" / "sibling.rs").read_text(
            encoding="utf-8"
        )

        with self.assertRaisesRegex(SourceRegistryError, "unchanged since export"):
            self.accept()

        self.assertEqual(
            (self.consumer / LOCK_FILE_NAME).read_text(encoding="utf-8"),
            lock_before,
        )
        self.assertEqual(
            (self.consumer / "vendor" / "sibling.rs").read_text(encoding="utf-8"),
            sibling_before,
        )

    def test_allow_ours_explicitly_carries_local_side_forward(self) -> None:
        self.make_conflict()
        revision = self.candidate()
        self.export(revision)

        self.accept(allow_ours=True)

        self.assertEqual(
            (self.consumer / "vendor" / "conflict.rs").read_text(encoding="utf-8"),
            "pub fn local_conflict() -> u32 { 7 }\n",
        )
        lock = self.lock()
        self.assertEqual(lock["items"][0]["revision"], revision)
        self.assertEqual(
            provenance_status(self.consumer, lock),
            [
                ("kernel", "vendor/conflict.rs", "modified"),
                ("kernel", "vendor/sibling.rs", "clean"),
            ],
        )

    def test_accept_rejects_stale_bundle_candidate(self) -> None:
        self.make_conflict()
        first_revision = self.candidate()
        self.export(first_revision)
        self.candidate(
            conflict="pub fn conflict() -> u32 { 3 }\n",
            sibling="pub fn sibling() -> u32 { 3 }\n",
        )
        lock_before = (self.consumer / LOCK_FILE_NAME).read_text(encoding="utf-8")

        with self.assertRaisesRegex(SourceRegistryError, "candidate revision"):
            self.accept()

        self.assertEqual(
            (self.consumer / LOCK_FILE_NAME).read_text(encoding="utf-8"),
            lock_before,
        )

    def test_accept_rejects_tampered_merge_artifact(self) -> None:
        self.make_conflict()
        revision = self.candidate()
        manifest = self.export(revision)
        ours = self.bundle_root / manifest["conflicts"][0]["artifacts"]["ours"]
        ours.write_text("tampered\n", encoding="utf-8")
        (self.consumer / "vendor" / "conflict.rs").write_text(
            "pub fn merged_conflict() -> u32 { 9 }\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(SourceRegistryError, "artifact hash"):
            self.accept()

    def test_export_refuses_when_no_real_conflict_exists(self) -> None:
        revision = self.candidate()

        with self.assertRaisesRegex(SourceRegistryError, "no both-changed"):
            self.export(revision)


if __name__ == "__main__":
    unittest.main()
