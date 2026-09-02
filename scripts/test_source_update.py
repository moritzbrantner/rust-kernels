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
from source_update import (  # noqa: E402
    SourceRegistryError,
    apply_safe_updates,
    plan_updates,
)


class SourceUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.registry_root = root / "registry"
        self.consumer = root / "consumer"
        self.registry_root.mkdir()
        (self.registry_root / "src").mkdir()
        (self.registry_root / "src" / "kernel.rs").write_text(
            "pub fn kernel() -> u32 { 1 }\n",
            encoding="utf-8",
        )
        (self.registry_root / "provenance.schema.json").write_text(
            "{}\n", encoding="utf-8"
        )

        self.registry = {
            "$schema": "./registry.schema.json",
            "name": "rust-kernels",
            "version": 2,
            "description": "update fixture",
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
                            "path": "src/kernel.rs",
                            "target": "vendor/kernel.rs",
                        }
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

    def commit_candidate(self, source: str | None = None) -> str:
        if source is not None:
            (self.registry_root / "src" / "kernel.rs").write_text(
                source,
                encoding="utf-8",
            )
        self.write_registry()
        self.generation += 1
        (self.registry_root / "generation.txt").write_text(
            f"{self.generation}\n", encoding="utf-8"
        )
        self.git("add", ".")
        self.git("commit", "-q", "-m", f"candidate {self.generation}")
        return self.git("rev-parse", "HEAD")

    def lock(self) -> dict:
        return load_json(self.consumer / LOCK_FILE_NAME)

    def states(self, lock: dict | None = None) -> list[str]:
        active_lock = lock if lock is not None else self.lock()
        plans = plan_updates(
            self.registry_root,
            load_json(self.registry_path),
            active_lock,
            self.consumer,
        )
        return [plan.state for plan in plans]

    def apply(self, revision: str, lock: dict | None = None) -> None:
        active_lock = lock if lock is not None else self.lock()
        apply_safe_updates(
            self.registry_root,
            self.registry_path,
            load_json(self.registry_path),
            active_lock,
            self.consumer,
            revision,
        )

    def test_upstream_only_update_is_applied_and_lock_advances(self) -> None:
        revision = self.commit_candidate("pub fn kernel() -> u32 { 2 }\n")

        self.assertEqual(self.states(), ["upstream-only"])
        self.apply(revision)

        self.assertEqual(
            (self.consumer / "vendor" / "kernel.rs").read_text(encoding="utf-8"),
            "pub fn kernel() -> u32 { 2 }\n",
        )
        lock = self.lock()
        self.assertEqual(lock["items"][0]["revision"], revision)
        self.assertEqual(
            provenance_status(self.consumer, lock),
            [("kernel", "vendor/kernel.rs", "clean")],
        )

    def test_local_only_update_keeps_specialization_and_advances_base(self) -> None:
        target = self.consumer / "vendor" / "kernel.rs"
        target.write_text("pub fn local_kernel() -> u32 { 7 }\n", encoding="utf-8")
        revision = self.commit_candidate()

        self.assertEqual(self.states(), ["local-only"])
        self.apply(revision)

        self.assertEqual(
            target.read_text(encoding="utf-8"),
            "pub fn local_kernel() -> u32 { 7 }\n",
        )
        lock = self.lock()
        self.assertEqual(lock["items"][0]["revision"], revision)
        self.assertEqual(
            provenance_status(self.consumer, lock),
            [("kernel", "vendor/kernel.rs", "modified")],
        )

    def test_converged_change_becomes_clean_at_candidate_revision(self) -> None:
        candidate = "pub fn kernel() -> u32 { 3 }\n"
        (self.consumer / "vendor" / "kernel.rs").write_text(candidate, encoding="utf-8")
        revision = self.commit_candidate(candidate)

        self.assertEqual(self.states(), ["converged"])
        self.apply(revision)

        lock = self.lock()
        self.assertEqual(lock["items"][0]["revision"], revision)
        self.assertEqual(
            provenance_status(self.consumer, lock),
            [("kernel", "vendor/kernel.rs", "clean")],
        )

    def test_both_changed_requires_manual_resolution_without_writes(self) -> None:
        target = self.consumer / "vendor" / "kernel.rs"
        target.write_text("pub fn local_kernel() -> u32 { 7 }\n", encoding="utf-8")
        revision = self.commit_candidate("pub fn kernel() -> u32 { 2 }\n")
        lock_before = (self.consumer / LOCK_FILE_NAME).read_text(encoding="utf-8")
        target_before = target.read_text(encoding="utf-8")

        self.assertEqual(self.states(), ["both-changed"])
        with self.assertRaisesRegex(SourceRegistryError, "manual resolution"):
            self.apply(revision)

        self.assertEqual(target.read_text(encoding="utf-8"), target_before)
        self.assertEqual(
            (self.consumer / LOCK_FILE_NAME).read_text(encoding="utf-8"),
            lock_before,
        )

    def test_missing_target_requires_manual_resolution(self) -> None:
        (self.consumer / "vendor" / "kernel.rs").unlink()
        revision = self.commit_candidate()
        lock_before = (self.consumer / LOCK_FILE_NAME).read_text(encoding="utf-8")

        self.assertEqual(self.states(), ["missing"])
        with self.assertRaisesRegex(SourceRegistryError, "manual resolution"):
            self.apply(revision)

        self.assertEqual(
            (self.consumer / LOCK_FILE_NAME).read_text(encoding="utf-8"),
            lock_before,
        )

    def test_registry_layout_change_is_never_applied_implicitly(self) -> None:
        self.registry["items"][0]["files"][0]["target"] = "vendor/renamed.rs"
        revision = self.commit_candidate()
        lock_before = (self.consumer / LOCK_FILE_NAME).read_text(encoding="utf-8")

        self.assertEqual(self.states(), ["layout-changed"])
        with self.assertRaisesRegex(SourceRegistryError, "manual resolution"):
            self.apply(revision)

        self.assertEqual(
            (self.consumer / LOCK_FILE_NAME).read_text(encoding="utf-8"),
            lock_before,
        )

    def test_registry_dependency_change_is_never_applied_implicitly(self) -> None:
        self.registry["items"][0]["registryDependencies"] = ["new-dependency"]
        revision = self.commit_candidate()

        self.assertEqual(self.states(), ["dependencies-changed"])
        with self.assertRaisesRegex(SourceRegistryError, "manual resolution"):
            self.apply(revision)

    def test_recorded_base_hash_is_verified_before_comparison(self) -> None:
        lock = self.lock()
        lock["items"][0]["files"][0]["sourceSha256"] = "0" * 64
        revision = self.commit_candidate()

        self.assertEqual(self.states(lock), ["base-mismatch"])
        with self.assertRaisesRegex(SourceRegistryError, "manual resolution"):
            self.apply(revision, lock)


if __name__ == "__main__":
    unittest.main()
