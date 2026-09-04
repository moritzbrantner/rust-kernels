#!/usr/bin/env python3

import copy
import json
import subprocess
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
)
from source_update import apply_safe_updates, plan_updates  # noqa: E402


class ProvenanceContractTests(unittest.TestCase):
    """Contract and state-machine checks for consumer-owned source copies."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.registry_root = root / "registry"
        self.consumer = root / "consumer"
        self.registry_root.mkdir()
        (self.registry_root / "src").mkdir()
        self.write_source("base.rs", "pub fn base() -> u32 { 1 }\n")
        self.write_source("leaf.rs", "pub fn leaf() -> u32 { 1 }\n")
        self.write_source("sibling.rs", "pub fn sibling() -> u32 { 1 }\n")
        (self.registry_root / "provenance.schema.json").write_text(
            "{}\n", encoding="utf-8"
        )

        self.registry = {
            "$schema": "./registry.schema.json",
            "name": "rust-kernels",
            "version": 2,
            "description": "provenance contract fixture",
            "source": {
                "repository": "https://github.com/example/rust-kernels",
                "revisionType": "git-commit",
            },
            "provenance": {
                "schema": "https://raw.githubusercontent.com/moritzbrantner/rust-kernels/main/provenance.schema.json",
                "lockFile": ".rust-kernels.lock.json",
            },
            "items": [
                self.item("base", "src/base.rs", "vendor/base.rs"),
                self.item(
                    "leaf",
                    "src/leaf.rs",
                    "vendor/leaf.rs",
                    dependencies=["base"],
                ),
                self.item("sibling", "src/sibling.rs", "vendor/sibling.rs"),
                {
                    "name": "pair",
                    "type": "registry:file",
                    "title": "Pair",
                    "description": "two-file transaction fixture",
                    "dependencies": [],
                    "registryDependencies": [],
                    "files": [
                        {"path": "src/base.rs", "target": "pair/base.rs"},
                        {"path": "src/sibling.rs", "target": "pair/sibling.rs"},
                    ],
                    "provides": [],
                },
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
        self.generation = 0

    @staticmethod
    def item(
        name: str,
        source: str,
        target: str,
        *,
        dependencies: list[str] | None = None,
    ) -> dict:
        return {
            "name": name,
            "type": "registry:file",
            "title": name.title(),
            "description": f"{name} fixture",
            "dependencies": [],
            "registryDependencies": dependencies or [],
            "files": [{"path": source, "target": target}],
            "provides": [],
        }

    def git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.registry_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def write_source(self, name: str, content: str) -> None:
        (self.registry_root / "src" / name).write_text(content, encoding="utf-8")

    def write_registry(self) -> None:
        self.registry_path.write_text(
            json.dumps(self.registry, indent=2) + "\n",
            encoding="utf-8",
        )

    def item_by_name(self, name: str) -> dict:
        return next(item for item in self.registry["items"] if item["name"] == name)

    def install(self, *names: str) -> None:
        install_items(
            self.registry_root,
            self.registry_path,
            self.registry,
            names,
            self.consumer,
            self.base_revision,
        )

    def commit_candidate(self, source_updates: dict[str, str] | None = None) -> str:
        for name, content in (source_updates or {}).items():
            self.write_source(name, content)
        self.write_registry()
        self.generation += 1
        (self.registry_root / "generation.txt").write_text(
            f"{self.generation}\n", encoding="utf-8"
        )
        self.git("add", "-A")
        self.git("commit", "-q", "-m", f"candidate {self.generation}")
        return self.git("rev-parse", "HEAD")

    def lock(self) -> dict:
        return load_json(self.consumer / LOCK_FILE_NAME)

    def plans(self, lock: dict | None = None, requested: tuple[str, ...] = ()):
        return plan_updates(
            self.registry_root,
            load_json(self.registry_path),
            lock if lock is not None else self.lock(),
            self.consumer,
            requested,
        )

    def apply(
        self,
        revision: str,
        *,
        lock: dict | None = None,
        requested: tuple[str, ...] = (),
    ) -> None:
        apply_safe_updates(
            self.registry_root,
            self.registry_path,
            load_json(self.registry_path),
            lock if lock is not None else self.lock(),
            self.consumer,
            revision,
            requested,
        )

    def test_reinstall_is_byte_for_byte_idempotent(self) -> None:
        self.install("leaf")
        lock_before = (self.consumer / LOCK_FILE_NAME).read_bytes()
        files_before = {
            path: (self.consumer / path).read_bytes()
            for path in ("vendor/base.rs", "vendor/leaf.rs")
        }

        self.install("leaf")

        self.assertEqual((self.consumer / LOCK_FILE_NAME).read_bytes(), lock_before)
        for path, expected in files_before.items():
            self.assertEqual((self.consumer / path).read_bytes(), expected)

    def test_identical_preexisting_source_is_adopted_as_clean(self) -> None:
        target = self.consumer / "vendor" / "base.rs"
        target.parent.mkdir(parents=True)
        target.write_bytes((self.registry_root / "src" / "base.rs").read_bytes())
        modified_before = target.stat().st_mtime_ns

        self.install("base")

        self.assertEqual(target.stat().st_mtime_ns, modified_before)
        self.assertEqual(
            provenance_status(self.consumer, self.lock()),
            [("base", "vendor/base.rs", "clean")],
        )

    def test_dependency_cycle_fails_before_installation(self) -> None:
        self.item_by_name("base")["registryDependencies"] = ["leaf"]
        self.write_registry()

        with self.assertRaisesRegex(SourceRegistryError, "dependency cycle"):
            self.install("leaf")

        self.assertFalse(self.consumer.exists())

    def test_conflicting_target_mapping_fails_before_any_copy(self) -> None:
        self.item_by_name("sibling")["files"][0]["target"] = "vendor/leaf.rs"
        self.write_registry()

        with self.assertRaisesRegex(SourceRegistryError, "different content"):
            self.install("leaf", "sibling")

        self.assertFalse((self.consumer / "vendor" / "base.rs").exists())
        self.assertFalse((self.consumer / "vendor" / "leaf.rs").exists())
        self.assertFalse((self.consumer / LOCK_FILE_NAME).exists())

    def test_divergent_dependency_aborts_whole_install(self) -> None:
        target = self.consumer / "vendor" / "base.rs"
        target.parent.mkdir(parents=True)
        target.write_text("consumer owned\n", encoding="utf-8")

        with self.assertRaisesRegex(SourceRegistryError, "refusing to overwrite"):
            self.install("leaf")

        self.assertEqual(target.read_text(encoding="utf-8"), "consumer owned\n")
        self.assertFalse((self.consumer / "vendor" / "leaf.rs").exists())
        self.assertFalse((self.consumer / LOCK_FILE_NAME).exists())

    def test_target_path_escape_is_rejected_without_writes(self) -> None:
        self.item_by_name("base")["files"][0]["target"] = "../escape.rs"
        self.write_registry()
        escaped = self.consumer.parent / "escape.rs"

        with self.assertRaisesRegex(SourceRegistryError, "relative path"):
            self.install("base")

        self.assertFalse(escaped.exists())
        self.assertFalse((self.consumer / LOCK_FILE_NAME).exists())

    def test_existing_lock_cannot_be_reused_for_another_registry(self) -> None:
        self.install("base")
        lock_before = (self.consumer / LOCK_FILE_NAME).read_bytes()
        other_registry = copy.deepcopy(self.registry)
        other_registry["source"]["repository"] = "https://github.com/example/other"

        with self.assertRaisesRegex(SourceRegistryError, "different source registry"):
            install_items(
                self.registry_root,
                self.registry_path,
                other_registry,
                ["base"],
                self.consumer,
                self.base_revision,
            )

        self.assertEqual((self.consumer / LOCK_FILE_NAME).read_bytes(), lock_before)

    def test_status_reports_mixed_drift_states_deterministically(self) -> None:
        self.install("leaf", "sibling")
        (self.consumer / "vendor" / "leaf.rs").write_text(
            "pub fn local_leaf() -> u32 { 9 }\n", encoding="utf-8"
        )
        (self.consumer / "vendor" / "sibling.rs").unlink()

        self.assertEqual(
            provenance_status(self.consumer, self.lock()),
            [
                ("base", "vendor/base.rs", "clean"),
                ("leaf", "vendor/leaf.rs", "modified"),
                ("sibling", "vendor/sibling.rs", "missing"),
            ],
        )

    def test_status_rejects_malformed_recorded_fingerprint(self) -> None:
        self.install("base")
        lock = self.lock()
        lock["items"][0]["files"][0]["sourceSha256"] = "not-a-sha256"

        with self.assertRaisesRegex(SourceRegistryError, "sourceSha256 is invalid"):
            provenance_status(self.consumer, lock)

    def test_unchanged_candidate_advances_only_provenance(self) -> None:
        self.install("base")
        target = self.consumer / "vendor" / "base.rs"
        bytes_before = target.read_bytes()
        revision = self.commit_candidate()

        self.assertEqual([plan.state for plan in self.plans()], ["unchanged"])
        self.apply(revision)

        self.assertEqual(target.read_bytes(), bytes_before)
        self.assertEqual(self.lock()["items"][0]["revision"], revision)
        self.assertEqual(
            provenance_status(self.consumer, self.lock()),
            [("base", "vendor/base.rs", "clean")],
        )

    def test_registry_hash_tamper_blocks_update_without_writes(self) -> None:
        self.install("base")
        target = self.consumer / "vendor" / "base.rs"
        target_before = target.read_bytes()
        lock_file_before = (self.consumer / LOCK_FILE_NAME).read_bytes()
        lock = self.lock()
        lock["items"][0]["registrySha256"] = "0" * 64
        revision = self.commit_candidate({"base.rs": "pub fn base() -> u32 { 2 }\n"})

        self.assertEqual([plan.state for plan in self.plans(lock)], ["registry-mismatch"])
        with self.assertRaisesRegex(SourceRegistryError, "manual resolution"):
            self.apply(revision, lock=lock)

        self.assertEqual(target.read_bytes(), target_before)
        self.assertEqual((self.consumer / LOCK_FILE_NAME).read_bytes(), lock_file_before)

    def test_lock_mapping_tamper_is_detected_before_comparison(self) -> None:
        self.install("base")
        lock = self.lock()
        lock["items"][0]["files"][0]["target"] = "vendor/renamed.rs"
        self.commit_candidate()

        self.assertEqual([plan.state for plan in self.plans(lock)], ["lock-mismatch"])

    def test_removed_registry_item_is_an_explicit_blocker(self) -> None:
        self.install("base")
        target_before = (self.consumer / "vendor" / "base.rs").read_bytes()
        lock_before = (self.consumer / LOCK_FILE_NAME).read_bytes()
        self.registry["items"] = [
            item for item in self.registry["items"] if item["name"] != "base"
        ]
        revision = self.commit_candidate()

        self.assertEqual([plan.state for plan in self.plans()], ["item-removed"])
        with self.assertRaisesRegex(SourceRegistryError, "manual resolution"):
            self.apply(revision)

        self.assertEqual((self.consumer / "vendor" / "base.rs").read_bytes(), target_before)
        self.assertEqual((self.consumer / LOCK_FILE_NAME).read_bytes(), lock_before)

    def test_mixed_safe_and_conflicting_files_abort_atomically(self) -> None:
        self.install("pair")
        base_target = self.consumer / "pair" / "base.rs"
        sibling_target = self.consumer / "pair" / "sibling.rs"
        base_before = base_target.read_bytes()
        lock_before = (self.consumer / LOCK_FILE_NAME).read_bytes()
        sibling_target.write_text(
            "pub fn local_sibling() -> u32 { 7 }\n", encoding="utf-8"
        )
        revision = self.commit_candidate(
            {
                "base.rs": "pub fn base() -> u32 { 2 }\n",
                "sibling.rs": "pub fn sibling() -> u32 { 2 }\n",
            }
        )

        self.assertEqual(
            [(plan.target, plan.state) for plan in self.plans()],
            [("pair/base.rs", "upstream-only"), ("pair/sibling.rs", "both-changed")],
        )
        with self.assertRaisesRegex(SourceRegistryError, "manual resolution"):
            self.apply(revision)

        self.assertEqual(base_target.read_bytes(), base_before)
        self.assertEqual(
            sibling_target.read_text(encoding="utf-8"),
            "pub fn local_sibling() -> u32 { 7 }\n",
        )
        self.assertEqual((self.consumer / LOCK_FILE_NAME).read_bytes(), lock_before)

    def test_selected_item_update_leaves_other_item_and_provenance_unchanged(self) -> None:
        self.install("base", "sibling")
        revision = self.commit_candidate(
            {
                "base.rs": "pub fn base() -> u32 { 2 }\n",
                "sibling.rs": "pub fn sibling() -> u32 { 2 }\n",
            }
        )

        self.apply(revision, requested=("base",))

        self.assertEqual(
            (self.consumer / "vendor" / "base.rs").read_text(encoding="utf-8"),
            "pub fn base() -> u32 { 2 }\n",
        )
        self.assertEqual(
            (self.consumer / "vendor" / "sibling.rs").read_text(encoding="utf-8"),
            "pub fn sibling() -> u32 { 1 }\n",
        )
        revisions = {item["name"]: item["revision"] for item in self.lock()["items"]}
        self.assertEqual(revisions["base"], revision)
        self.assertEqual(revisions["sibling"], self.base_revision)

    def test_duplicate_update_selection_is_rejected(self) -> None:
        self.install("base")
        self.commit_candidate()

        with self.assertRaisesRegex(SourceRegistryError, "contains duplicates"):
            self.plans(requested=("base", "base"))

    def test_advanced_local_base_becomes_next_three_way_ancestor(self) -> None:
        self.install("base")
        target = self.consumer / "vendor" / "base.rs"
        target.write_text("pub fn local_base() -> u32 { 7 }\n", encoding="utf-8")
        first_revision = self.commit_candidate()

        self.assertEqual([plan.state for plan in self.plans()], ["local-only"])
        self.apply(first_revision)
        self.assertEqual(self.lock()["items"][0]["revision"], first_revision)

        self.commit_candidate({"base.rs": "pub fn base() -> u32 { 2 }\n"})

        self.assertEqual([plan.state for plan in self.plans()], ["both-changed"])
        self.assertEqual(
            target.read_text(encoding="utf-8"),
            "pub fn local_base() -> u32 { 7 }\n",
        )


if __name__ == "__main__":
    unittest.main()
