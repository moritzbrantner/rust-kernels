#!/usr/bin/env python3
"""Real catalog, real copied sources, Cargo consumer and provenance lifecycle.

No generated or rewritten upstream manifest is substituted for an installed file.
The fixture supplies only the consumer-owned workspace and application wiring.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib
import unittest

from source_registry import (
    LOCK_FILE_NAME, SourceRegistryError, install_items, load_json,
    provenance_status, resolve_items, sha256_file,
)
from source_update import apply_safe_updates, plan_updates
from source_resolution import MERGE_BUNDLE_FILE, accept_merge_bundle, export_merge_bundle

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = load_json(ROOT / "registry.json")
ITEMS = {item["name"]: item for item in REGISTRY["items"]}


def run(command: list[str], cwd: Path, **kwargs) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=300, **kwargs)
    if result.returncode:
        raise AssertionError(f"{' '.join(command)} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


class CatalogInstallTests(unittest.TestCase):
    def test_every_registered_crate_includes_sources_and_declared_cargo_targets(self):
        for item in REGISTRY["items"]:
            if item["type"] != "registry:crate":
                continue
            manifest = ROOT / item["crate"]["manifest"]
            paths = {entry["path"] for entry in item["files"]}
            for source in (manifest.parent / "src").rglob("*.rs"):
                self.assertIn(source.relative_to(ROOT).as_posix(), paths)
            data = tomllib.loads(manifest.read_text())
            for kind, folder in [("bench", "benches"), ("test", "tests"), ("example", "examples"), ("bin", "src/bin")]:
                for target in data.get(kind, []):
                    path = manifest.parent / target.get("path", f"{folder}/{target['name']}.rs")
                    self.assertIn(path.relative_to(ROOT).as_posix(), paths,
                                  f"{item['name']} installs a manifest with an absent {kind} target")
        for name in ["search-kernels", "selection"]:
            self.assertIn("top_k_by", {s for p in ITEMS[name]["provides"] for s in p["symbols"]})
        self.assertEqual(ITEMS["similarity-kernels"]["registryDependencies"], [])
        self.assertEqual(ITEMS["similarity-kernels"]["dependencies"], [])
        expected = {"levenshtein", "bounded-levenshtein", "sorted-unique-sets", "jaccard",
                    "sorted-k-way-merge", "shingles-rolling-hash", "minhash", "simhash", "bk-tree", "myers-distance"}
        self.assertEqual({p["name"] for p in ITEMS["similarity-kernels"]["provides"]}, expected)

    def test_fresh_consumer_builds_exact_installed_manifests_and_all_advertised_symbols(self):
        with tempfile.TemporaryDirectory(prefix="kernel-catalog-consumer-") as temporary:
            consumer = Path(temporary)
            names = [i["name"] for i in REGISTRY["items"] if i["type"] == "registry:crate"]
            installed = install_items(ROOT, ROOT / "registry.json", REGISTRY, names, consumer, "a" * 40)
            members = [str(Path(ITEMS[name]["crate"]["manifest"]).parent) for name in installed]
            package = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]
            (consumer / "Cargo.toml").write_text(
                '[workspace]\nresolver="2"\nmembers=' + json.dumps(["app", *members]) + '\n[workspace.package]\n'
                + '\n'.join(f'{key}={json.dumps(value)}' for key, value in package.items()) + '\n'
            )
            app = consumer / "app"
            (app / "src").mkdir(parents=True)
            (app / "Cargo.toml").write_text(
                '[package]\nname="catalog-consumer"\nversion="0.0.0"\nedition="2024"\n[dependencies]\n'
                + '\n'.join(f'{name}={{path="../crates/{name}"}}' for name in installed) + '\n'
            )
            imports = []
            for name in installed:
                symbols = sorted({s for provider in ITEMS[name]["provides"] for s in provider["symbols"]})
                imports.append(f'#[allow(unused_imports)] use {name.replace("-", "_")}::{{{", ".join(symbols)}}};')
            (app / "src/main.rs").write_text('\n'.join(imports) + r'''
fn main() {
    use similarity_kernels as s;
    assert_eq!(s::levenshtein(b"kitten", b"sitting"), 3);
    assert_eq!(s::levenshtein_bounded(b"kitten", b"sitting", 2), None);
    assert_eq!(s::LevenshteinWorkspace::new().distance(b"kitten", b"sitting", 3), Some(3));
    assert_eq!(s::sorted_unique_union(&[1,3], &[2,3]), [1,2,3]);
    assert_eq!(s::sorted_unique_intersection(&[1,3], &[2,3]), [3]);
    assert_eq!(s::sorted_unique_difference(&[1,3], &[2,3]), [1]);
    assert_eq!(s::sorted_unique_symmetric_difference(&[1,3], &[2,3]), [1,2]);
    assert_eq!(s::sorted_unique_intersection_count(&[1,3], &[2,3]), 1);
    assert_eq!(s::sorted_unique_union_count(&[1,3], &[2,3]), 3);
    assert_eq!(s::jaccard_similarity_sorted_unique(&[1,3], &[2,3]), 1.0/3.0);
    assert_eq!(s::jaccard_distance_sorted_unique(&[1,3], &[1,3]), 0.0);
    assert_eq!(s::merge_sorted_unique_many(&[&[1,3], &[2,3]]), [1,2,3]);
    assert_eq!(s::shingles(b"abcd", 2).count(), 3);
    assert_eq!(s::rolling_hashes(b"abcd", 2).count(), 3);
    let a=s::minhash_signature([1,2,3], 32, 7).unwrap();
    assert_eq!(s::minhash_jaccard_estimate(&a, &a).unwrap(), 1.0);
    assert_eq!(s::simhash64([1,3]), s::simhash64_weighted([(1,1),(3,1)]));
    assert_eq!(s::hamming_distance64(1,3), 1);
    let metric=|a:&u32,b:&u32| a.abs_diff(*b) as usize;
    let mut tree=s::BkTree::new();tree.insert(10,metric);tree.insert(11,metric);
    assert_eq!(tree.search(&10,1,metric).matches.len(),2);
    assert_eq!(s::myers_levenshtein_bytes(b"kitten", b"sitting").unwrap(),3);
    assert_eq!(search_kernels::top_k_by([3,1,2],usize::MAX,Ord::cmp),[1,2,3]);
    use graph_kernels as g;
    let flow=g::dinic_max_flow(2,&[g::CapacityEdge{from:0,to:1,capacity:7}],0,1).unwrap();
    assert_eq!(flow.value,7);assert_eq!(flow.edge_flows,[7]);
    assert_eq!(g::hopcroft_karp(2,2,&[(0,0),(0,1),(1,0)]).unwrap().size,2);
    assert_eq!(g::hungarian_assignment(&[[1,2],[2,100]]).unwrap().total_cost,4);
    println!("CATALOG_CONSUMER_OK");
}
''')
            # Reuse dependencies, not consumer outputs; each temporary consumer
            # path is a separate Cargo source identity. No network is forced.
            env = dict(os.environ, CARGO_TARGET_DIR=str(ROOT / "target/catalog-consumer"))
            run(["cargo", "check", "--workspace", "--all-targets", "--all-features"], consumer, env=env)
            self.assertIn("CATALOG_CONSUMER_OK", run(["cargo", "run", "--quiet", "-p", "catalog-consumer"], consumer, env=env))
            lock = load_json(consumer / LOCK_FILE_NAME)
            self.assertTrue(all(state == "clean" for _, _, state in provenance_status(consumer, lock)))
            for name in installed:
                for file in ITEMS[name]["files"]:
                    self.assertEqual((ROOT / file["path"]).read_bytes(), (consumer / file["target"]).read_bytes())


class SimilarityProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="similarity-provenance-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "registry"
        self.consumer = Path(self.tmp.name) / "consumer"
        self.root.mkdir()
        for relative in ["registry.json", "registry.schema.json", "provenance.schema.json", "merge-bundle.schema.json",
                         *[file["path"] for file in ITEMS["similarity-kernels"]["files"]]]:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        run(["git", "init", "-q"], self.root)
        run(["git", "config", "user.name", "catalog tests"], self.root)
        run(["git", "config", "user.email", "tests@example.invalid"], self.root)
        self.base = self.commit("base")
        self.registry_path = self.root / "registry.json"
        self.registry = load_json(self.registry_path)
        self.install(self.base)
        self.relative = "crates/similarity-kernels/src/bounded.rs"

    def commit(self, message):
        run(["git", "add", "."], self.root)
        run(["git", "commit", "-qm", message], self.root)
        return run(["git", "rev-parse", "HEAD"], self.root).strip()

    def install(self, revision):
        return install_items(self.root, self.registry_path, self.registry, ["similarity-kernels"], self.consumer, revision)

    def lock(self):
        return load_json(self.consumer / LOCK_FILE_NAME)

    def test_idempotent_install_hashes_and_local_divergence(self):
        before = (self.consumer / LOCK_FILE_NAME).read_bytes()
        self.install(self.base)
        self.assertEqual(before, (self.consumer / LOCK_FILE_NAME).read_bytes())
        for item in self.lock()["items"]:
            self.assertEqual(item["registrySha256"], sha256_file(self.registry_path))
            self.assertEqual(item["revision"], self.base)
        target = self.consumer / self.relative
        target.write_text(target.read_text() + "\n// local specialization\n")
        self.assertIn(("similarity-kernels", self.relative, "modified"), provenance_status(self.consumer, self.lock()))
        with self.assertRaisesRegex(SourceRegistryError, "locally divergent"):
            self.install(self.base)
        self.assertEqual(before, (self.consumer / LOCK_FILE_NAME).read_bytes())

    def test_real_item_update_and_three_way_resolution(self):
        source = self.root / self.relative
        source.write_text(source.read_text() + "\n// upstream revision one\n")
        first = self.commit("upstream one")
        plans = apply_safe_updates(self.root, self.registry_path, self.registry, self.lock(), self.consumer, first)
        self.assertEqual([p.target for p in plans if p.state == "upstream-only"], [self.relative])
        self.assertTrue(all(state == "clean" for _, _, state in provenance_status(self.consumer, self.lock())))
        target = self.consumer / self.relative
        target.write_text(target.read_text() + "\n// local change\n")
        source.write_text(source.read_text() + "\n// upstream revision two\n")
        second = self.commit("upstream two")
        before = target.read_bytes()
        with self.assertRaisesRegex(SourceRegistryError, "manual resolution"):
            apply_safe_updates(self.root, self.registry_path, self.registry, self.lock(), self.consumer, second)
        self.assertEqual(target.read_bytes(), before)
        bundle_dir = Path(self.tmp.name) / "bundle"
        bundle = export_merge_bundle(self.root, self.registry_path, self.registry, self.lock(), self.consumer, bundle_dir, second)
        self.assertEqual(len(bundle["conflicts"]), 1)
        entry = bundle["conflicts"][0]
        self.assertEqual((bundle_dir / entry["artifacts"]["ours"]).read_bytes(), before)
        self.assertEqual((bundle_dir / entry["artifacts"]["theirs"]).read_bytes(), source.read_bytes())
        # Explicitly simulate the consumer accepting the upstream side.
        target.write_bytes(source.read_bytes())
        accept_merge_bundle(self.root, self.registry_path, self.registry, self.lock(), self.consumer,
                            bundle_dir / MERGE_BUNDLE_FILE, ["similarity-kernels"])
        self.assertEqual(self.lock()["items"][0]["revision"], second)
        self.assertTrue(all(state == "clean" for _, _, state in provenance_status(self.consumer, self.lock())))

    def test_layout_changes_still_require_explicit_reconciliation(self):
        registry = json.loads(json.dumps(self.registry))
        item = next(i for i in registry["items"] if i["name"] == "similarity-kernels")
        item["files"][0]["target"] = "renamed/Cargo.toml"
        plans = plan_updates(self.root, registry, self.lock(), self.consumer)
        self.assertTrue(any(plan.state == "layout-changed" for plan in plans))
        before = (self.consumer / LOCK_FILE_NAME).read_bytes()
        with self.assertRaises(SourceRegistryError):
            apply_safe_updates(self.root, self.registry_path, registry, self.lock(), self.consumer, self.base)
        self.assertEqual((self.consumer / LOCK_FILE_NAME).read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
