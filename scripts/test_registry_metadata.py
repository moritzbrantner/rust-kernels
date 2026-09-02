#!/usr/bin/env python3

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "registry.json"


def fail(message: str) -> None:
    raise SystemExit(f"registry metadata test failed: {message}")


try:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    fail(f"cannot read registry.json: {error}")

if registry.get("version") != 3:
    fail("registry must use metadata format version 3")

items = registry.get("items")
if not isinstance(items, list):
    fail("registry items must be an array")

standalone = [
    item
    for item in items
    if isinstance(item, dict)
    and isinstance(item.get("integration"), dict)
    and item["integration"].get("mode") == "standalone-module"
]
if not standalone:
    fail("registry must contain standalone modules")

benchmark_backed = 0
for item in standalone:
    name = item.get("name")
    if not isinstance(name, str) or not name:
        fail("standalone item has no name")

    files = item.get("files")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        fail(f"{name!r} must have exactly one registered source file")
    source_value = files[0].get("path")
    if not isinstance(source_value, str) or not source_value:
        fail(f"{name!r} registered source path is invalid")

    characteristics = item.get("characteristics")
    if not isinstance(characteristics, dict):
        fail(f"{name!r} has no characteristics metadata")
    if not isinstance(characteristics.get("deterministic"), bool):
        fail(f"{name!r} deterministic metadata must be boolean")
    operations = characteristics.get("operations")
    if not isinstance(operations, list) or not operations:
        fail(f"{name!r} must describe at least one operation")

    verification = item.get("verification")
    if not isinstance(verification, dict):
        fail(f"{name!r} has no verification metadata")
    tests = verification.get("tests")
    if not isinstance(tests, list) or source_value not in tests:
        fail(
            f"{name!r} must cite its registered standalone source in verification.tests"
        )

    source = ROOT / source_value
    try:
        source_text = source.read_text(encoding="utf-8")
    except OSError as error:
        fail(f"cannot read verification source for {name!r}: {error}")
    if "#[cfg(test)]" not in source_text:
        fail(f"{name!r} claims inline test verification but has no #[cfg(test)] module")

    benchmarks = verification.get("benchmarks")
    if not isinstance(benchmarks, list):
        fail(f"{name!r} verification.benchmarks must be an array")
    if benchmarks:
        benchmark_backed += 1

print(
    "registry metadata ok: "
    f"{len(standalone)} characterized/test-backed standalone module(s), "
    f"{benchmark_backed} with registered benchmark evidence"
)
