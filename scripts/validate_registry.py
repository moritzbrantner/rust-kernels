#!/usr/bin/env python3

import json
import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "registry.json"
SCHEMA_PATH = ROOT / "registry.schema.json"
PROVENANCE_SCHEMA_PATH = ROOT / "provenance.schema.json"
PROVENANCE_SCHEMA_URI = "https://raw.githubusercontent.com/moritzbrantner/rust-kernels/main/provenance.schema.json"
ALLOWED_ITEM_TYPES = {
    "registry:crate",
    "registry:algorithm",
    "registry:data-structure",
    "registry:file",
}
ALLOWED_PROVIDED_TYPES = {"algorithm", "data-structure", "interface"}
ALLOWED_MUTATIONS = {"none", "input", "internal-state"}
ALLOWED_ALLOCATIONS = {"none", "fixed", "input-sized", "capacity-bounded", "may-grow"}
MODULE_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def fail(message: str) -> None:
    raise SystemExit(f"registry validation failed: {message}")


def load_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path.relative_to(ROOT)}: {error}")

    if not isinstance(value, dict):
        fail(f"{path.relative_to(ROOT)} must contain a JSON object")
    return value


def require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        fail(f"{context} must be a non-empty string")
    return value


def require_string_array(value: object, context: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(entry, str) and entry for entry in value
    ):
        fail(f"{context} must be an array of non-empty strings")
    if len(value) != len(set(value)):
        fail(f"{context} contains duplicates")
    return value


def require_relative_path(value: object, context: str) -> PurePosixPath:
    path = PurePosixPath(require_string(value, context))
    if path.is_absolute() or ".." in path.parts:
        fail(f"{context} must be a repository-relative path without '..'")
    return path


def validate_characteristics(value: object, context: str) -> None:
    if not isinstance(value, dict):
        fail(f"{context} must be an object")
    deterministic = value.get("deterministic")
    if not isinstance(deterministic, bool):
        fail(f"{context}.deterministic must be a boolean")

    operations = value.get("operations")
    if not isinstance(operations, list) or not operations:
        fail(f"{context}.operations must be a non-empty array")

    operation_names: set[str] = set()
    for index, operation in enumerate(operations):
        operation_context = f"{context}.operations[{index}]"
        if not isinstance(operation, dict):
            fail(f"{operation_context} must be an object")
        name = require_string(operation.get("operation"), f"{operation_context}.operation")
        if name in operation_names:
            fail(f"{context} describes operation {name!r} more than once")
        operation_names.add(name)
        require_string(operation.get("time"), f"{operation_context}.time")
        require_string(operation.get("extraSpace"), f"{operation_context}.extraSpace")
        mutation = operation.get("mutation")
        if mutation not in ALLOWED_MUTATIONS:
            fail(f"{operation_context}.mutation has unsupported value {mutation!r}")
        allocation = operation.get("allocation")
        if allocation not in ALLOWED_ALLOCATIONS:
            fail(f"{operation_context}.allocation has unsupported value {allocation!r}")


def validate_verification(value: object, context: str) -> None:
    if not isinstance(value, dict):
        fail(f"{context} must be an object")
    tests = require_string_array(value.get("tests"), f"{context}.tests")
    if not tests:
        fail(f"{context}.tests must list at least one verification source")
    benchmarks = require_string_array(value.get("benchmarks"), f"{context}.benchmarks")

    for kind, paths in (("tests", tests), ("benchmarks", benchmarks)):
        for index, path_value in enumerate(paths):
            path = require_relative_path(path_value, f"{context}.{kind}[{index}]")
            if not (ROOT / path).is_file():
                fail(f"{context}.{kind}[{index}] references missing file {path}")


registry = load_json(REGISTRY_PATH)
load_json(SCHEMA_PATH)
load_json(PROVENANCE_SCHEMA_PATH)

if registry.get("$schema") != "./registry.schema.json":
    fail("$schema must point to ./registry.schema.json")
if registry.get("name") != "rust-kernels":
    fail("name must be rust-kernels")
if registry.get("version") != 3:
    fail("version must be 3")

source = registry.get("source")
if not isinstance(source, dict):
    fail("source must be an object")
require_string(source.get("repository"), "source.repository")
if source.get("revisionType") != "git-commit":
    fail("source.revisionType must be git-commit")

provenance = registry.get("provenance")
if not isinstance(provenance, dict):
    fail("provenance must be an object")
if provenance.get("schema") != PROVENANCE_SCHEMA_URI:
    fail("provenance.schema must point to the published provenance schema")
if provenance.get("lockFile") != ".rust-kernels.lock.json":
    fail("provenance.lockFile must be .rust-kernels.lock.json")

items = registry.get("items")
if not isinstance(items, list) or not items:
    fail("items must be a non-empty array")

item_names: set[str] = set()
registry_dependencies: dict[str, set[str]] = {}
standalone_count = 0

for index, item in enumerate(items):
    context = f"items[{index}]"
    if not isinstance(item, dict):
        fail(f"{context} must be an object")

    name = require_string(item.get("name"), f"{context}.name")
    if name in item_names:
        fail(f"duplicate item name {name!r}")
    item_names.add(name)

    item_type = require_string(item.get("type"), f"{context}.type")
    if item_type not in ALLOWED_ITEM_TYPES:
        fail(f"{context}.type has unsupported value {item_type!r}")

    require_string(item.get("title"), f"{context}.title")
    require_string(item.get("description"), f"{context}.description")

    files = item.get("files")
    if not isinstance(files, list) or not files:
        fail(f"{context}.files must be a non-empty array")

    source_paths: set[PurePosixPath] = set()
    target_paths: set[PurePosixPath] = set()
    for file_index, file_entry in enumerate(files):
        file_context = f"{context}.files[{file_index}]"
        if not isinstance(file_entry, dict):
            fail(f"{file_context} must be an object")
        source_path = require_relative_path(
            file_entry.get("path"), f"{file_context}.path"
        )
        target_path = require_relative_path(
            file_entry.get("target"), f"{file_context}.target"
        )
        if source_path in source_paths:
            fail(f"{context} lists source file {source_path} more than once")
        if target_path in target_paths:
            fail(f"{context} maps target file {target_path} more than once")
        source_paths.add(source_path)
        target_paths.add(target_path)
        if not (ROOT / source_path).is_file():
            fail(f"{context} references missing source file {source_path}")

    dependencies = require_string_array(item.get("dependencies", []), f"{context}.dependencies")
    registry_dependency_values = require_string_array(
        item.get("registryDependencies", []), f"{context}.registryDependencies"
    )
    registry_dependencies[name] = set(registry_dependency_values)

    crate = item.get("crate")
    integration = item.get("integration")
    characteristics = item.get("characteristics")
    verification = item.get("verification")

    if item_type == "registry:crate":
        if not isinstance(crate, dict):
            fail(f"{context}.crate is required for registry:crate items")
        if integration is not None:
            fail(f"{context}.integration is reserved for non-crate source items")
        if characteristics is not None:
            fail(f"{context}.characteristics is reserved for granular source items")
        require_string(crate.get("package"), f"{context}.crate.package")
        manifest = require_relative_path(
            crate.get("manifest"), f"{context}.crate.manifest"
        )
        if manifest not in source_paths:
            fail(f"{context}.crate.manifest must also be listed in files")
        if verification is not None:
            validate_verification(verification, f"{context}.verification")
    else:
        if crate is not None:
            fail(f"{context}.crate is not allowed for non-crate items")
        if not isinstance(integration, dict):
            fail(f"{context}.integration is required for non-crate items")
        if integration.get("mode") != "standalone-module":
            fail(f"{context}.integration.mode must be standalone-module")
        module = require_string(integration.get("module"), f"{context}.integration.module")
        if MODULE_RE.fullmatch(module) is None:
            fail(f"{context}.integration.module is not a valid Rust module identifier")
        if len(files) != 1:
            fail(f"{context} standalone modules must contain exactly one source file")
        only_source = next(iter(source_paths))
        only_target = next(iter(target_paths))
        if only_source.suffix != ".rs":
            fail(f"{context} standalone module source must be a .rs file")
        expected_target = PurePosixPath("src") / "kernels" / f"{module}.rs"
        if only_target != expected_target:
            fail(
                f"{context} standalone module target must be {expected_target}, "
                f"got {only_target}"
            )
        if dependencies:
            fail(f"{context} standalone modules cannot declare external dependencies")
        if registry_dependency_values:
            fail(f"{context} standalone modules cannot declare registry dependencies")
        validate_characteristics(characteristics, f"{context}.characteristics")
        validate_verification(verification, f"{context}.verification")
        standalone_count += 1

    provided = item.get("provides", [])
    if not isinstance(provided, list):
        fail(f"{context}.provides must be an array")
    if item_type != "registry:crate" and not provided:
        fail(f"{context}.provides must describe the standalone kernel")
    provided_names: set[str] = set()
    for provided_index, entry in enumerate(provided):
        provided_context = f"{context}.provides[{provided_index}]"
        if not isinstance(entry, dict):
            fail(f"{provided_context} must be an object")
        provided_name = require_string(entry.get("name"), f"{provided_context}.name")
        if provided_name in provided_names:
            fail(f"{context} provides {provided_name!r} more than once")
        provided_names.add(provided_name)
        provided_type = require_string(entry.get("type"), f"{provided_context}.type")
        if provided_type not in ALLOWED_PROVIDED_TYPES:
            fail(f"{provided_context}.type has unsupported value {provided_type!r}")
        symbols = entry.get("symbols")
        if not isinstance(symbols, list) or not symbols or not all(
            isinstance(symbol, str) and symbol for symbol in symbols
        ):
            fail(f"{provided_context}.symbols must be a non-empty string array")
        if len(symbols) != len(set(symbols)):
            fail(f"{provided_context}.symbols contains duplicates")

for item_name, dependencies in registry_dependencies.items():
    missing = dependencies - item_names
    if missing:
        fail(f"{item_name!r} has unknown registry dependencies: {sorted(missing)}")
    if item_name in dependencies:
        fail(f"{item_name!r} cannot depend on itself")

print(f"registry ok: {len(items)} item(s), {standalone_count} characterized standalone module(s)")
