#!/usr/bin/env python3

import json
import tomllib
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "verification-matrix.json"
WORKSPACE_PATH = ROOT / "Cargo.toml"

EXPECTED_STATUS_VALUES = {"present", "partial", "missing", "planned", "n/a"}
EVIDENCE_FIELDS = (
    "oracleDifferential",
    "propertyMetamorphic",
    "coverage",
    "mutation",
    "deterministicPerformance",
    "criterion",
    "runtimeProfiler",
)


def fail(message: str) -> None:
    raise SystemExit(f"verification matrix validation failed: {message}")


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


def require_strings(value: object, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        fail(f"{context} must be a non-empty array")
    if not all(isinstance(item, str) and item for item in value):
        fail(f"{context} must contain only non-empty strings")
    if len(value) != len(set(value)):
        fail(f"{context} contains duplicates")
    return value


def require_source_path(value: object, context: str) -> str:
    raw = require_string(value, context)
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        fail(f"{context} must be a repository-relative path without '..'")
    if path.suffix != ".rs" or "src" not in path.parts:
        fail(f"{context} must point to a Rust source file under src/")
    if not (ROOT / path).is_file():
        fail(f"{context} references missing file {raw}")
    return raw


try:
    with WORKSPACE_PATH.open("rb") as handle:
        cargo = tomllib.load(handle)
except (OSError, tomllib.TOMLDecodeError) as error:
    fail(f"cannot read Cargo.toml: {error}")

workspace = cargo.get("workspace")
if not isinstance(workspace, dict):
    fail("Cargo.toml must define [workspace]")
members = workspace.get("members")
if not isinstance(members, list) or not members:
    fail("workspace.members must be a non-empty array")
if not all(isinstance(member, str) and member for member in members):
    fail("workspace.members must contain only non-empty strings")

member_by_crate: dict[str, str] = {}
workspace_sources: set[str] = set()
for member in members:
    member_path = ROOT / member
    manifest_path = member_path / "Cargo.toml"
    try:
        with manifest_path.open("rb") as handle:
            manifest = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        fail(f"cannot read {manifest_path.relative_to(ROOT)}: {error}")

    package = manifest.get("package")
    if not isinstance(package, dict):
        fail(f"{manifest_path.relative_to(ROOT)} must define [package]")
    crate = require_string(
        package.get("name"), f"{manifest_path.relative_to(ROOT)} package.name"
    )
    if crate in member_by_crate:
        fail(f"duplicate workspace package name {crate!r}")
    member_by_crate[crate] = member

    src_dir = member_path / "src"
    if not src_dir.is_dir():
        fail(f"workspace member {member!r} has no src directory")
    for source in src_dir.rglob("*.rs"):
        workspace_sources.add(source.relative_to(ROOT).as_posix())

matrix = load_json(MATRIX_PATH)
if matrix.get("schemaVersion") != 1:
    fail("schemaVersion must be 1")
if matrix.get("issue") != 52:
    fail("issue must be 52")

status_values = matrix.get("statusValues")
if not isinstance(status_values, list) or set(status_values) != EXPECTED_STATUS_VALUES:
    fail(f"statusValues must contain exactly {sorted(EXPECTED_STATUS_VALUES)}")

facades = require_strings(matrix.get("facades"), "facades")
facade_sources: set[str] = set()
for index, value in enumerate(facades):
    source = require_source_path(value, f"facades[{index}]")
    if not source.endswith("/src/lib.rs"):
        fail(f"facades[{index}] must be a crate src/lib.rs")
    facade_sources.add(source)

kernels = matrix.get("kernels")
if not isinstance(kernels, list) or not kernels:
    fail("kernels must be a non-empty array")

kernel_ids: set[str] = set()
kernel_sources: set[str] = set()
crates_with_kernels: set[str] = set()

for index, kernel in enumerate(kernels):
    context = f"kernels[{index}]"
    if not isinstance(kernel, dict):
        fail(f"{context} must be an object")

    kernel_id = require_string(kernel.get("id"), f"{context}.id")
    if kernel_id in kernel_ids:
        fail(f"duplicate kernel id {kernel_id!r}")
    kernel_ids.add(kernel_id)

    crate = require_string(kernel.get("crate"), f"{context}.crate")
    member = member_by_crate.get(crate)
    if member is None:
        fail(f"{context}.crate {crate!r} is not a workspace package")
    crates_with_kernels.add(crate)

    require_string(kernel.get("module"), f"{context}.module")
    sources = require_strings(kernel.get("sources"), f"{context}.sources")
    expected_prefix = f"{member}/src/"
    for source_index, value in enumerate(sources):
        source = require_source_path(value, f"{context}.sources[{source_index}]")
        if not source.startswith(expected_prefix):
            fail(
                f"{context}.sources[{source_index}] must belong to workspace package {crate!r}"
            )
        if source in facade_sources:
            fail(f"{context} uses facade source {source}; remove it from facades first")
        kernel_sources.add(source)

    notes = kernel.get("notes", [])
    if not isinstance(notes, list) or not all(
        isinstance(note, str) and note for note in notes
    ):
        fail(f"{context}.notes must be an array of non-empty strings")
    joined_notes = "\n".join(notes)

    for field in EVIDENCE_FIELDS:
        state = require_string(kernel.get(field), f"{context}.{field}")
        if state not in EXPECTED_STATUS_VALUES:
            fail(f"{context}.{field} has unsupported state {state!r}")
        if state in {"n/a", "planned"} and f"{field}:" not in joined_notes:
            fail(
                f"{context}.{field}={state!r} requires a note prefixed with '{field}:'"
            )

    if kernel["coverage"] != "present":
        fail(
            f"{context}.coverage must be 'present': the workspace cargo llvm-cov baseline "
            "covers every workspace source in this slice"
        )

unassigned = sorted(workspace_sources - kernel_sources - facade_sources)
if unassigned:
    fail("workspace Rust sources missing from matrix/facades: " + ", ".join(unassigned))

unknown = sorted((kernel_sources | facade_sources) - workspace_sources)
if unknown:
    fail("matrix references Rust sources outside the workspace inventory: " + ", ".join(unknown))

missing_crates = sorted(set(member_by_crate) - crates_with_kernels)
if missing_crates:
    fail("workspace packages without a kernel row: " + ", ".join(missing_crates))

print(
    f"verification matrix valid: {len(kernels)} kernels, "
    f"{len(kernel_sources)} implementation sources, {len(facade_sources)} facades"
)
