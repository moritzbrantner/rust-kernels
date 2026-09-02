#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from source_registry import (
    LOCK_FILE_NAME,
    LOCK_SCHEMA_RE,
    REGISTRY_PATH,
    ROOT,
    SourceRegistryError,
    current_git_revision,
    ensure_clean_sources,
    item_map,
    load_json,
    lock_path,
    prepare_item_snapshot,
    registry_identity,
    relative_path,
    safe_join,
    sha256_bytes,
    sha256_file,
    validate_revision,
    write_lock,
)

SAFE_UPDATE_STATES = {
    "unchanged",
    "upstream-only",
    "local-only",
    "converged",
}
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class UpdatePlan:
    item: str
    target: str
    state: str


def git_file_at_revision(
    repository_root: Path, revision: str, path: PurePosixPath
) -> bytes:
    validate_revision(revision)
    try:
        result = subprocess.run(
            ["git", "show", f"{revision}:{path.as_posix()}"],
            cwd=repository_root,
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise SourceRegistryError("cannot invoke git to reconstruct provenance") from error

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise SourceRegistryError(
            f"cannot read {path} at recorded revision {revision}{suffix}"
        )
    return result.stdout


def registry_at_revision(
    repository_root: Path,
    revision: str,
    cache: dict[str, tuple[dict, bytes]],
) -> tuple[dict, bytes]:
    cached = cache.get(revision)
    if cached is not None:
        return cached

    raw = git_file_at_revision(repository_root, revision, PurePosixPath("registry.json"))
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceRegistryError(
            f"registry.json at recorded revision {revision} is not valid UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        raise SourceRegistryError(
            f"registry.json at recorded revision {revision} must contain an object"
        )
    cache[revision] = (value, raw)
    return value, raw


def validate_required_lock(destination: Path, registry: dict) -> dict:
    path = lock_path(destination)
    if not path.is_file():
        raise SourceRegistryError(f"no provenance lock found at {path}")

    lock = load_json(path)
    schema = lock.get("$schema")
    if (
        not isinstance(schema, str)
        or LOCK_SCHEMA_RE.fullmatch(schema) is None
        or lock.get("version") != 1
    ):
        raise SourceRegistryError(f"{path} uses an unsupported provenance format")
    if lock.get("registry") != registry_identity(registry):
        raise SourceRegistryError(f"{path} belongs to a different source registry")
    if not isinstance(lock.get("items"), list):
        raise SourceRegistryError(f"{path}.items must be an array")
    return lock


def locked_item_map(lock: dict) -> dict[str, dict]:
    items = lock.get("items")
    if not isinstance(items, list):
        raise SourceRegistryError("provenance lock items must be an array")

    result: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            raise SourceRegistryError("provenance lock item must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise SourceRegistryError("provenance lock item name must be a string")
        if name in result:
            raise SourceRegistryError(f"duplicate locked registry item {name!r}")
        result[name] = item
    return result


def selected_locked_items(lock: dict, requested: Iterable[str]) -> list[dict]:
    items = locked_item_map(lock)
    names = list(requested)
    if not names:
        return [items[name] for name in sorted(items)]

    if len(names) != len(set(names)):
        raise SourceRegistryError("update item list contains duplicates")

    missing = [name for name in names if name not in items]
    if missing:
        raise SourceRegistryError(f"items are not present in the provenance lock: {missing}")
    return [items[name] for name in names]


def registry_file_mapping(item: dict, context: str) -> list[tuple[PurePosixPath, PurePosixPath]]:
    files = item.get("files")
    if not isinstance(files, list) or not files:
        raise SourceRegistryError(f"{context}.files must be a non-empty array")

    result: list[tuple[PurePosixPath, PurePosixPath]] = []
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise SourceRegistryError(f"{context}.files[{index}] must be an object")
        source = relative_path(entry.get("path"), f"{context}.files[{index}].path")
        target = relative_path(entry.get("target"), f"{context}.files[{index}].target")
        result.append((source, target))
    return result


def lock_file_mapping(item: dict, context: str) -> list[tuple[PurePosixPath, PurePosixPath]]:
    files = item.get("files")
    if not isinstance(files, list) or not files:
        raise SourceRegistryError(f"{context}.files must be a non-empty array")

    result: list[tuple[PurePosixPath, PurePosixPath]] = []
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise SourceRegistryError(f"{context}.files[{index}] must be an object")
        source = relative_path(entry.get("source"), f"{context}.files[{index}].source")
        target = relative_path(entry.get("target"), f"{context}.files[{index}].target")
        result.append((source, target))
    return result


def registry_dependencies(item: dict, context: str) -> tuple[str, ...]:
    dependencies = item.get("registryDependencies", [])
    if not isinstance(dependencies, list) or not all(
        isinstance(dependency, str) and dependency for dependency in dependencies
    ):
        raise SourceRegistryError(f"{context}.registryDependencies must be a string array")
    return tuple(dependencies)


def classify_bytes(base: bytes, ours: bytes | None, theirs: bytes) -> str:
    if ours is None:
        return "missing"
    if ours == theirs:
        return "unchanged" if ours == base else "converged"
    if ours == base:
        return "upstream-only"
    if theirs == base:
        return "local-only"
    return "both-changed"


def plan_item_update(
    repository_root: Path,
    candidate_registry: dict,
    locked_item: dict,
    destination: Path,
    registry_cache: dict[str, tuple[dict, bytes]],
) -> list[UpdatePlan]:
    name = locked_item.get("name")
    revision = locked_item.get("revision")
    registry_sha256 = locked_item.get("registrySha256")
    if not isinstance(name, str) or not name:
        raise SourceRegistryError("locked item name must be a non-empty string")
    if not isinstance(revision, str):
        raise SourceRegistryError(f"locked item {name!r} has no revision")
    validate_revision(revision)
    if not isinstance(registry_sha256, str) or HASH_RE.fullmatch(registry_sha256) is None:
        raise SourceRegistryError(f"locked item {name!r} has an invalid registrySha256")

    historical_registry, historical_registry_bytes = registry_at_revision(
        repository_root, revision, registry_cache
    )
    if sha256_bytes(historical_registry_bytes) != registry_sha256:
        return [UpdatePlan(name, "*", "registry-mismatch")]

    historical_items = item_map(historical_registry)
    historical_item = historical_items.get(name)
    if historical_item is None:
        return [UpdatePlan(name, "*", "registry-mismatch")]

    locked_mapping = lock_file_mapping(locked_item, f"lock item {name!r}")
    historical_mapping = registry_file_mapping(
        historical_item, f"historical registry item {name!r}"
    )
    if locked_mapping != historical_mapping:
        return [UpdatePlan(name, "*", "lock-mismatch")]

    candidate_item = item_map(candidate_registry).get(name)
    if candidate_item is None:
        return [UpdatePlan(name, "*", "item-removed")]

    candidate_mapping = registry_file_mapping(candidate_item, f"registry item {name!r}")
    if candidate_mapping != locked_mapping:
        return [UpdatePlan(name, "*", "layout-changed")]

    if registry_dependencies(candidate_item, f"registry item {name!r}") != registry_dependencies(
        historical_item, f"historical registry item {name!r}"
    ):
        return [UpdatePlan(name, "*", "dependencies-changed")]

    locked_files = locked_item.get("files")
    if not isinstance(locked_files, list):
        raise SourceRegistryError(f"locked item {name!r}.files must be an array")

    plans: list[UpdatePlan] = []
    for index, ((source_relative, target_relative), locked_file) in enumerate(
        zip(locked_mapping, locked_files, strict=True)
    ):
        if not isinstance(locked_file, dict):
            raise SourceRegistryError(f"locked item {name!r}.files[{index}] must be an object")
        expected_source_sha256 = locked_file.get("sourceSha256")
        if (
            not isinstance(expected_source_sha256, str)
            or HASH_RE.fullmatch(expected_source_sha256) is None
        ):
            raise SourceRegistryError(
                f"locked item {name!r}.files[{index}] has invalid sourceSha256"
            )

        base = git_file_at_revision(repository_root, revision, source_relative)
        if sha256_bytes(base) != expected_source_sha256:
            plans.append(UpdatePlan(name, target_relative.as_posix(), "base-mismatch"))
            continue

        candidate_source = safe_join(
            repository_root, source_relative, f"registry item {name!r} source"
        )
        if not candidate_source.is_file():
            raise SourceRegistryError(f"candidate source file is missing: {source_relative}")
        theirs = candidate_source.read_bytes()

        target = safe_join(destination, target_relative, f"locked item {name!r} target")
        ours = target.read_bytes() if target.is_file() else None
        plans.append(
            UpdatePlan(
                name,
                target_relative.as_posix(),
                classify_bytes(base, ours, theirs),
            )
        )

    return plans


def plan_updates(
    repository_root: Path,
    candidate_registry: dict,
    lock: dict,
    destination: Path,
    requested: Iterable[str] = (),
) -> list[UpdatePlan]:
    destination = destination.resolve()
    selected = selected_locked_items(lock, requested)
    cache: dict[str, tuple[dict, bytes]] = {}
    plans: list[UpdatePlan] = []
    for locked_item in selected:
        plans.extend(
            plan_item_update(
                repository_root,
                candidate_registry,
                locked_item,
                destination,
                cache,
            )
        )
    return sorted(plans, key=lambda plan: (plan.item, plan.target, plan.state))


def unsafe_plans(plans: Iterable[UpdatePlan]) -> list[UpdatePlan]:
    return [plan for plan in plans if plan.state not in SAFE_UPDATE_STATES]


def candidate_items_for_clean_check(
    candidate_registry: dict, locked_items: Iterable[dict]
) -> list[dict]:
    candidates = item_map(candidate_registry)
    result: list[dict] = []
    for locked_item in locked_items:
        name = locked_item.get("name")
        if isinstance(name, str) and name in candidates:
            result.append(candidates[name])
    return result


def apply_safe_updates(
    repository_root: Path,
    registry_path: Path,
    candidate_registry: dict,
    lock: dict,
    destination: Path,
    candidate_revision: str,
    requested: Iterable[str] = (),
) -> list[UpdatePlan]:
    validate_revision(candidate_revision)
    destination = destination.resolve()
    selected = selected_locked_items(lock, requested)
    plans = plan_updates(
        repository_root,
        candidate_registry,
        lock,
        destination,
        [item["name"] for item in selected],
    )
    manual = unsafe_plans(plans)
    if manual:
        states = ", ".join(
            f"{plan.item}:{plan.target}={plan.state}" for plan in manual
        )
        raise SourceRegistryError(
            f"update requires manual resolution; no files were changed: {states}"
        )

    candidates = item_map(candidate_registry)
    registry_sha256 = sha256_file(registry_path)
    plan_by_target = {(plan.item, plan.target): plan for plan in plans}
    snapshots: dict[str, dict] = {}

    for locked_item in selected:
        name = locked_item["name"]
        candidate_item = candidates[name]
        snapshot, copies = prepare_item_snapshot(
            repository_root,
            candidate_item,
            candidate_revision,
            registry_sha256,
        )
        snapshots[name] = snapshot

        for source, target_relative, _source_sha256 in copies:
            plan = plan_by_target[(name, target_relative.as_posix())]
            if plan.state != "upstream-only":
                continue
            target = safe_join(destination, target_relative, f"update target for {name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    existing = locked_item_map(lock)
    existing.update(snapshots)
    lock["items"] = [existing[name] for name in sorted(existing)]
    write_lock(lock_path(destination), lock)
    return plans


def print_plans(plans: Iterable[UpdatePlan]) -> None:
    for plan in plans:
        print(f"{plan.state:20} {plan.item}: {plan.target}")


def command_plan(args: argparse.Namespace) -> int:
    candidate_registry = load_json(REGISTRY_PATH)
    destination = Path(args.root)
    lock = validate_required_lock(destination, candidate_registry)
    selected = selected_locked_items(lock, args.items)
    ensure_clean_sources(
        ROOT,
        candidate_items_for_clean_check(candidate_registry, selected),
    )
    candidate_revision = current_git_revision(ROOT)
    plans = plan_updates(
        ROOT,
        candidate_registry,
        lock,
        destination,
        [item["name"] for item in selected],
    )
    print(f"candidate {candidate_revision}")
    print_plans(plans)
    if args.require_safe and unsafe_plans(plans):
        return 1
    return 0


def command_apply(args: argparse.Namespace) -> int:
    candidate_registry = load_json(REGISTRY_PATH)
    destination = Path(args.root)
    lock = validate_required_lock(destination, candidate_registry)
    selected = selected_locked_items(lock, args.items)
    ensure_clean_sources(
        ROOT,
        candidate_items_for_clean_check(candidate_registry, selected),
    )
    candidate_revision = current_git_revision(ROOT)
    plans = apply_safe_updates(
        ROOT,
        REGISTRY_PATH,
        candidate_registry,
        lock,
        destination,
        candidate_revision,
        [item["name"] for item in selected],
    )
    print_plans(plans)
    print(
        f"advanced {', '.join(item['name'] for item in selected)} to {candidate_revision}"
    )
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Reference update planner for rust-kernels source copies."
    )
    subcommands = value.add_subparsers(dest="command", required=True)

    plan = subcommands.add_parser(
        "plan",
        help="classify base/consumer/candidate changes without modifying the consumer",
    )
    plan.add_argument("items", nargs="*", help="locked item names; defaults to all")
    plan.add_argument("--root", required=True, help="consumer repository root")
    plan.add_argument(
        "--require-safe",
        action="store_true",
        help="exit non-zero when any selected item needs manual resolution",
    )
    plan.set_defaults(handler=command_plan)

    apply = subcommands.add_parser(
        "apply",
        help="apply only a fully safe update plan and advance item provenance",
    )
    apply.add_argument("items", nargs="*", help="locked item names; defaults to all")
    apply.add_argument("--root", required=True, help="consumer repository root")
    apply.set_defaults(handler=command_apply)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        return args.handler(args)
    except SourceRegistryError as error:
        print(f"source update error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
