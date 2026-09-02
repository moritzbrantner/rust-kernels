#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "registry.json"
LOCK_FILE_NAME = ".rust-kernels.lock.json"
LOCK_SCHEMA = "https://raw.githubusercontent.com/moritzbrantner/rust-kernels/main/provenance.schema.json"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class SourceRegistryError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourceRegistryError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise SourceRegistryError(f"{path} must contain a JSON object")
    return value


def relative_path(value: object, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise SourceRegistryError(f"{context} must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise SourceRegistryError(
            f"{context} must be a relative path without '..'"
        )
    return path


def safe_join(root: Path, relative: PurePosixPath, context: str) -> Path:
    root = root.resolve()
    candidate = (root / Path(*relative.parts)).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise SourceRegistryError(f"{context} escapes {root}") from error
    return candidate


def item_map(registry: dict) -> dict[str, dict]:
    items = registry.get("items")
    if not isinstance(items, list):
        raise SourceRegistryError("registry items must be an array")
    result: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise SourceRegistryError("every registry item must have a string name")
        name = item["name"]
        if name in result:
            raise SourceRegistryError(f"duplicate registry item {name!r}")
        result[name] = item
    return result


def resolve_items(registry: dict, requested: Iterable[str]) -> list[dict]:
    items = item_map(registry)
    resolved: list[dict] = []
    state: dict[str, str] = {}

    def visit(name: str) -> None:
        item = items.get(name)
        if item is None:
            raise SourceRegistryError(f"unknown registry item {name!r}")
        if state.get(name) == "done":
            return
        if state.get(name) == "visiting":
            raise SourceRegistryError(f"registry dependency cycle at {name!r}")

        state[name] = "visiting"
        dependencies = item.get("registryDependencies", [])
        if not isinstance(dependencies, list) or not all(
            isinstance(dependency, str) and dependency for dependency in dependencies
        ):
            raise SourceRegistryError(
                f"{name!r}.registryDependencies must be a string array"
            )
        for dependency in dependencies:
            visit(dependency)
        state[name] = "done"
        resolved.append(item)

    requested_names = list(requested)
    if not requested_names:
        raise SourceRegistryError("at least one registry item is required")
    for name in requested_names:
        visit(name)
    return resolved


def registry_identity(registry: dict) -> dict[str, str]:
    source = registry.get("source")
    if not isinstance(source, dict):
        raise SourceRegistryError("registry.source must be an object")
    name = registry.get("name")
    repository = source.get("repository")
    if not isinstance(name, str) or not name:
        raise SourceRegistryError("registry.name must be a non-empty string")
    if not isinstance(repository, str) or not repository:
        raise SourceRegistryError(
            "registry.source.repository must be a non-empty string"
        )
    return {"name": name, "repository": repository}


def validate_revision(revision: str) -> None:
    if REVISION_RE.fullmatch(revision) is None:
        raise SourceRegistryError(
            "revision must be a full lowercase 40-character Git commit SHA"
        )


def current_git_revision(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SourceRegistryError("cannot determine the current Git revision") from error
    revision = result.stdout.strip().lower()
    validate_revision(revision)
    return revision


def ensure_clean_sources(root: Path, items: list[dict]) -> None:
    paths = {"registry.json", "registry.schema.json", "provenance.schema.json"}
    for item in items:
        files = item.get("files")
        if not isinstance(files, list):
            raise SourceRegistryError(f"{item.get('name')!r}.files must be an array")
        for index, entry in enumerate(files):
            if not isinstance(entry, dict):
                raise SourceRegistryError(
                    f"{item.get('name')!r}.files[{index}] must be an object"
                )
            source = relative_path(
                entry.get("path"), f"{item.get('name')!r}.files[{index}].path"
            )
            paths.add(source.as_posix())

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", *sorted(paths)],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SourceRegistryError("cannot inspect Git source state") from error
    if result.stdout.strip():
        raise SourceRegistryError(
            "source snapshot is dirty; commit the registry and selected source files first"
        )


def lock_path(destination: Path) -> Path:
    return destination / LOCK_FILE_NAME


def new_lock(registry: dict) -> dict:
    return {
        "$schema": LOCK_SCHEMA,
        "version": 1,
        "registry": registry_identity(registry),
        "items": [],
    }


def load_existing_lock(destination: Path, registry: dict) -> dict:
    path = lock_path(destination)
    if not path.exists():
        return new_lock(registry)

    lock = load_json(path)
    if lock.get("$schema") != LOCK_SCHEMA or lock.get("version") != 1:
        raise SourceRegistryError(f"{path} uses an unsupported provenance format")
    if lock.get("registry") != registry_identity(registry):
        raise SourceRegistryError(
            f"{path} belongs to a different source registry"
        )
    if not isinstance(lock.get("items"), list):
        raise SourceRegistryError(f"{path}.items must be an array")
    return lock


def prepare_item_snapshot(
    registry_root: Path,
    item: dict,
    revision: str,
    registry_sha256: str,
) -> tuple[dict, list[tuple[Path, PurePosixPath, str]]]:
    name = item.get("name")
    files = item.get("files")
    if not isinstance(name, str) or not name:
        raise SourceRegistryError("registry item name must be a non-empty string")
    if not isinstance(files, list) or not files:
        raise SourceRegistryError(f"{name!r}.files must be a non-empty array")

    lock_files: list[dict[str, str]] = []
    copies: list[tuple[Path, PurePosixPath, str]] = []
    seen_targets: set[PurePosixPath] = set()

    for index, entry in enumerate(files):
        context = f"{name!r}.files[{index}]"
        if not isinstance(entry, dict):
            raise SourceRegistryError(f"{context} must be an object")
        source_relative = relative_path(entry.get("path"), f"{context}.path")
        target_relative = relative_path(entry.get("target"), f"{context}.target")
        if target_relative in seen_targets:
            raise SourceRegistryError(
                f"{name!r} maps target {target_relative} more than once"
            )
        seen_targets.add(target_relative)

        source = safe_join(registry_root, source_relative, f"{context}.path")
        if not source.is_file():
            raise SourceRegistryError(f"source file does not exist: {source_relative}")
        source_sha256 = sha256_file(source)
        lock_files.append(
            {
                "source": source_relative.as_posix(),
                "target": target_relative.as_posix(),
                "sourceSha256": source_sha256,
            }
        )
        copies.append((source, target_relative, source_sha256))

    return (
        {
            "name": name,
            "revision": revision,
            "registrySha256": registry_sha256,
            "files": lock_files,
        },
        copies,
    )


def write_lock(path: Path, lock: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(lock, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def install_items(
    registry_root: Path,
    registry_path: Path,
    registry: dict,
    requested: Iterable[str],
    destination: Path,
    revision: str,
) -> list[str]:
    validate_revision(revision)
    selected = resolve_items(registry, requested)
    registry_sha256 = sha256_file(registry_path)
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    lock = load_existing_lock(destination, registry)

    snapshots: list[tuple[dict, list[tuple[Path, PurePosixPath, str]]]] = []
    all_targets: dict[PurePosixPath, str] = {}

    for item in selected:
        snapshot, copies = prepare_item_snapshot(
            registry_root, item, revision, registry_sha256
        )
        for _source, target_relative, source_sha256 in copies:
            previous = all_targets.get(target_relative)
            if previous is not None and previous != source_sha256:
                raise SourceRegistryError(
                    f"selected items map different content to {target_relative}"
                )
            all_targets[target_relative] = source_sha256
        snapshots.append((snapshot, copies))

    # Preflight every target before writing anything. Existing identical files are
    # accepted; existing divergent files are treated as locally owned and never
    # overwritten by this install operation.
    for _snapshot, copies in snapshots:
        for _source, target_relative, source_sha256 in copies:
            target = safe_join(destination, target_relative, "target path")
            if target.exists():
                if not target.is_file():
                    raise SourceRegistryError(f"target is not a file: {target_relative}")
                if sha256_file(target) != source_sha256:
                    raise SourceRegistryError(
                        f"refusing to overwrite locally divergent file {target_relative}"
                    )

    for _snapshot, copies in snapshots:
        for source, target_relative, _source_sha256 in copies:
            target = safe_join(destination, target_relative, "target path")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)

    existing = {
        entry.get("name"): entry
        for entry in lock["items"]
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    installed_names: list[str] = []
    for snapshot, _copies in snapshots:
        existing[snapshot["name"]] = snapshot
        installed_names.append(snapshot["name"])
    lock["items"] = [existing[name] for name in sorted(existing)]
    write_lock(lock_path(destination), lock)
    return installed_names


def provenance_status(destination: Path, lock: dict) -> list[tuple[str, str, str]]:
    results: list[tuple[str, str, str]] = []
    destination = destination.resolve()
    items = lock.get("items")
    if not isinstance(items, list):
        raise SourceRegistryError("provenance lock items must be an array")

    for item in items:
        if not isinstance(item, dict):
            raise SourceRegistryError("provenance lock item must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise SourceRegistryError("provenance lock item name must be a string")
        files = item.get("files")
        if not isinstance(files, list):
            raise SourceRegistryError(f"{name!r}.files must be an array")

        for index, entry in enumerate(files):
            if not isinstance(entry, dict):
                raise SourceRegistryError(f"{name!r}.files[{index}] must be an object")
            target_relative = relative_path(
                entry.get("target"), f"{name!r}.files[{index}].target"
            )
            expected = entry.get("sourceSha256")
            if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
                raise SourceRegistryError(
                    f"{name!r}.files[{index}].sourceSha256 is invalid"
                )
            target = safe_join(destination, target_relative, "target path")
            if not target.is_file():
                state = "missing"
            elif sha256_file(target) == expected:
                state = "clean"
            else:
                state = "modified"
            results.append((name, target_relative.as_posix(), state))
    return results


def command_install(args: argparse.Namespace) -> int:
    registry = load_json(REGISTRY_PATH)
    selected = resolve_items(registry, args.items)
    revision = args.revision.lower() if args.revision else current_git_revision(ROOT)
    validate_revision(revision)
    current_revision = current_git_revision(ROOT)
    if revision != current_revision:
        raise SourceRegistryError(
            f"requested revision {revision} does not match checkout {current_revision}"
        )
    ensure_clean_sources(ROOT, selected)

    installed = install_items(
        ROOT,
        REGISTRY_PATH,
        registry,
        args.items,
        Path(args.root),
        revision,
    )
    print(f"installed {', '.join(installed)} at {revision}")
    print(f"wrote {Path(args.root) / LOCK_FILE_NAME}")
    return 0


def command_status(args: argparse.Namespace) -> int:
    destination = Path(args.root)
    path = lock_path(destination)
    if not path.is_file():
        raise SourceRegistryError(f"no provenance lock found at {path}")
    lock = load_json(path)
    results = provenance_status(destination, lock)
    for item, target, state in results:
        print(f"{state:8} {item}: {target}")
    dirty = any(state != "clean" for _item, _target, state in results)
    if args.require_clean and dirty:
        return 1
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Reference helper for rust-kernels source-level integration."
    )
    subcommands = value.add_subparsers(dest="command", required=True)

    install = subcommands.add_parser(
        "install",
        help="copy registry items from this checkout and record immutable provenance",
    )
    install.add_argument("items", nargs="+", help="registry item names")
    install.add_argument("--root", required=True, help="consumer repository root")
    install.add_argument(
        "--revision",
        help="full Git commit SHA; defaults to this checkout's HEAD",
    )
    install.set_defaults(handler=command_install)

    status = subcommands.add_parser(
        "status",
        help="classify copied files as clean, modified, or missing",
    )
    status.add_argument("--root", required=True, help="consumer repository root")
    status.add_argument(
        "--require-clean",
        action="store_true",
        help="exit non-zero when copied files are modified or missing",
    )
    status.set_defaults(handler=command_status)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        return args.handler(args)
    except SourceRegistryError as error:
        print(f"source registry error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
