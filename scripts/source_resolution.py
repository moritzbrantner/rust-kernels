#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable

from source_registry import (
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
    safe_join,
    sha256_bytes,
    sha256_file,
    validate_revision,
    write_lock,
)
from source_update import (
    SAFE_UPDATE_STATES,
    candidate_items_for_clean_check,
    git_file_at_revision,
    lock_file_mapping,
    locked_item_map,
    plan_updates,
    registry_file_mapping,
    selected_locked_items,
    validate_required_lock,
)

MERGE_BUNDLE_FILE = "merge-bundle.json"
MERGE_BUNDLE_SCHEMA_RE = re.compile(
    r"^https://raw\.githubusercontent\.com/moritzbrantner/rust-kernels/"
    r"[0-9a-f]{40}/merge-bundle\.schema\.json$"
)
RESOLUTION_STATES = SAFE_UPDATE_STATES | {"both-changed"}
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def merge_bundle_schema_uri(revision: str) -> str:
    validate_revision(revision)
    return (
        "https://raw.githubusercontent.com/moritzbrantner/rust-kernels/"
        f"{revision}/merge-bundle.schema.json"
    )


def assert_candidate_revision(repository_root: Path, revision: str) -> None:
    validate_revision(revision)
    checkout_revision = current_git_revision(repository_root)
    if revision != checkout_revision:
        raise SourceRegistryError(
            f"candidate revision {revision} does not match checkout {checkout_revision}"
        )


def artifact_path(item: str, target: PurePosixPath, kind: str) -> PurePosixPath:
    return PurePosixPath("conflicts") / item / PurePosixPath(
        f"{target.as_posix()}.{kind}"
    )


def require_empty_output(output: Path) -> None:
    if output.exists():
        if not output.is_dir():
            raise SourceRegistryError(f"merge bundle output is not a directory: {output}")
        if any(output.iterdir()):
            raise SourceRegistryError(
                f"merge bundle output must be empty before export: {output}"
            )


def bundle_conflict_index(bundle: dict) -> dict[tuple[str, str], dict]:
    conflicts = bundle.get("conflicts")
    if not isinstance(conflicts, list) or not conflicts:
        raise SourceRegistryError("merge bundle conflicts must be a non-empty array")

    result: dict[tuple[str, str], dict] = {}
    for index, conflict in enumerate(conflicts):
        if not isinstance(conflict, dict):
            raise SourceRegistryError(
                f"merge bundle conflicts[{index}] must be an object"
            )
        item = conflict.get("item")
        target = conflict.get("target")
        if not isinstance(item, str) or not item:
            raise SourceRegistryError(
                f"merge bundle conflicts[{index}].item must be a non-empty string"
            )
        if not isinstance(target, str) or not target:
            raise SourceRegistryError(
                f"merge bundle conflicts[{index}].target must be a non-empty string"
            )
        key = (item, target)
        if key in result:
            raise SourceRegistryError(
                f"merge bundle contains duplicate conflict {item}:{target}"
            )
        result[key] = conflict
    return result


def validate_bundle(
    bundle: dict,
    candidate_registry: dict,
    candidate_revision: str,
    candidate_registry_sha256: str,
) -> None:
    schema = bundle.get("$schema")
    if (
        not isinstance(schema, str)
        or MERGE_BUNDLE_SCHEMA_RE.fullmatch(schema) is None
        or bundle.get("version") != 1
    ):
        raise SourceRegistryError("merge bundle uses an unsupported format")
    if bundle.get("registry") != registry_identity(candidate_registry):
        raise SourceRegistryError("merge bundle belongs to a different source registry")
    if bundle.get("candidateRevision") != candidate_revision:
        raise SourceRegistryError(
            "merge bundle candidate revision does not match the current checkout"
        )
    if bundle.get("candidateRegistrySha256") != candidate_registry_sha256:
        raise SourceRegistryError(
            "merge bundle registry hash does not match the current candidate registry"
        )
    bundle_conflict_index(bundle)


def verify_hash(value: object, context: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise SourceRegistryError(f"{context} must be a lowercase SHA-256")
    return value


def verify_bundle_artifact(
    bundle_root: Path,
    relative_value: object,
    expected_hash: str,
    context: str,
) -> bytes:
    if not isinstance(relative_value, str) or not relative_value:
        raise SourceRegistryError(f"{context} path must be a non-empty string")
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise SourceRegistryError(f"{context} path must be relative without '..'")
    path = safe_join(bundle_root, relative, f"{context} path")
    if not path.is_file():
        raise SourceRegistryError(f"{context} artifact is missing: {relative}")
    data = path.read_bytes()
    if sha256_bytes(data) != expected_hash:
        raise SourceRegistryError(f"{context} artifact hash does not match its manifest")
    return data


def export_merge_bundle(
    repository_root: Path,
    registry_path: Path,
    candidate_registry: dict,
    lock: dict,
    destination: Path,
    output: Path,
    candidate_revision: str,
    requested: Iterable[str] = (),
) -> dict:
    assert_candidate_revision(repository_root, candidate_revision)
    destination = destination.resolve()
    selected = selected_locked_items(lock, requested)
    selected_names = [item["name"] for item in selected]
    plans = plan_updates(
        repository_root,
        candidate_registry,
        lock,
        destination,
        selected_names,
    )
    blockers = [
        plan
        for plan in plans
        if plan.state not in SAFE_UPDATE_STATES and plan.state != "both-changed"
    ]
    if blockers:
        states = ", ".join(
            f"{plan.item}:{plan.target}={plan.state}" for plan in blockers
        )
        raise SourceRegistryError(
            f"cannot export merge evidence while provenance or layout is unresolved: {states}"
        )

    conflicts = [plan for plan in plans if plan.state == "both-changed"]
    if not conflicts:
        raise SourceRegistryError("selected update has no both-changed conflicts to export")

    require_empty_output(output)

    locked = locked_item_map(lock)
    candidates = item_map(candidate_registry)
    prepared: list[tuple[dict, bytes, bytes, bytes]] = []

    for plan in conflicts:
        locked_item = locked[plan.item]
        candidate_item = candidates[plan.item]
        revision = locked_item.get("revision")
        if not isinstance(revision, str):
            raise SourceRegistryError(f"locked item {plan.item!r} has no revision")
        validate_revision(revision)

        mapping = lock_file_mapping(locked_item, f"locked item {plan.item!r}")
        source_by_target = {target.as_posix(): source for source, target in mapping}
        source_relative = source_by_target.get(plan.target)
        if source_relative is None:
            raise SourceRegistryError(
                f"cannot resolve source mapping for {plan.item}:{plan.target}"
            )
        target_relative = PurePosixPath(plan.target)

        base = git_file_at_revision(repository_root, revision, source_relative)
        target = safe_join(
            destination, target_relative, f"consumer target for {plan.item!r}"
        )
        if not target.is_file():
            raise SourceRegistryError(
                f"consumer conflict target disappeared before export: {plan.target}"
            )
        ours = target.read_bytes()

        candidate_mapping = {
            target.as_posix(): source
            for source, target in registry_file_mapping(
                candidate_item, f"candidate item {plan.item!r}"
            )
        }
        candidate_source_relative = candidate_mapping.get(plan.target)
        if candidate_source_relative != source_relative:
            raise SourceRegistryError(
                f"candidate source mapping changed for {plan.item}:{plan.target}"
            )
        candidate_source = safe_join(
            repository_root,
            source_relative,
            f"candidate source for {plan.item!r}",
        )
        if not candidate_source.is_file():
            raise SourceRegistryError(
                f"candidate source disappeared before export: {source_relative}"
            )
        theirs = candidate_source.read_bytes()

        base_artifact = artifact_path(plan.item, target_relative, "base")
        ours_artifact = artifact_path(plan.item, target_relative, "ours")
        theirs_artifact = artifact_path(plan.item, target_relative, "theirs")
        entry = {
            "item": plan.item,
            "source": source_relative.as_posix(),
            "target": plan.target,
            "baseRevision": revision,
            "baseSha256": sha256_bytes(base),
            "oursSha256": sha256_bytes(ours),
            "theirsSha256": sha256_bytes(theirs),
            "artifacts": {
                "base": base_artifact.as_posix(),
                "ours": ours_artifact.as_posix(),
                "theirs": theirs_artifact.as_posix(),
            },
        }
        prepared.append((entry, base, ours, theirs))

    prepared.sort(key=lambda value: (value[0]["item"], value[0]["target"]))
    manifest = {
        "$schema": merge_bundle_schema_uri(candidate_revision),
        "version": 1,
        "registry": registry_identity(candidate_registry),
        "candidateRevision": candidate_revision,
        "candidateRegistrySha256": sha256_file(registry_path),
        "conflicts": [entry for entry, _base, _ours, _theirs in prepared],
    }

    output.mkdir(parents=True, exist_ok=True)
    for entry, base, ours, theirs in prepared:
        artifacts = entry["artifacts"]
        for kind, data in (("base", base), ("ours", ours), ("theirs", theirs)):
            relative = PurePosixPath(artifacts[kind])
            path = safe_join(output, relative, f"{kind} merge artifact")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    write_lock(output / MERGE_BUNDLE_FILE, manifest)
    return manifest


def accept_merge_bundle(
    repository_root: Path,
    registry_path: Path,
    candidate_registry: dict,
    lock: dict,
    destination: Path,
    bundle_path: Path,
    requested: Iterable[str],
    *,
    allow_ours: bool = False,
) -> list:
    destination = destination.resolve()
    requested_names = list(requested)
    if not requested_names:
        raise SourceRegistryError("accept requires at least one explicit registry item")

    candidate_revision = current_git_revision(repository_root)
    candidate_registry_sha256 = sha256_file(registry_path)
    bundle = load_json(bundle_path)
    validate_bundle(
        bundle,
        candidate_registry,
        candidate_revision,
        candidate_registry_sha256,
    )
    bundle_root = bundle_path.parent.resolve()
    bundle_conflicts = bundle_conflict_index(bundle)

    selected = selected_locked_items(lock, requested_names)
    selected_names = [item["name"] for item in selected]
    plans = plan_updates(
        repository_root,
        candidate_registry,
        lock,
        destination,
        selected_names,
    )
    blockers = [plan for plan in plans if plan.state not in RESOLUTION_STATES]
    if blockers:
        states = ", ".join(
            f"{plan.item}:{plan.target}={plan.state}" for plan in blockers
        )
        raise SourceRegistryError(
            f"cannot accept resolved update while provenance or layout is unresolved: {states}"
        )

    selected_name_set = set(selected_names)
    selected_bundle_conflicts = {
        key: conflict
        for key, conflict in bundle_conflicts.items()
        if key[0] in selected_name_set
    }
    if not selected_bundle_conflicts:
        raise SourceRegistryError(
            "none of the explicitly selected items has a conflict in this merge bundle"
        )

    locked = locked_item_map(lock)
    candidates = item_map(candidate_registry)
    plan_by_target = {(plan.item, plan.target): plan for plan in plans}
    resolved_conflict_keys: set[tuple[str, str]] = set()

    # Preflight every originally exported conflict and every current both-changed
    # target before writing consumer files or advancing provenance.
    for key, conflict in selected_bundle_conflicts.items():
        item_name, target_value = key
        plan = plan_by_target.get(key)
        if plan is None:
            raise SourceRegistryError(
                f"merge bundle conflict is no longer mapped by the selected item: "
                f"{item_name}:{target_value}"
            )
        if plan.state not in {"both-changed", "converged"}:
            raise SourceRegistryError(
                f"merge bundle conflict {item_name}:{target_value} is now {plan.state}; "
                "re-plan and export fresh merge evidence"
            )

        locked_item = locked[item_name]
        base_revision = conflict.get("baseRevision")
        if base_revision != locked_item.get("revision"):
            raise SourceRegistryError(
                f"merge bundle base revision is stale for {item_name}:{target_value}"
            )
        if not isinstance(base_revision, str):
            raise SourceRegistryError(
                f"merge bundle base revision is invalid for {item_name}:{target_value}"
            )
        validate_revision(base_revision)

        source_value = conflict.get("source")
        if not isinstance(source_value, str) or not source_value:
            raise SourceRegistryError(
                f"merge bundle source is invalid for {item_name}:{target_value}"
            )
        source_relative = PurePosixPath(source_value)
        if source_relative.is_absolute() or ".." in source_relative.parts:
            raise SourceRegistryError(
                f"merge bundle source path is unsafe for {item_name}:{target_value}"
            )
        mapping = lock_file_mapping(locked_item, f"locked item {item_name!r}")
        source_by_target = {target.as_posix(): source for source, target in mapping}
        if source_by_target.get(target_value) != source_relative:
            raise SourceRegistryError(
                f"merge bundle source mapping is stale for {item_name}:{target_value}"
            )

        base_hash = verify_hash(
            conflict.get("baseSha256"),
            f"merge bundle {item_name}:{target_value} baseSha256",
        )
        ours_hash = verify_hash(
            conflict.get("oursSha256"),
            f"merge bundle {item_name}:{target_value} oursSha256",
        )
        theirs_hash = verify_hash(
            conflict.get("theirsSha256"),
            f"merge bundle {item_name}:{target_value} theirsSha256",
        )
        artifacts = conflict.get("artifacts")
        if not isinstance(artifacts, dict):
            raise SourceRegistryError(
                f"merge bundle artifacts are invalid for {item_name}:{target_value}"
            )

        bundled_base = verify_bundle_artifact(
            bundle_root,
            artifacts.get("base"),
            base_hash,
            f"{item_name}:{target_value} base",
        )
        bundled_ours = verify_bundle_artifact(
            bundle_root,
            artifacts.get("ours"),
            ours_hash,
            f"{item_name}:{target_value} ours",
        )
        bundled_theirs = verify_bundle_artifact(
            bundle_root,
            artifacts.get("theirs"),
            theirs_hash,
            f"{item_name}:{target_value} theirs",
        )

        actual_base = git_file_at_revision(
            repository_root, base_revision, source_relative
        )
        if actual_base != bundled_base:
            raise SourceRegistryError(
                f"merge bundle base no longer matches Git history for {item_name}:{target_value}"
            )

        candidate_source = safe_join(
            repository_root,
            source_relative,
            f"candidate source for {item_name!r}",
        )
        if not candidate_source.is_file():
            raise SourceRegistryError(
                f"candidate source is missing for {item_name}:{target_value}"
            )
        if candidate_source.read_bytes() != bundled_theirs:
            raise SourceRegistryError(
                f"merge bundle theirs no longer matches the candidate checkout for "
                f"{item_name}:{target_value}"
            )

        target_relative = PurePosixPath(target_value)
        target = safe_join(
            destination,
            target_relative,
            f"consumer target for {item_name!r}",
        )
        if not target.is_file():
            raise SourceRegistryError(
                f"resolved consumer target is missing for {item_name}:{target_value}"
            )
        current = target.read_bytes()
        if plan.state == "both-changed" and current == bundled_ours and not allow_ours:
            raise SourceRegistryError(
                f"consumer target {item_name}:{target_value} is unchanged since export; "
                "resolve it first or pass --allow-ours to explicitly keep the local side"
            )
        resolved_conflict_keys.add(key)

    current_conflicts = {
        (plan.item, plan.target) for plan in plans if plan.state == "both-changed"
    }
    missing_evidence = current_conflicts - resolved_conflict_keys
    if missing_evidence:
        formatted = ", ".join(
            f"{item}:{target}" for item, target in sorted(missing_evidence)
        )
        raise SourceRegistryError(
            f"current conflicts are missing from the accepted merge bundle: {formatted}"
        )

    # Prepare candidate snapshots before performing any writes.
    snapshots: dict[str, dict] = {}
    copies_by_item: dict[str, list] = {}
    for locked_item in selected:
        item_name = locked_item["name"]
        candidate_item = candidates[item_name]
        snapshot, copies = prepare_item_snapshot(
            repository_root,
            candidate_item,
            candidate_revision,
            candidate_registry_sha256,
        )
        snapshots[item_name] = snapshot
        copies_by_item[item_name] = copies

    # Safe sibling changes are applied while resolved conflict targets remain
    # consumer-owned.
    for item_name, copies in copies_by_item.items():
        for source, target_relative, _source_sha256 in copies:
            plan = plan_by_target[(item_name, target_relative.as_posix())]
            if plan.state != "upstream-only":
                continue
            target = safe_join(
                destination,
                target_relative,
                f"resolved update target for {item_name!r}",
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    existing = locked_item_map(lock)
    existing.update(snapshots)
    lock["items"] = [existing[name] for name in sorted(existing)]
    write_lock(lock_path(destination), lock)
    return plans


def command_export(args: argparse.Namespace) -> int:
    candidate_registry = load_json(REGISTRY_PATH)
    destination = Path(args.root)
    lock = validate_required_lock(destination, candidate_registry)
    selected = selected_locked_items(lock, args.items)
    ensure_clean_sources(
        ROOT,
        candidate_items_for_clean_check(candidate_registry, selected),
    )
    candidate_revision = current_git_revision(ROOT)
    manifest = export_merge_bundle(
        ROOT,
        REGISTRY_PATH,
        candidate_registry,
        lock,
        destination,
        Path(args.out),
        candidate_revision,
        [item["name"] for item in selected],
    )
    print(
        f"exported {len(manifest['conflicts'])} conflict(s) at {candidate_revision} "
        f"to {Path(args.out) / MERGE_BUNDLE_FILE}"
    )
    return 0


def command_accept(args: argparse.Namespace) -> int:
    candidate_registry = load_json(REGISTRY_PATH)
    destination = Path(args.root)
    lock = validate_required_lock(destination, candidate_registry)
    selected = selected_locked_items(lock, args.items)
    ensure_clean_sources(
        ROOT,
        candidate_items_for_clean_check(candidate_registry, selected),
    )
    plans = accept_merge_bundle(
        ROOT,
        REGISTRY_PATH,
        candidate_registry,
        lock,
        destination,
        Path(args.bundle),
        [item["name"] for item in selected],
        allow_ours=args.allow_ours,
    )
    print(
        f"accepted resolved provenance for {', '.join(item['name'] for item in selected)} "
        f"at {current_git_revision(ROOT)}"
    )
    for plan in plans:
        print(f"{plan.state:20} {plan.item}: {plan.target}")
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Reference conflict handoff for rust-kernels source updates."
    )
    subcommands = value.add_subparsers(dest="command", required=True)

    export = subcommands.add_parser(
        "export",
        help="materialize deterministic base/ours/theirs evidence for both-changed files",
    )
    export.add_argument("items", nargs="*", help="locked item names; defaults to all")
    export.add_argument("--root", required=True, help="consumer repository root")
    export.add_argument(
        "--out",
        required=True,
        help="empty directory that will receive merge-bundle.json and conflict artifacts",
    )
    export.set_defaults(handler=command_export)

    accept = subcommands.add_parser(
        "accept",
        help="accept explicitly resolved conflicts and advance their upstream provenance",
    )
    accept.add_argument("items", nargs="+", help="resolved registry item names")
    accept.add_argument("--root", required=True, help="consumer repository root")
    accept.add_argument(
        "--bundle",
        required=True,
        help="merge-bundle.json created by the export command",
    )
    accept.add_argument(
        "--allow-ours",
        action="store_true",
        help="explicitly allow an exported conflict to remain byte-identical to its local side",
    )
    accept.set_defaults(handler=command_accept)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        return args.handler(args)
    except SourceRegistryError as error:
        print(f"source resolution error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
