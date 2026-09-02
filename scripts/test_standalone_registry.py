#!/usr/bin/env python3

import json
import subprocess
import tempfile
from pathlib import Path

from source_registry import LOCK_FILE_NAME, install_items, load_json, provenance_status

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "registry.json"
FIXTURE_REVISION = "a" * 40


def fail(message: str) -> None:
    raise SystemExit(f"standalone registry test failed: {message}")


try:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    fail(f"cannot read registry.json: {error}")

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
    fail("registry must contain at least one standalone module")

with tempfile.TemporaryDirectory() as temporary:
    output_root = Path(temporary)
    names: list[str] = []

    for item in standalone:
        name = item.get("name")
        integration = item.get("integration")
        files = item.get("files")
        if not isinstance(name, str) or not name:
            fail("standalone item must have a name")
        names.append(name)
        if not isinstance(integration, dict):
            fail(f"{name!r} integration must be an object")
        if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
            fail(f"{name!r} must contain exactly one file")

        source_value = files[0].get("path")
        module = integration.get("module")
        if not isinstance(source_value, str) or not source_value:
            fail(f"{name!r} source path must be a string")
        if not isinstance(module, str) or not module:
            fail(f"{name!r} module must be a string")

        source = ROOT / source_value
        if not source.is_file():
            fail(f"{name!r} source does not exist: {source_value}")

        crate_name = f"registry_{module}"
        output = output_root / f"{crate_name}.rmeta"
        result = subprocess.run(
            [
                "rustc",
                "--edition=2024",
                "--crate-type=lib",
                "--emit=metadata",
                "--crate-name",
                crate_name,
                str(source),
                "-o",
                str(output),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            fail(f"{name!r} does not compile as an independent module:\n{detail}")

    consumer = output_root / "consumer"
    installed = install_items(
        ROOT,
        REGISTRY_PATH,
        registry,
        names,
        consumer,
        FIXTURE_REVISION,
    )
    if installed != names:
        fail(f"standalone install order changed: expected {names}, got {installed}")

    for item in standalone:
        name = item["name"]
        file_entry = item["files"][0]
        source = ROOT / file_entry["path"]
        target = consumer / file_entry["target"]
        if not target.is_file():
            fail(f"{name!r} did not create {file_entry['target']}")
        if target.read_bytes() != source.read_bytes():
            fail(f"{name!r} copied bytes differ from the registered source")

    lock = load_json(consumer / LOCK_FILE_NAME)
    statuses = provenance_status(consumer, lock)
    dirty = [status for status in statuses if status[2] != "clean"]
    if dirty:
        fail(f"fresh standalone install is not provenance-clean: {dirty}")

print(
    f"standalone registry ok: {len(standalone)} module(s) compile and install independently"
)
