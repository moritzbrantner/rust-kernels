#!/usr/bin/env python3

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "registry.json"


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
    for item in standalone:
        name = item.get("name")
        integration = item.get("integration")
        files = item.get("files")
        if not isinstance(name, str) or not name:
            fail("standalone item must have a name")
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

print(f"standalone registry ok: {len(standalone)} module(s) compile independently")
