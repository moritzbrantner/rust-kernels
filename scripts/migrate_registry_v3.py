#!/usr/bin/env python3

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "registry.json"

CHARACTERISTICS = {
    "morton-code": {
        "deterministic": True,
        "operations": [
            {
                "operation": "2D/3D encode and decode",
                "time": "O(1)",
                "extraSpace": "O(1)",
                "mutation": "none",
                "allocation": "none",
            }
        ],
    },
    "bit-set": {
        "deterministic": True,
        "operations": [
            {
                "operation": "construct fixed universe",
                "time": "O(w), w = 64-bit word count",
                "extraSpace": "O(w)",
                "mutation": "none",
                "allocation": "input-sized",
            },
            {
                "operation": "set / clear",
                "time": "O(1)",
                "extraSpace": "O(1)",
                "mutation": "internal-state",
                "allocation": "none",
            },
            {
                "operation": "contains",
                "time": "O(1)",
                "extraSpace": "O(1)",
                "mutation": "none",
                "allocation": "none",
            },
            {
                "operation": "clear all",
                "time": "O(w), w = 64-bit word count",
                "extraSpace": "O(1)",
                "mutation": "internal-state",
                "allocation": "none",
            },
            {
                "operation": "iterate set bits",
                "time": "O(w + k), w = word count, k = yielded set bits",
                "extraSpace": "O(1)",
                "mutation": "none",
                "allocation": "none",
            },
        ],
    },
    "fenwick-tree": {
        "deterministic": True,
        "operations": [
            {
                "operation": "construct empty tree",
                "time": "O(n)",
                "extraSpace": "O(n)",
                "mutation": "none",
                "allocation": "input-sized",
            },
            {
                "operation": "build from slice",
                "time": "O(n log n)",
                "extraSpace": "O(n)",
                "mutation": "none",
                "allocation": "input-sized",
            },
            {
                "operation": "add point delta",
                "time": "O(log n)",
                "extraSpace": "O(1)",
                "mutation": "internal-state",
                "allocation": "none",
            },
            {
                "operation": "prefix / range sum",
                "time": "O(log n)",
                "extraSpace": "O(1)",
                "mutation": "none",
                "allocation": "none",
            },
        ],
    },
    "generational-arena": {
        "deterministic": True,
        "operations": [
            {
                "operation": "insert",
                "time": "O(1) amortized",
                "extraSpace": "O(1) excluding retained arena storage",
                "mutation": "internal-state",
                "allocation": "may-grow",
            },
            {
                "operation": "get",
                "time": "O(1)",
                "extraSpace": "O(1)",
                "mutation": "none",
                "allocation": "none",
            },
            {
                "operation": "get mutable",
                "time": "O(1)",
                "extraSpace": "O(1)",
                "mutation": "internal-state",
                "allocation": "none",
            },
            {
                "operation": "remove",
                "time": "O(1)",
                "extraSpace": "O(1)",
                "mutation": "internal-state",
                "allocation": "none",
            },
            {
                "operation": "clear",
                "time": "O(s), s = slot count",
                "extraSpace": "O(1)",
                "mutation": "internal-state",
                "allocation": "none",
            },
            {
                "operation": "iterate live entries",
                "time": "O(s), s = slot count",
                "extraSpace": "O(1)",
                "mutation": "none",
                "allocation": "none",
            },
        ],
    },
    "lru-cache": {
        "deterministic": True,
        "operations": [
            {
                "operation": "get",
                "time": "O(1) expected",
                "extraSpace": "O(1)",
                "mutation": "internal-state",
                "allocation": "none",
            },
            {
                "operation": "peek / contains",
                "time": "O(1) expected",
                "extraSpace": "O(1)",
                "mutation": "none",
                "allocation": "none",
            },
            {
                "operation": "insert",
                "time": "O(1) expected",
                "extraSpace": "O(1)",
                "mutation": "internal-state",
                "allocation": "capacity-bounded",
            },
            {
                "operation": "remove",
                "time": "O(1) expected",
                "extraSpace": "O(1)",
                "mutation": "internal-state",
                "allocation": "none",
            },
            {
                "operation": "clear",
                "time": "O(n)",
                "extraSpace": "O(1)",
                "mutation": "internal-state",
                "allocation": "none",
            },
            {
                "operation": "iterate recency order",
                "time": "O(n)",
                "extraSpace": "O(1)",
                "mutation": "none",
                "allocation": "none",
            },
        ],
    },
    "ring-buffer": {
        "deterministic": True,
        "operations": [
            {
                "operation": "construct",
                "time": "O(c), c = capacity",
                "extraSpace": "O(c)",
                "mutation": "none",
                "allocation": "input-sized",
            },
            {
                "operation": "push / pop",
                "time": "O(1)",
                "extraSpace": "O(1)",
                "mutation": "internal-state",
                "allocation": "none",
            },
            {
                "operation": "peek",
                "time": "O(1)",
                "extraSpace": "O(1)",
                "mutation": "none",
                "allocation": "none",
            },
            {
                "operation": "clear",
                "time": "O(c), c = capacity",
                "extraSpace": "O(1)",
                "mutation": "internal-state",
                "allocation": "none",
            },
            {
                "operation": "iterate",
                "time": "O(n)",
                "extraSpace": "O(1)",
                "mutation": "none",
                "allocation": "none",
            },
        ],
    },
    "sparse-set": {
        "deterministic": True,
        "operations": [
            {
                "operation": "contains",
                "time": "O(1)",
                "extraSpace": "O(1)",
                "mutation": "none",
                "allocation": "none",
            },
            {
                "operation": "insert",
                "time": "O(1) amortized when the sparse slot exists; O(key - s) when growing sparse storage",
                "extraSpace": "O(key - s) when growing, s = previous sparse length",
                "mutation": "internal-state",
                "allocation": "may-grow",
            },
            {
                "operation": "remove",
                "time": "O(1)",
                "extraSpace": "O(1)",
                "mutation": "internal-state",
                "allocation": "none",
            },
            {
                "operation": "clear",
                "time": "O(n), n = dense key count",
                "extraSpace": "O(1)",
                "mutation": "internal-state",
                "allocation": "none",
            },
            {
                "operation": "iterate dense keys",
                "time": "O(n)",
                "extraSpace": "O(1)",
                "mutation": "none",
                "allocation": "none",
            },
        ],
    },
    "union-find": {
        "deterministic": True,
        "operations": [
            {
                "operation": "construct",
                "time": "O(n)",
                "extraSpace": "O(n)",
                "mutation": "none",
                "allocation": "input-sized",
            },
            {
                "operation": "find",
                "time": "O(alpha(n)) amortized",
                "extraSpace": "O(1)",
                "mutation": "internal-state",
                "allocation": "none",
            },
            {
                "operation": "union / connected / component size",
                "time": "O(alpha(n)) amortized",
                "extraSpace": "O(1)",
                "mutation": "internal-state",
                "allocation": "none",
            },
        ],
    },
    "selection": {
        "deterministic": True,
        "operations": [
            {
                "operation": "quickselect",
                "time": "O(n) average; O(n^2) worst",
                "extraSpace": "O(1)",
                "mutation": "input",
                "allocation": "none",
            },
            {
                "operation": "top-k smallest",
                "time": "O(n + k log k) average; O(n^2 + k log k) worst",
                "extraSpace": "O(n)",
                "mutation": "none",
                "allocation": "input-sized",
            },
        ],
    },
    "radix-sort": {
        "deterministic": True,
        "operations": [
            {
                "operation": "sort u32",
                "time": "O(4 * (n + 256)) = O(n)",
                "extraSpace": "O(n + 256) = O(n)",
                "mutation": "input",
                "allocation": "input-sized",
            },
            {
                "operation": "sort u64",
                "time": "O(8 * (n + 256)) = O(n)",
                "extraSpace": "O(n + 256) = O(n)",
                "mutation": "input",
                "allocation": "input-sized",
            },
        ],
    },
    "running-statistics": {
        "deterministic": True,
        "operations": [
            {
                "operation": "push / merge",
                "time": "O(1)",
                "extraSpace": "O(1)",
                "mutation": "internal-state",
                "allocation": "none",
            },
            {
                "operation": "extend",
                "time": "O(n)",
                "extraSpace": "O(1)",
                "mutation": "internal-state",
                "allocation": "none",
            },
            {
                "operation": "read statistics",
                "time": "O(1)",
                "extraSpace": "O(1)",
                "mutation": "none",
                "allocation": "none",
            },
        ],
    },
}

registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
items = registry.get("items")
if not isinstance(items, list):
    raise SystemExit("registry items must be an array")

standalone = {
    item.get("name")
    for item in items
    if isinstance(item, dict)
    and isinstance(item.get("integration"), dict)
    and item["integration"].get("mode") == "standalone-module"
}
expected = set(CHARACTERISTICS)
if standalone != expected:
    missing = sorted(expected - standalone)
    unexpected = sorted(standalone - expected)
    raise SystemExit(
        f"standalone catalog changed; refusing v3 migration; missing={missing}, unexpected={unexpected}"
    )

registry["version"] = 3
for item in items:
    name = item.get("name")
    if name not in CHARACTERISTICS:
        continue
    files = item.get("files")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise SystemExit(f"{name}: expected exactly one registered file")
    source = files[0].get("path")
    if not isinstance(source, str) or not source:
        raise SystemExit(f"{name}: registered source path is invalid")
    item["characteristics"] = CHARACTERISTICS[name]
    item["verification"] = {
        "tests": [source],
        "benchmarks": [],
    }

REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
print(f"migrated registry to v3 with {len(expected)} characterized standalone items")
