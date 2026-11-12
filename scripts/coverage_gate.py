#!/usr/bin/env python3
"""Per-crate line-coverage regression gate.

Reads a `cargo llvm-cov --json --summary-only` report, aggregates line
coverage per workspace member, and compares it against `coverage-baseline.json`.

    python3 scripts/coverage_gate.py check  target/llvm-cov-baseline/summary.json
    python3 scripts/coverage_gate.py update target/llvm-cov-baseline/summary.json

`check` fails when a crate drops more than the baseline's tolerance below its
recorded percentage, or when the measured crate set differs from the baseline.
`update` rewrites the baseline from the report and keeps the tolerance.
"""

import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "coverage-baseline.json"
WORKSPACE_PATH = ROOT / "Cargo.toml"
DEFAULT_TOLERANCE = 1.0
UPDATE_HINT = (
    "update the baseline with `python3 scripts/coverage_gate.py update <summary.json>` "
    "(see docs/verification.md)"
)


def fail(message: str) -> None:
    raise SystemExit(f"coverage gate failed: {message}")


def workspace_members() -> dict[str, str]:
    """Map each member directory (e.g. `crates/bvh-kernels`) to its package name."""
    workspace = tomllib.loads(WORKSPACE_PATH.read_text())
    members = {}
    for member in workspace["workspace"]["members"]:
        manifest = tomllib.loads((ROOT / member / "Cargo.toml").read_text())
        members[member] = manifest["package"]["name"]
    return members


def measure(report_path: Path) -> dict[str, dict[str, int]]:
    try:
        report = json.loads(report_path.read_text())
        files = report["data"][0]["files"]
    except (OSError, ValueError, KeyError, IndexError) as error:
        fail(f"cannot read llvm-cov JSON summary {report_path}: {error}")

    members = workspace_members()
    totals = {name: {"lines": 0, "covered": 0} for name in members.values()}
    for entry in files:
        # Report paths are absolute on the machine that ran the tests; match on
        # the member directory so a downloaded CI report also works locally.
        filename = "/" + entry["filename"].replace("\\", "/")
        owners = [name for member, name in members.items() if f"/{member}/" in filename]
        if len(owners) != 1:
            fail(f"cannot attribute {entry['filename']} to exactly one workspace member")
        lines = entry["summary"]["lines"]
        totals[owners[0]]["lines"] += lines["count"]
        totals[owners[0]]["covered"] += lines["covered"]
    return {name: total for name, total in totals.items() if total["lines"] > 0}


def percent(total: dict[str, int]) -> float:
    return round(100.0 * total["covered"] / total["lines"], 2)


def load_baseline() -> dict:
    try:
        return json.loads(BASELINE_PATH.read_text())
    except (OSError, ValueError) as error:
        fail(f"cannot read {BASELINE_PATH.name}: {error}; {UPDATE_HINT}")


def update(report_path: Path) -> None:
    tolerance = DEFAULT_TOLERANCE
    if BASELINE_PATH.exists():
        tolerance = load_baseline()["tolerancePercentagePoints"]
    measured = measure(report_path)
    baseline = {
        "metric": "lines",
        "tolerancePercentagePoints": tolerance,
        "crates": {
            name: {"percent": percent(total), "lines": total["lines"]}
            for name, total in sorted(measured.items())
        },
    }
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"wrote {BASELINE_PATH.name} for {len(measured)} crates")


def check(report_path: Path) -> None:
    baseline = load_baseline()
    tolerance = baseline["tolerancePercentagePoints"]
    recorded = baseline["crates"]
    measured = measure(report_path)

    rows = []
    failures = []
    for name in sorted(set(recorded) | set(measured)):
        if name not in measured:
            failures.append(f"{name} is in the baseline but was not measured")
            rows.append((name, f"{recorded[name]['percent']:.2f}", "-", "-", "missing"))
            continue
        current = percent(measured[name])
        if name not in recorded:
            failures.append(f"{name} has no recorded baseline")
            rows.append((name, "-", f"{current:.2f}", "-", "new"))
            continue
        expected = recorded[name]["percent"]
        delta = current - expected
        status = "ok"
        if delta < -tolerance:
            status = "regressed"
            failures.append(
                f"{name} line coverage {current:.2f}% is {-delta:.2f} pp below "
                f"its baseline {expected:.2f}% (tolerance {tolerance} pp)"
            )
        elif delta > tolerance:
            status = "improved; consider updating the baseline"
        rows.append((name, f"{expected:.2f}", f"{current:.2f}", f"{delta:+.2f}", status))

    print(f"Line coverage per crate (tolerance {tolerance} pp below baseline)")
    print()
    print("| crate | baseline % | current % | delta pp | status |")
    print("| --- | ---: | ---: | ---: | --- |")
    for row in rows:
        print("| " + " | ".join(row) + " |")

    if failures:
        print()
        for failure in failures:
            print(f"- {failure}")
        fail(f"{len(failures)} crate(s) outside the baseline; if the change is legitimate, {UPDATE_HINT}")


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in {"check", "update"}:
        raise SystemExit(f"usage: {sys.argv[0]} check|update <llvm-cov summary.json>")
    command, report = sys.argv[1], Path(sys.argv[2])
    (check if command == "check" else update)(report)


if __name__ == "__main__":
    main()
