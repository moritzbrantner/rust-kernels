#!/usr/bin/env python3
"""Run the same native tests/Divan harness against the audit base and candidate.

Correctness and deterministic work are gates. Shared-runner latency is evidence,
never a gate. Nothing is checked out over the caller's working tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
AUDIT_BASE = "ddec70d25398341c7130ee68a95e4e0db34e34fc"
EXPECTED_FAILURES = {
    "audit_crossing_short_segments_have_zero_distance",
    "audit_crossing_short_capsules_overlap",
    "audit_intersecting_near_parallel_segments_do_not_collapse_to_endpoint_case",
    "audit_sphere_support_preserves_small_nonzero_direction",
    "audit_sphere_support_preserves_large_finite_direction",
    "audit_capsule_support_preserves_small_nonzero_direction",
    "audit_gjk_agrees_with_sphere_oracle_for_well_resolved_small_shapes",
    "audit_planar_adapter_resolves_small_overlapping_hulls",
    "audit_merge_preserves_finite_representable_variance",
}
CONTROLS = {
    "audit_control_unit_scale_spheres_overlap",
    "audit_control_unit_scale_crossing_segments_have_zero_distance",
    "audit_control_ordinary_merge_matches_single_pass",
}
CRATES = {"geometry": "geometry-kernels", "statistics": "statistics-kernels"}
BENCHMARKS = {
    "geometry/capsule_small_crossing", "geometry/segment_near_parallel",
    "statistics/merge_ordinary", "statistics/merge_large_finite",
    *(f"geometry/{name}/{scale}" for name in ["capsule_support", "sphere_support"]
      for scale in [-1023, -44, 0, 1023]),
    *(f"geometry/{name}/{scale}" for name in ["segment_crossing", "gjk_overlap"]
      for scale in [-24, 0, 24]),
    *(f"geometry/{name}/{vertices}" for name in ["epa_no_trace", "epa_trace"]
      for vertices in [4, 16, 64]),
    *(f"statistics/merge_chunks/{chunks}" for chunks in [4, 64, 1024]),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(command: list[str], cwd: Path, log: Path, env: dict[str, str], *, allow_failure: bool = False) -> str:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=900, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout, encoding="utf-8")
    require(allow_failure or result.returncode == 0,
            f"command failed ({result.returncode}): {' '.join(command)}; see {log}")
    return result.stdout


def parse_tests(text: str) -> dict[str, str]:
    matches = re.findall(r"^test (audit_[a-z0-9_]+) \.\.\. (ok|FAILED)$", text, re.MULTILINE)
    require(len(matches) == len(dict(matches)), "duplicate native test results")
    result = dict(matches)
    require(set(result) == EXPECTED_FAILURES | CONTROLS,
            "native test inventory is incomplete or changed; compile errors are not red-test evidence")
    return result


def parse_probes(text: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for line in text.splitlines():
        if not line.startswith("AUDIT\t"):
            continue
        _, case, metric, raw = line.split("\t", 3)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw  # Debug status, exact witness record, or non-finite result.
        require(metric not in result.setdefault(case, {}), f"duplicate probe: {case}/{metric}")
        result[case][metric] = value
    require(bool(result), "no native correctness probes were emitted")
    return result


def duration_ns(value: str) -> float:
    match = re.fullmatch(r"\s*([\d.]+)\s*(ps|ns|µs|ms|s)\s*", value)
    require(match is not None, f"unsupported Divan duration: {value!r}")
    assert match is not None
    return float(match[1]) * {"ps": .001, "ns": 1, "µs": 1_000, "ms": 1_000_000, "s": 1_000_000_000}[match[2]]


def parse_divan(text: str) -> dict[str, dict[str, float]]:
    """Parse pinned Divan 0.1.21's tree, keeping real alloc/realloc counts.

    The 'max alloc' row is intentionally NOT treated as allocations per call.
    Raw output is retained so formatting changes fail closed, not silently pass.
    """
    result: dict[str, dict[str, float]] = {}
    stack: list[str] = []
    current: str | None = None
    pending_counter: str | None = None
    for line in text.splitlines():
        branch = re.match(r"^([ │]*)(?:├─|╰─) (.*)$", line)
        if branch:
            depth = len(branch[1]) // 3
            columns = branch[2].split("│")
            first = columns[0].strip()
            timed = re.fullmatch(r"(.*?)\s+([\d.]+\s*(?:ps|ns|µs|ms|s))", first)
            name = timed[1].strip() if timed else first
            stack = stack[:depth] + [name]
            pending_counter = None
            if timed:
                require(len(columns) >= 6, "incomplete Divan timing row")
                current = "/".join(stack)
                require(current not in result, f"duplicate benchmark {current}")
                result[current] = {"median_ns": duration_ns(columns[2]),
                                   "alloc_calls": 0.0, "grow_calls": 0.0, "shrink_calls": 0.0}
            else:
                current = None
            continue
        stripped = line.lstrip(" │")
        label = stripped.split("│", 1)[0].strip()
        if label in {"alloc:", "grow:", "shrink:"}:
            pending_counter = label[:-1] + "_calls"
            continue
        if pending_counter and current:
            columns = stripped.split("│")
            require(len(columns) >= 4, "incomplete allocation row")
            count = columns[2].strip()
            require(re.fullmatch(r"[\d.]+", count) is not None, f"unsupported allocation count {count!r}")
            result[current][pending_counter] = float(count)
            pending_counter = None
    require(bool(result), "no timed benchmarks; native Divan execution requires --bench")
    return result


def write_manifest(directory: Path, library_root: Path, candidate_root: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    lines = ['[package]', 'name = "kernel-audit-harness"', 'version = "0.0.0"',
             'edition = "2024"', 'rust-version = "1.85"', '[workspace]', '[lib]',
             'path = "lib.rs"', '[dependencies]', 'divan = "=0.1.21"']
    for crate in CRATES.values():
        lines.append(f'{crate} = {{ path = {json.dumps(str(library_root / "crates" / crate))} }}')
    for kind, crate in CRATES.items():
        base = candidate_root / "crates" / crate
        lines += ['[[test]]', f'name = "{kind}_audit"',
                  f'path = {json.dumps(str(base / "tests/audit_regressions.rs"))}',
                  '[[bench]]', f'name = "{kind}_audit"', 'harness = false',
                  f'path = {json.dumps(str(base / "benches/audit_numerics.rs"))}']
    (directory / "lib.rs").write_text("// External comparison harness; no copied kernel implementation.\n")
    manifest = directory / "Cargo.toml"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def harness_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for crate in CRATES.values():
        for relative in ["tests/audit_regressions.rs", "benches/audit_numerics.rs"]:
            path = root / "crates" / crate / relative
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_inventory(rows: dict) -> None:
    require(set(rows) == BENCHMARKS,
            f"workload v1 requires all 27 benchmark rows; missing={BENCHMARKS - set(rows)}, unexpected={set(rows) - BENCHMARKS}")


def source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for crate in ["geometry-kernels", "statistics-kernels", "spatial-kernels"]:
        directory = root / "crates" / crate
        for path in sorted([directory / "Cargo.toml", *directory.joinpath("src").rglob("*.rs")]):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_probes(candidate: dict, baseline: dict) -> None:
    require(candidate["segment_small"]["distance"] == 0.0, "small segment distance regressed")
    for field in ["left_parameter", "right_parameter"]:
        require(candidate["segment_small"][field] == 0.5, f"small segment {field} regressed")
    require(candidate["capsule_small"]["overlaps"] is True, "crossing capsule false negative")
    require(candidate["segment_near_parallel"]["distance"] <= 1e-9, "near-parallel distance regressed")
    for case in ["support_small", "support_large"]:
        require(candidate[case]["point"] == [0.0, 1.0, 0.0], f"{case} direction changed")
    for exponent in [-24, 0, 24]:
        result = candidate[f"gjk_{exponent}"]
        require(result["status"] == "Intersecting" and result["iterations"] <= 8,
                f"GJK correctness/work budget failed at scale 2^{exponent}")
    require(math.isclose(candidate["merge_large"]["variance"], 2.5e307, rel_tol=1e-13),
            "finite merged variance regressed")
    require(candidate["merge_ordinary"] == {"mean": 1.0, "variance": 1.0}, "ordinary merge changed")
    for vertices in [4, 16, 64]:
        require(candidate[f"epa_{vertices}"]["status"] == "Converged", f"EPA {vertices} did not converge")
    # Only this already-correct workload supports an isolated speedup claim.
    require(candidate["epa_4"]["seed"] == baseline["epa_4"]["seed"], "EPA-4 seeds differ; timing incomparable")
    require(candidate["epa_4"]["result"] == baseline["epa_4"]["result"], "EPA-4 result/witnesses differ")


def summarize(output: Path, data: dict) -> None:
    tests = data["tests"]
    lines = ["# Native numerical audit evidence", "", f"Baseline: `{data['baseline']}`.",
             "The same current test/benchmark harness is compiled separately against each source tree.", "",
             "## Audited regression tests", "", "| Revision | Passed | Failed |", "|---|---:|---:|"]
    for label in ["baseline", "candidate"]:
        lines.append(f"| {label} | {sum(v == 'ok' for v in tests[label].values())} | {sum(v == 'FAILED' for v in tests[label].values())} |")
    lines += ["", "Only the known baseline failures are accepted; missing tests and compiler failures are fatal.",
              "Expanded scale sweeps, independent oracles and snapshot-copy gates also run in the ordinary workspace suite.",
              "", "## Behavioral and work evidence", "", "| Probe | Before | After |", "|---|---|---|"]
    fields = [("segment_small", "distance"), ("capsule_small", "overlaps"),
              ("segment_near_parallel", "distance"), ("support_small", "point"), ("support_large", "point"),
              ("gjk_-24", "status"), ("gjk_-24", "iterations"), ("merge_large", "variance"),
              ("epa_4", "status"), ("epa_16", "status"), ("epa_64", "status")]
    for case, metric in fields:
        before, after = (data["probes"][label][case][metric] for label in ["baseline", "candidate"])
        lines.append(f"| `{case}/{metric}` | `{before}` | `{after}` |")
    lines += ["", "## Allocation-profiled wall time (informational)", "",
              f"Median of {data['trials']} alternating-order runs; each uses {data['samples']} Divan samples of {data['sampleSize']} calls.",
              "Allocations are `alloc` calls per invocation (reallocations are retained separately in JSON).",
              "The allocation profiler affects timing on both revisions. There is no wall-clock CI threshold.", "",
              "| Workload | Before (ns) | After (ns) | After / before | Alloc calls before / after | Interpretation |",
              "|---|---:|---:|---:|---:|---|"]
    selected = {
        "geometry/gjk_overlap/-24": "Previously indeterminate; now correct",
        "geometry/gjk_overlap/0": "Ordinary-scale control",
        "geometry/epa_no_trace/4": "Exact same seed, result and witnesses",
        "geometry/epa_no_trace/16": "Previously invalid; not an equivalent-result speed comparison",
        "geometry/epa_no_trace/64": "Previously invalid; not an equivalent-result speed comparison",
        "statistics/merge_ordinary": "Ordinary-value control",
        "statistics/merge_large_finite": "Previously infinite instead of finite",
    }
    for name, note in selected.items():
        before = data["benchmarks"]["baseline"][name]
        after = data["benchmarks"]["candidate"][name]
        lines.append(f"| `{name}` | {before['median_ns']:.2f} | {after['median_ns']:.2f} | "
                     f"{after['median_ns']/before['median_ns']:.3f} | {before['alloc_calls']:g} / {after['alloc_calls']:g} | {note} |")
    lines += ["", "Complete benchmark rows, raw per-trial output, correctness probes, source hashes, dependency lock hash,",
              "toolchain and platform fingerprint are retained beside this report.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=AUDIT_BASE)
    parser.add_argument("--baseline-root", type=Path, help="existing baseline directory instead of a Git worktree")
    parser.add_argument("--output", type=Path, default=ROOT / "target/audit-evidence")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    require(min(args.samples, args.sample_size, args.trials) > 0, "sampling values must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CARGO_TERM_COLOR="never", RUST_BACKTRACE="0")
    data: dict = {"workloadVersion": 1, "baseline": args.baseline, "samples": args.samples,
                  "sampleSize": args.sample_size, "trials": args.trials, "tests": {}, "probes": {},
                  "benchmarks": {}, "sourceSha256": {}, "fingerprint": {
                      "platform": platform.platform(), "machine": platform.machine(),
                      "cpuCount": os.cpu_count(), "allocationProfiler": "Divan 0.1.21 system allocator",
                      "harnessSha256": harness_fingerprint(ROOT),
                      "candidateCheckout": run(["git", "rev-parse", "HEAD"], ROOT, output / "candidate-checkout.txt", env).strip(),
                      "profile": "bench (optimized)", "rustflags": env.get("RUSTFLAGS", ""),
                      "rustc": run(["rustc", "-vV"], ROOT, output / "rustc.txt", env),
                      "cargo": run(["cargo", "-V"], ROOT, output / "cargo.txt", env)}}
    with tempfile.TemporaryDirectory(prefix="kernel-audit-") as temporary:
        directory = Path(temporary)
        baseline = args.baseline_root.resolve() if args.baseline_root else directory / "base"
        managed_worktree = args.baseline_root is None
        if managed_worktree:
            run(["git", "worktree", "add", "--detach", str(baseline), args.baseline], ROOT, output / "worktree.txt", env)
        try:
            executables: dict[str, dict[str, str]] = {}
            dependency_lock: bytes | None = None
            for label, source in [("baseline", baseline), ("candidate", ROOT)]:
                print(f"Building and testing {label}: {source}", flush=True)
                data["sourceSha256"][label] = source_fingerprint(source)
                manifest = write_manifest(directory / label, source, ROOT)
                target = output / "build" / label
                build_env = dict(env, CARGO_TARGET_DIR=str(target))
                if dependency_lock is None:
                    run(["cargo", "generate-lockfile", "--manifest-path", str(manifest)], ROOT, output / label / "lock.log", build_env)
                    dependency_lock = manifest.with_name("Cargo.lock").read_bytes()
                    (output / "Cargo.lock").write_bytes(dependency_lock)
                    data["fingerprint"]["cargoLockSha256"] = hashlib.sha256(dependency_lock).hexdigest()
                else:
                    manifest.with_name("Cargo.lock").write_bytes(dependency_lock)
                test_text = run(["cargo", "test", "--manifest-path", str(manifest), "--locked", "--tests",
                                 "--no-fail-fast", "audit_", "--", "--test-threads=1"],
                                ROOT, output / label / "tests.log", build_env, allow_failure=True)
                tests = parse_tests(test_text)
                data["tests"][label] = tests
                failed = {name for name, status in tests.items() if status == "FAILED"}
                require(not failed if label == "candidate" else not (failed & CONTROLS), f"unexpected {label} test failures: {failed}")
                if label == "baseline" and args.baseline == AUDIT_BASE:
                    require(failed == EXPECTED_FAILURES, "audited baseline did not reproduce exactly its nine known failures")
                build_text = run(["cargo", "bench", "--manifest-path", str(manifest), "--locked", "--no-run", "--message-format=json"],
                                 ROOT, output / label / "build.log", build_env)
                executables[label] = {}
                for line in build_text.splitlines():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("reason") == "compiler-artifact" and event.get("executable") and "bench" in event["target"]["kind"]:
                        executables[label][event["target"]["name"].removesuffix("_audit")] = event["executable"]
                require(set(executables[label]) == set(CRATES), f"missing benchmark executables for {label}")
                probes: dict = {}
                for kind, executable in executables[label].items():
                    probes.update(parse_probes(run([executable, "--audit-probe"], ROOT, output / label / f"{kind}-probes.tsv", env)))
                data["probes"][label] = probes
            validate_probes(data["probes"]["candidate"], data["probes"]["baseline"])
            measurements: dict[str, dict[str, list]] = {"baseline": {}, "candidate": {}}
            for trial in range(args.trials):
                labels = ["baseline", "candidate"] if trial % 2 == 0 else ["candidate", "baseline"]
                for label in labels:
                    for kind, executable in executables[label].items():
                        print(f"Timing {label}/{kind}, trial {trial + 1}", flush=True)
                        text = run([executable, "--bench", "--sample-count", str(args.samples), "--sample-size", str(args.sample_size),
                                    "--threads", "1", "--color", "never", "--timer", "os"],
                                   ROOT, output / label / f"{kind}-trial-{trial + 1}.txt", env)
                        for name, metrics in parse_divan(text).items():
                            measurements[label].setdefault(f"{kind}/{name}", []).append(metrics)
            require(set(measurements["baseline"]) == set(measurements["candidate"]), "benchmark inventory mismatch")
            for label, rows in measurements.items():
                validate_inventory(rows)
                data["benchmarks"][label] = {}
                for name, trials in rows.items():
                    require(len(trials) == args.trials, f"missing trials for {label}/{name}")
                    for counter in ["alloc_calls", "grow_calls", "shrink_calls"]:
                        require(len({trial[counter] for trial in trials}) == 1, f"non-deterministic allocation counts: {label}/{name}")
                    data["benchmarks"][label][name] = {**trials[0], "median_ns": statistics.median(t["median_ns"] for t in trials), "trials": trials}
            require(data["benchmarks"]["candidate"]["geometry/epa_no_trace/4"]["alloc_calls"] <= 4,
                    "EPA no-trace allocation sentinel regressed")
            (output / "results.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            summarize(output, data)
            print((output / "summary.md").read_text(), flush=True)
        finally:
            if managed_worktree:
                subprocess.run(["git", "worktree", "remove", "--force", str(baseline)], cwd=ROOT, check=False)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.SubprocessError, KeyError, ValueError) as error:
        raise SystemExit(f"audit evidence failed: {error}") from error
