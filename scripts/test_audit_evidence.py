#!/usr/bin/env python3
"""Fail-closed parsing tests for the pinned native audit evidence contract."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("audit_evidence", Path(__file__).with_name("audit-evidence.py"))
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class NativeEvidenceTests(unittest.TestCase):
    def results(self) -> str:
        return "\n".join(f"test {name} ... ok" for name in sorted(audit.CONTROLS | audit.EXPECTED_FAILURES))

    def test_native_results_require_every_known_case(self):
        self.assertEqual(len(audit.parse_tests(self.results())), 12)
        with self.assertRaisesRegex(RuntimeError, "inventory"):
            audit.parse_tests("\n".join(self.results().splitlines()[1:]))

    def test_compiler_failure_is_not_a_regression_reproduction(self):
        with self.assertRaisesRegex(RuntimeError, "compile errors"):
            audit.parse_tests("error[E0432]: unresolved import\nerror: could not compile")

    def test_duplicate_test_results_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            audit.parse_tests(self.results() + "\n" + self.results().splitlines()[0])

    def test_native_failures_remain_distinct_from_passes(self):
        text = self.results().replace(" ... ok", " ... FAILED")
        self.assertEqual(set(audit.parse_tests(text).values()), {"FAILED"})

    def test_probes_preserve_statuses_witnesses_and_overflow(self):
        text = 'AUDIT\tsmall\tdistance\t0.0\nAUDIT\tsmall\toverlaps\ttrue\nAUDIT\tgjk\tstatus\tIntersecting\nAUDIT\tmerge\tvariance\tinf'
        rows = audit.parse_probes(text)
        self.assertEqual(rows["small"], {"distance": 0.0, "overlaps": True})
        self.assertEqual(rows["merge"]["variance"], "inf")
        self.assertEqual(rows["gjk"]["status"], "Intersecting")

    def test_missing_or_duplicate_probes_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "no native"):
            audit.parse_probes("benchmark list only")
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            audit.parse_probes("AUDIT\ta\tb\t1\nAUDIT\ta\tb\t2")

    def test_divan_tree_uses_per_call_alloc_not_max_alloc(self):
        text = '''geometry_audit   fastest │ slowest │ median │ mean │ samples │ iters
├─ epa_no_trace          │         │        │      │         │
│  ├─ 4         200 ns   │ 400 ns  │ 300 ns │ 310 ns │ 200   │ 12800
│  │  max alloc:        │         │        │      │         │
│  │    0.015          │ 0.015   │ 0.015  │ 0.015│         │
│  │  alloc:            │         │        │      │         │
│  │    4               │ 4       │ 4      │ 4    │         │
│  │  grow:             │         │        │      │         │
│  │    1               │ 1       │ 1      │ 1    │         │
│  ╰─ 16        1 µs    │ 2 µs    │ 1.5 µs │ 1.4 µs │ 200 │ 12800
╰─ direct       1 ns    │ 3 ns    │ 2 ns   │ 2 ns │ 200     │ 12800'''
        rows = audit.parse_divan(text)
        self.assertEqual(set(rows), {"epa_no_trace/4", "epa_no_trace/16", "direct"})
        self.assertEqual(rows["epa_no_trace/4"], {"median_ns": 300.0, "alloc_calls": 4.0, "grow_calls": 1.0, "shrink_calls": 0.0})
        self.assertEqual(rows["epa_no_trace/16"]["median_ns"], 1500.0)
        self.assertEqual(rows["direct"]["alloc_calls"], 0.0)

    def test_listing_without_bench_flag_is_not_timing_evidence(self):
        with self.assertRaisesRegex(RuntimeError, "requires --bench"):
            audit.parse_divan("geometry_audit\n╰─ epa_no_trace\n   ╰─ 4")

    def test_malformed_timing_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            audit.parse_divan("╰─ direct 1 ns │ 2 ns │ 1 ns")

    def test_inventory_cannot_silently_drop_a_workload_from_both_revisions(self):
        audit.validate_inventory(dict.fromkeys(audit.BENCHMARKS))
        with self.assertRaisesRegex(RuntimeError, "27 benchmark"):
            audit.validate_inventory(dict.fromkeys(audit.BENCHMARKS - {"geometry/epa_trace/64"}))

    def test_external_harness_uses_identical_current_test_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = audit.write_manifest(root / "base-build", root / "base-source", root / "candidate-source").read_text()
            candidate = audit.write_manifest(root / "candidate-build", root / "candidate-source", root / "candidate-source").read_text()
            self.assertIn(str(root / "base-source/crates/geometry-kernels"), base)
            for text in [base, candidate]:
                self.assertIn(str(root / "candidate-source/crates/geometry-kernels/tests/audit_regressions.rs"), text)
                self.assertIn('divan = "=0.1.21"', text)

    def test_duration_units_are_explicit(self):
        self.assertEqual(audit.duration_ns("200 ps"), 0.2)
        self.assertEqual(audit.duration_ns("2 µs"), 2000.0)
        with self.assertRaisesRegex(RuntimeError, "unsupported"):
            audit.duration_ns("2 ticks")


class BoundaryEvidenceTests(unittest.TestCase):
    def test_boundary_inventory_requires_every_known_case(self):
        suite = audit.SUITES["boundary"]
        expected = suite["failures"] | suite["controls"]
        text = "\n".join(f"test {name} ... ok" for name in sorted(expected))
        self.assertEqual(len(audit.parse_tests(text, "boundary")), 11)
        with self.assertRaisesRegex(RuntimeError, "inventory"):
            audit.parse_tests("\n".join(text.splitlines()[1:]), "boundary")
        with self.assertRaisesRegex(RuntimeError, "compile errors"):
            audit.parse_tests("error: missing spatial_kernels dependency", "boundary")

    def test_boundary_workloads_cannot_disappear_from_both_revisions(self):
        expected = audit.SUITES["boundary"]["benchmarks"]
        self.assertEqual(len(expected), 26)
        audit.validate_inventory(dict.fromkeys(expected), "boundary")
        with self.assertRaisesRegex(RuntimeError, "26 benchmark"):
            audit.validate_inventory(dict.fromkeys(expected - {"octree/detect_pairs/256"}), "boundary")

    def test_nan_failure_is_serializable_evidence_not_zero(self):
        import json
        probes = audit.parse_probes("AUDIT\tpush\tmean\tNaN\nAUDIT\tpush\tvariance\tinf")
        self.assertEqual(probes["push"], {"mean": "NaN", "variance": "inf"})
        json.dumps(probes, allow_nan=False)

    def test_boundary_manifest_includes_public_spatial_types(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = audit.write_manifest(root / "build", root / "source", root / "candidate", "boundary").read_text()
            self.assertIn(str(root / "source/crates/spatial-kernels"), manifest)
            self.assertIn(str(root / "candidate/crates/graph-kernels/tests/boundary_regressions.rs"), manifest)
            self.assertNotIn("audit_regressions.rs", manifest)

    def test_boundary_allocation_gates_reject_eager_and_copy_regressions(self):
        rows = {"search/empty_limit/65536": {"alloc_calls": 0},
                "octree/detect_pairs/256": {"alloc_calls": 239}}
        audit.validate_allocations(rows, "boundary")
        rows["search/empty_limit/65536"]["alloc_calls"] = 1
        with self.assertRaisesRegex(RuntimeError, "empty top-k"):
            audit.validate_allocations(rows, "boundary")
        rows["search/empty_limit/65536"]["alloc_calls"] = 0
        rows["octree/detect_pairs/256"]["alloc_calls"] = 240
        with self.assertRaisesRegex(RuntimeError, "octree"):
            audit.validate_allocations(rows, "boundary")


if __name__ == "__main__":
    unittest.main()
